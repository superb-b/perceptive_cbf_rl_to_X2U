"""AMP NPZ motion loader.

Pre-computes per-frame anchor-relative body pose/velocity tensors from a single
NPZ or a directory of NPZs, then yields (state, next_state) batches for
discriminator training.

Ported from the previous vendored rsl_rl fork (rsl_rl/utils/motion_loader.py).
The per-frame features (anchor-relative position, 6D orientation, body-frame
linear/angular velocity) match the env's ``amp`` observation group exactly, so
the discriminator compares like-with-like.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from collections.abc import Sequence

import numpy as np
import torch
from tqdm import tqdm

import mjlab.utils.lab_api.math as math_utils


def _read_amp_sampling_manifest(manifest_file):
  """Return clips and probabilities; group mass is independent of clip counts."""
  manifest_path = Path(manifest_file).resolve()
  spec = json.loads(manifest_path.read_text())
  if spec.get("format") != "amp_group_sampling_v1":
    raise ValueError("Unsupported AMP sampling manifest format")
  groups = spec.get("groups", [])
  if not groups:
    raise ValueError("AMP sampling manifest has no groups")
  files, names, probabilities, labels = [], [], [], []
  seen_files, seen_labels = set(), set()
  group_mass = []
  for group in groups:
    label = group["name"]
    probability = float(group["probability"])
    if label in seen_labels or not np.isfinite(probability) or probability <= 0:
      raise ValueError("Group names must be unique and probabilities positive/finite")
    seen_labels.add(label)
    directory = (manifest_path.parent / group["directory"]).resolve()
    paths = []
    for root, _, entries in os.walk(directory):
      paths.extend(os.path.join(root, name) for name in entries if name.endswith(".npz"))
    paths.sort()
    if not paths:
      raise ValueError(f"No NPZ files in AMP group {label}: {directory}")
    for path in paths:
      canonical = str(Path(path).resolve())
      if canonical in seen_files:
        raise ValueError(f"Duplicate clip in AMP manifest: {canonical}")
      seen_files.add(canonical)
      files.append(path)
      names.append(Path(path).stem)
      probabilities.append(probability / len(paths))
      labels.append(label)
    group_mass.append(probability)
    print(f"[AMP MIX] {label}: probability={probability:.3f}, clips={len(paths)}")
  if not np.isclose(sum(group_mass), 1.0, rtol=0.0, atol=1e-8):
    raise ValueError("AMP group probabilities must sum to 1")
  return files, names, probabilities, labels


class AMPLoader:
  """Holds expert motion clips in body-local form for AMP discriminator training."""

  def __init__(
    self,
    motion_file: str,
    body_names: Sequence[str],
    anchor_name: str,
    all_body_names: Sequence[str],
    device: str = "cuda:0",
  ) -> None:
    """Args:
      motion_file: Path to a single .npz or a directory of .npz files.
      body_names: Tracked body names (used for AMP obs).
      anchor_name: Anchor body (typically root/torso) for the body-local frame.
      all_body_names: Ordered list of *all* body names in the model. Indexes
        each body in the NPZs by its position here.
      device: Torch device.
    """
    assert os.path.exists(motion_file), f"Invalid path: {motion_file}"

    all_names = list(all_body_names)
    self._body_indexes = [all_names.index(n) for n in body_names]
    self._anchor_indexes = all_names.index(anchor_name)
    self._num_bodies = len(self._body_indexes)

    # Opt-in manifest mode; existing NPZ/directory tasks keep their old behavior.
    self._amp_sampling_probabilities = None
    self._amp_group_labels = None
    if os.path.isfile(motion_file) and str(motion_file).endswith(".json"):
      (motion_files, motion_names, self._amp_sampling_probabilities,
       self._amp_group_labels) = _read_amp_sampling_manifest(motion_file)
    elif os.path.isfile(motion_file):
      motion_files = [motion_file]
      motion_names = [os.path.splitext(os.path.basename(motion_file))[0]]
    elif os.path.isdir(motion_file):
      motion_names = []
      motion_files = []
      for root, _dirs, files in os.walk(motion_file):
        for filename in sorted(files):
          if filename.endswith(".npz"):
            motion_names.append(os.path.splitext(filename)[0])
            motion_files.append(os.path.join(root, filename))
      if motion_files:
        motion_files, motion_names = zip(*sorted(zip(motion_files, motion_names)))
        motion_files = list(motion_files)
        motion_names = list(motion_names)
      assert motion_files, f"No npz files found in directory: {motion_file}"
    else:
      raise ValueError(f"Path is neither a file nor a directory: {motion_file}")

    self.motion_names = motion_names
    self._body_pos_b_list: list[torch.Tensor] = []
    self._body_quat_b_list: list[torch.Tensor] = []
    self._body_ori_b_list: list[torch.Tensor] = []
    self._body_lin_vel_b_list: list[torch.Tensor] = []
    self._body_ang_vel_b_list: list[torch.Tensor] = []

    for motion_idx, (motion_name, motion_path) in enumerate(
      zip(motion_names, motion_files)
    ):
      print(f"Processing motion {motion_idx + 1}/{len(motion_files)}: {motion_name}")
      data = np.load(motion_path)
      if motion_idx == 0:
        self.fps = float(np.asarray(data["fps"]).item())

      body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
      body_quat_w = torch.tensor(
        data["body_quat_w"], dtype=torch.float32, device=device
      )
      body_lin_vel_w = torch.tensor(
        data["body_lin_vel_w"], dtype=torch.float32, device=device
      )
      body_ang_vel_w = torch.tensor(
        data["body_ang_vel_w"], dtype=torch.float32, device=device
      )

      n_steps = body_pos_w.shape[0]
      pos_b = torch.zeros(
        (n_steps, self._num_bodies, 3), dtype=torch.float32, device=device
      )
      quat_b = torch.zeros(
        (n_steps, self._num_bodies, 4), dtype=torch.float32, device=device
      )
      ori_b = torch.zeros(
        (n_steps, self._num_bodies, 6), dtype=torch.float32, device=device
      )
      lin_b = torch.zeros(
        (n_steps, self._num_bodies, 3), dtype=torch.float32, device=device
      )
      ang_b = torch.zeros(
        (n_steps, self._num_bodies, 3), dtype=torch.float32, device=device
      )

      for frame_idx in tqdm(
        range(n_steps), desc=f"Preloading AMP data for {motion_name}"
      ):
        anchor_pos_w = (
          body_pos_w[frame_idx, self._anchor_indexes]
          .unsqueeze(0)
          .repeat(self._num_bodies, 1)
        )
        anchor_quat_w = (
          body_quat_w[frame_idx, self._anchor_indexes]
          .unsqueeze(0)
          .repeat(self._num_bodies, 1)
        )
        bp_w = body_pos_w[frame_idx, self._body_indexes]
        bq_w = body_quat_w[frame_idx, self._body_indexes]
        blv_w = body_lin_vel_w[frame_idx, self._body_indexes]
        bav_w = body_ang_vel_w[frame_idx, self._body_indexes]

        bp_b, bq_b = math_utils.subtract_frame_transforms(
          anchor_pos_w, anchor_quat_w, bp_w, bq_w
        )
        mat = math_utils.matrix_from_quat(bq_b)
        # First two columns of the rotation matrix -> 6D continuous rotation.
        ori_6 = mat[..., :, :2].reshape(self._num_bodies, 6)
        lin_b_frame = math_utils.quat_apply_inverse(bq_w, blv_w)
        ang_b_frame = math_utils.quat_apply_inverse(bq_w, bav_w)

        pos_b[frame_idx] = bp_b
        quat_b[frame_idx] = bq_b
        ori_b[frame_idx] = ori_6
        lin_b[frame_idx] = lin_b_frame
        ang_b[frame_idx] = ang_b_frame

      self._body_pos_b_list.append(pos_b)
      self._body_quat_b_list.append(quat_b)
      self._body_ori_b_list.append(ori_b)
      self._body_lin_vel_b_list.append(lin_b)
      self._body_ang_vel_b_list.append(ang_b)

    self.time_step_total = self._body_pos_b_list[0].shape[0]
    self.motion_total_time = self.time_step_total / self.fps
    if self._amp_sampling_probabilities is not None:
      self._prepare_weighted_sampling()

  @property
  def observation_dim(self) -> int:
    return (3 + 6 + 3 + 3) * self._num_bodies

  def feed_forward_generator(self, num_mini_batch: int, mini_batch_size: int):
    if self._amp_sampling_probabilities is not None:
      yield from self._weighted_feed_forward_generator(num_mini_batch, mini_batch_size)
      return
    num_motions = len(self._body_pos_b_list)
    for batch_idx in range(num_mini_batch):
      motion_idx = batch_idx % num_motions
      pos = self._body_pos_b_list[motion_idx]
      ori = self._body_ori_b_list[motion_idx]
      lin = self._body_lin_vel_b_list[motion_idx]
      ang = self._body_ang_vel_b_list[motion_idx]
      n = pos.shape[0]

      idx = torch.randint(0, n, (mini_batch_size,), device=pos.device).clamp(max=n - 1)
      next_idx = (idx + 1).clamp(max=n - 1)

      def _flatten(p, o, l, a, sel):
        return torch.cat(
          [
            p[sel].reshape(mini_batch_size, -1),
            o[sel].reshape(mini_batch_size, -1),
            l[sel].reshape(mini_batch_size, -1),
            a[sel].reshape(mini_batch_size, -1),
          ],
          dim=-1,
        )

      yield _flatten(pos, ori, lin, ang, idx), _flatten(pos, ori, lin, ang, next_idx)

  def _prepare_weighted_sampling(self):
    """Flatten once for efficient vectorized, per-pair sampling on the GPU."""
    lengths = [x.shape[0] for x in self._body_pos_b_list]
    if any(n < 2 for n in lengths):
      raise ValueError("AMP mixture requires at least two frames in every clip")
    device = self._body_pos_b_list[0].device
    frames = []
    for i, n in enumerate(lengths):
      frames.append(torch.cat([
        self._body_pos_b_list[i].reshape(n, -1),
        self._body_ori_b_list[i].reshape(n, -1),
        self._body_lin_vel_b_list[i].reshape(n, -1),
        self._body_ang_vel_b_list[i].reshape(n, -1),
      ], dim=-1))
    self._amp_flat_frames = torch.cat(frames, dim=0)
    lengths_t = torch.tensor(lengths, dtype=torch.long, device=device)
    self._amp_pair_counts = lengths_t - 1
    self._amp_offsets = torch.cumsum(lengths_t, dim=0) - lengths_t
    self._amp_clip_probs = torch.tensor(
      self._amp_sampling_probabilities, dtype=torch.float32, device=device
    )
    self._amp_clip_probs /= self._amp_clip_probs.sum()
    self._amp_sample_counts = torch.zeros(len(lengths), dtype=torch.long, device=device)
    self._amp_group_indices = {
      label: torch.tensor([i for i, x in enumerate(self._amp_group_labels) if x == label],
                          dtype=torch.long, device=device)
      for label in dict.fromkeys(self._amp_group_labels)
    }
    self._amp_generator_calls = 0

  def _weighted_feed_forward_generator(self, num_mini_batch, mini_batch_size):
    """Group probability -> uniform clip -> uniform valid adjacent frame pair.

    70/30 is a probability per expert pair, not a promised exact split in each
    finite batch. All clips remain eligible even if num_mini_batch is small.
    Terminal frames are not sampled as self-transitions or across clip edges.
    """
    self._amp_generator_calls += 1
    device = self._amp_flat_frames.device
    for batch_idx in range(num_mini_batch):
      clip_ids = torch.multinomial(self._amp_clip_probs, mini_batch_size, replacement=True)
      counts = self._amp_pair_counts[clip_ids]
      local_idx = (torch.rand(mini_batch_size, device=device) * counts).long()
      local_idx = torch.minimum(local_idx, counts - 1)
      idx = self._amp_offsets[clip_ids] + local_idx
      self._amp_sample_counts += torch.bincount(clip_ids, minlength=len(self.motion_names))
      if batch_idx == 0 and (self._amp_generator_calls == 1 or self._amp_generator_calls % 200 == 0):
        total = int(self._amp_sample_counts.sum().item())
        report = ", ".join(
          f"{name}={self._amp_sample_counts[ids].sum().item() / max(total, 1):.3%}"
          for name, ids in self._amp_group_indices.items()
        )
        covered = int((self._amp_sample_counts > 0).sum().item())
        print(f"[AMP MIX SAMPLE] {report}; clips_seen={covered}/{len(self.motion_names)}; pairs={total}")
      yield self._amp_flat_frames[idx], self._amp_flat_frames[idx + 1]
