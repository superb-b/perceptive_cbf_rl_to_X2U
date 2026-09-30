"""Script to train RL agent with RSL-RL."""

import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

import tyro

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import dump_yaml, get_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from src.utils.partial_checkpoint import (
    load_locomotion_into_dodge,
)

@dataclass(frozen=True)
class TrainConfig:
  env: ManagerBasedRlEnvCfg
  agent: RslRlBaseRunnerCfg
  motion_file: str | None = None
  video: bool = False
  video_length: int = 200
  video_interval: int = 2000
  enable_nan_guard: bool = False
  torchrunx_log_dir: str | None = None
  gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])
  pretrained_checkpoint: str | None = None

  @staticmethod
  def from_task(task_id: str) -> "TrainConfig":
    env_cfg = load_env_cfg(task_id)
    agent_cfg = load_rl_cfg(task_id)
    return TrainConfig(env=env_cfg, agent=agent_cfg)

import torch
from pathlib import Path


def _extract_model_state(checkpoint):
  if "model_state_dict" in checkpoint:
    return checkpoint["model_state_dict"]
  elif "model" in checkpoint:
    return checkpoint["model"]
  elif "state_dict" in checkpoint:
    return checkpoint["state_dict"]
  return checkpoint


def _load_module_partial(
    module,
    old_state,
    prefix_candidates,
    old_input_dim,
    new_input_dim,
    tag,
    action_dim=None,
):
  new_state = module.state_dict()

  exact = []
  expanded = []
  skipped = []
  missing = []

  def find_old_key(local_key):
    candidates = [local_key]

    for prefix in prefix_candidates:
      candidates.append(f"{prefix}.{local_key}")

    for key in candidates:
      if key in old_state:
        return key

    return None

  for new_key, new_tensor in new_state.items():
    old_key = find_old_key(new_key)

    if old_key is None:
      missing.append((new_key, tuple(new_tensor.shape)))
      continue

    old_tensor = old_state[old_key]

    # Same shape -> copy directly.
    if old_tensor.shape == new_tensor.shape:
      new_state[new_key] = old_tensor.to(
        device=new_tensor.device,
        dtype=new_tensor.dtype,
      )
      exact.append((new_key, tuple(old_tensor.shape)))
      continue

    # First input layer expanded, e.g. 408 -> 414.
    is_input_expansion = (
      old_input_dim is not None
      and new_input_dim is not None
      and old_tensor.ndim == 2
      and new_tensor.ndim == 2
      and old_tensor.shape[0] == new_tensor.shape[0]
      and old_tensor.shape[1] == old_input_dim
      and new_tensor.shape[1] == new_input_dim
    )

    if is_input_expansion:
      patched = new_tensor.clone()

      patched[:, :old_input_dim] = old_tensor.to(
        device=patched.device,
        dtype=patched.dtype,
      )

      # New ball-state input has zero influence initially.
      patched[:, old_input_dim:] = 0.0

      new_state[new_key] = patched

      expanded.append(
        (new_key, tuple(old_tensor.shape), tuple(new_tensor.shape))
      )
      continue

    skipped.append(
      (new_key, tuple(old_tensor.shape), tuple(new_tensor.shape))
    )

  module.load_state_dict(new_state, strict=True)

  print(f"\n[TRANSFER:{tag}] exact    : {len(exact)}")
  print(f"[TRANSFER:{tag}] expanded : {len(expanded)}")
  print(f"[TRANSFER:{tag}] skipped  : {len(skipped)}")
  print(f"[TRANSFER:{tag}] missing  : {len(missing)}")

  if expanded:
    print(f"[TRANSFER:{tag}] expanded layers:")
    for name, old_shape, new_shape in expanded:
      print(f"  {name}: {old_shape} -> {new_shape}")

  if skipped:
    print(f"[TRANSFER:{tag}] skipped mismatches:")
    for name, old_shape, new_shape in skipped:
      print(f"  {name}: {old_shape} -> {new_shape}")

  return {
    "exact": exact,
    "expanded": expanded,
    "skipped": skipped,
    "missing": missing,
  }


def load_x2_locomotion_into_dodge(
  runner,
  checkpoint_path: str | Path,
  actor_old_dim: int = 408,
  actor_new_dim: int = 414,
  critic_old_dim: int | None = 888,
  critic_new_dim: int | None = 894,
):
  checkpoint_path = Path(checkpoint_path).expanduser().resolve()

  print("=" * 80)
  print("[TRANSFER] X2 locomotion -> X2 dodge")
  print(f"[TRANSFER] checkpoint: {checkpoint_path}")

  checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=False,
  )

  print(
    "[TRANSFER] checkpoint top-level keys:",
    checkpoint.keys() if isinstance(checkpoint, dict) else type(checkpoint),
  )

  if "actor_state_dict" not in checkpoint:
    raise RuntimeError("checkpoint has no actor_state_dict")

  if "critic_state_dict" not in checkpoint:
      raise RuntimeError("checkpoint has no critic_state_dict")

  old_actor_state = checkpoint["actor_state_dict"]
  old_critic_state = checkpoint["critic_state_dict"]

  alg = runner.alg

  actor = alg.actor
  critic = alg.critic

  print("\n[TRANSFER] actor type:", type(actor))
  print("[TRANSFER] actor keys:")
  for key in list(actor.state_dict().keys())[:20]:
    print(" ", key)

  print("\n[TRANSFER] critic type:", type(critic))
  print("[TRANSFER] critic keys:")
  for key in list(critic.state_dict().keys())[:20]:
    print(" ", key)

  actor_stats = _load_module_partial(
      module=actor,
      old_state=old_actor_state,
      prefix_candidates=(
        "actor",
        "_raw_actor",
        "module.actor",
      ),
      old_input_dim=408,
      new_input_dim=414,
      tag="actor",
      action_dim=31,
  )

  critic_stats = _load_module_partial(
      module=critic,
      old_state=old_critic_state,
      prefix_candidates=(
        "critic",
        "_raw_critic",
        "module.critic",
      ),
      old_input_dim=critic_old_dim,
      new_input_dim=critic_new_dim,
      tag="critic",
      action_dim=None,
  )

  print("\n[TRANSFER] optimizer NOT restored")
  print("[TRANSFER] iteration NOT restored")
  print("=" * 80)

  return {
    "actor": actor_stats,
    "critic": critic_stats,
  }


def run_train(task_id: str, cfg: TrainConfig, log_dir: Path) -> None:
  cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
  if cuda_visible == "":
    device = "cpu"
    seed = cfg.agent.seed
    rank = 0
  else:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
    device = f"cuda:{local_rank}"
    seed = cfg.agent.seed + local_rank

  configure_torch_backends()

  cfg.agent.seed = seed
  cfg.env.seed = seed

  print(f"[INFO] Training with: device={device}, seed={seed}, rank={rank}")

  is_tracking_task = "motion" in cfg.env.commands and isinstance(
    cfg.env.commands["motion"], MotionCommandCfg
  )

  if is_tracking_task:
    if not cfg.motion_file:
      raise ValueError("For tracking tasks, --motion-file must be set ...")

    motion_path = Path(cfg.motion_file).expanduser().resolve()

    if not motion_path.exists():
      raise FileNotFoundError(f"Motion file not found: {motion_path}")

    motion_cmd = cfg.env.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)

    motion_cmd.motion_file = str(motion_path)

    print(f"[INFO] Using motion file: {motion_cmd.motion_file}")

    if motion_cmd.motion_file and Path(motion_cmd.motion_file).exists():
      print(f"[INFO] Using local motion file: {motion_cmd.motion_file}")

  if cfg.enable_nan_guard:
    cfg.env.sim.nan_guard.enabled = True
    print(
      f"[INFO] NaN guard enabled, "
      f"output dir: {cfg.env.sim.nan_guard.output_dir}"
    )

  if rank == 0:
    print(f"[INFO] Logging experiment in directory: {log_dir}")

  env = ManagerBasedRlEnv(
    cfg=cfg.env,
    device=device,
    render_mode="rgb_array" if cfg.video else None,
  )

  log_root_path = log_dir.parent

  # ----------------------------------------------------------------
  # Exact resume checkpoint.
  # ----------------------------------------------------------------
  resume_path: Path | None = None

  if cfg.agent.resume:
    resume_path = get_checkpoint_path(
      log_root_path,
      cfg.agent.load_run,
      cfg.agent.load_checkpoint,
    )

  # ----------------------------------------------------------------
  # IMPORTANT:
  # Do not use resume and pretrained transfer at the same time.
  # ----------------------------------------------------------------
  if cfg.agent.resume and cfg.pretrained_checkpoint is not None:
    raise ValueError(
      "Do not use --resume and --pretrained-checkpoint together. "
      "--resume is exact continuation; "
      "--pretrained-checkpoint is locomotion->dodge transfer."
    )

  if cfg.video and rank == 0:
    env = VideoRecorder(
      env,
      video_folder=Path(log_dir) / "videos" / "train",
      step_trigger=lambda step: step % cfg.video_interval == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )
    print("[INFO] Recording videos during training.")

  env = RslRlVecEnvWrapper(
    env,
    clip_actions=cfg.agent.clip_actions,
  )

  agent_cfg = asdict(cfg.agent)
  env_cfg = asdict(cfg.env)

  runner_cls = load_runner_cls(task_id)

  if runner_cls is None:
    runner_cls = MjlabOnPolicyRunner

  runner_kwargs = {}

  runner = runner_cls(
    env,
    agent_cfg,
    str(log_dir),
    device,
    **runner_kwargs,
  )
  print("\n===== RUNNER DEBUG =====")
  print("runner type:", type(runner))
  print("runner attrs:", vars(runner).keys())

  if hasattr(runner, "alg"):
    print("alg type:", type(runner.alg))
    print("alg attrs:", vars(runner.alg).keys())

  print("========================\n")


  runner.add_git_repo_to_log(__file__)

  # ================================================================
  # A. Exact same-task resume
  # ================================================================
  if resume_path is not None:
    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.load(str(resume_path))

  # ================================================================
  # B. X2 locomotion -> X2 dodge transfer
  # ================================================================
  elif cfg.pretrained_checkpoint is not None:

    if task_id != "X2-AMP-Dodge-State-Flat":
      print(
        f"[WARNING] pretrained checkpoint requested for task "
        f"{task_id!r}, not X2-AMP-Dodge-State-Flat."
      )

    load_x2_locomotion_into_dodge(
      runner=runner,
      checkpoint_path=cfg.pretrained_checkpoint,

      # Confirm these from startup logs.
      actor_old_dim=408,
      actor_new_dim=414,

      # If new critic stays 888 rather than 894,
      # set both of these to None.
      critic_old_dim=888,
      critic_new_dim=894,
    )

  if rank == 0:
    dump_yaml(
      log_dir / "params" / "env.yaml",
      env_cfg,
    )
    dump_yaml(
      log_dir / "params" / "agent.yaml",
      agent_cfg,
    )

  runner.learn(
    num_learning_iterations=cfg.agent.max_iterations,
    init_at_random_ep_len=True,
  )

  env.close()

def launch_training(task_id: str, args: TrainConfig | None = None):
  args = args or TrainConfig.from_task(task_id)

  # Create log directory once before launching workers.
  log_root_path = Path("logs") / "rsl_rl" / args.agent.experiment_name
  log_root_path.resolve()
  log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if args.agent.run_name:
    log_dir_name += f"_{args.agent.run_name}"
  log_dir = log_root_path / log_dir_name

  # Select GPUs based on CUDA_VISIBLE_DEVICES and user specification.
  selected_gpus, num_gpus = select_gpus(args.gpu_ids)

  # Set environment variables for all modes.
  if selected_gpus is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
  else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
  os.environ["MUJOCO_GL"] = "egl"

  if num_gpus <= 1:
    # CPU or single GPU: run directly without torchrunx.
    run_train(task_id, args, log_dir)
  else:
    # Multi-GPU: use torchrunx.
    import torchrunx

    # torchrunx redirects stdout to logging.
    logging.basicConfig(level=logging.INFO)

    # Configure torchrunx logging directory.
    # Priority: 1) existing env var, 2) user flag, 3) default to {log_dir}/torchrunx.
    if "TORCHRUNX_LOG_DIR" not in os.environ:
      if args.torchrunx_log_dir is not None:
        # User specified a value via flag (could be "" to disable).
        os.environ["TORCHRUNX_LOG_DIR"] = args.torchrunx_log_dir
      else:
        # Default: put logs in training directory.
        os.environ["TORCHRUNX_LOG_DIR"] = str(log_dir / "torchrunx")

    print(f"[INFO] Launching training with {num_gpus} GPUs", flush=True)
    torchrunx.Launcher(
      hostnames=["localhost"],
      workers_per_host=num_gpus,
      backend=None,  # Let rsl_rl handle process group initialization.
      copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*",),
    ).run(run_train, task_id, args, log_dir)


def main():
  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401
  import src.tasks

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  args = tyro.cli(
    TrainConfig,
    args=remaining_args,
    default=TrainConfig.from_task(chosen_task),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args

  launch_training(task_id=chosen_task, args=args)


if __name__ == "__main__":
  main()
