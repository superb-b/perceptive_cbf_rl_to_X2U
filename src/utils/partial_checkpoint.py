from __future__ import annotations

import os
from typing import Any

import torch


def _get_actor_critic(runner):
    """Support common RSL-RL runner layouts."""
    if hasattr(runner, "alg") and hasattr(runner.alg, "actor_critic"):
        return runner.alg.actor_critic

    if hasattr(runner, "actor_critic"):
        return runner.actor_critic

    raise AttributeError(
        "Cannot find actor_critic in runner. "
        "Expected runner.alg.actor_critic or runner.actor_critic."
    )


def _extract_model_state(checkpoint: Any) -> dict[str, torch.Tensor]:
    """Extract model state_dict from common RSL-RL checkpoint formats."""

    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Checkpoint must be dict, got {type(checkpoint)}"
        )

    candidates = (
        "model_state_dict",
        "model",
        "state_dict",
    )

    for key in candidates:
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value

    # Some checkpoints are directly a state_dict.
    if checkpoint and all(
        isinstance(v, torch.Tensor)
        for v in checkpoint.values()
    ):
        return checkpoint

    raise KeyError(
        "Cannot find model state_dict in checkpoint. "
        f"Top-level keys: {list(checkpoint.keys())}"
    )


def load_locomotion_into_dodge(
    runner,
    checkpoint_path: str,
    extra_actor_obs: int = 6,
    verbose: bool = True,
):
    """Initialize X2 dodge policy from a trained X2 locomotion policy.

    Expected use case
    -----------------
    Old locomotion actor:
        408 obs -> ...

    New dodge actor:
        414 obs -> ...

    The first-layer input is expanded by 6 dimensions corresponding to:
        ball relative position: 3
        ball relative velocity: 3

    Loading rules
    -------------
    1. Same name + same shape:
       copy exactly.

    2. 2-D Linear weight whose input dimension increased by exactly
       `extra_actor_obs`:
       copy old columns into the beginning and initialize new columns to zero.

       Example:
           old: [512, 408]
           new: [512, 414]

           new[:, :408] = old
           new[:, 408:] = 0

    3. Other shape mismatches:
       keep the newly initialized parameter.

    Optimizer state, learning iteration and scheduler are NOT restored.
    This is transfer learning, not resume.
    """

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    print("\n" + "=" * 88)
    print("[X2 transfer] locomotion -> dodge")
    print(f"[X2 transfer] checkpoint: {checkpoint_path}")
    print("=" * 88)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    old_state = _extract_model_state(checkpoint)

    model = _get_actor_critic(runner)
    new_state = model.state_dict()

    exact_loaded = []
    expanded_loaded = []
    skipped_shape = []
    missing_old = []

    for name, new_tensor in new_state.items():

        if name not in old_state:
            missing_old.append(
                (name, tuple(new_tensor.shape))
            )
            continue

        old_tensor = old_state[name]

        # ------------------------------------------------------------
        # 1. Exact match
        # ------------------------------------------------------------
        if old_tensor.shape == new_tensor.shape:
            new_state[name] = old_tensor.to(
                device=new_tensor.device,
                dtype=new_tensor.dtype,
            )

            exact_loaded.append(
                (name, tuple(old_tensor.shape))
            )
            continue

        # ------------------------------------------------------------
        # 2. Expanded Linear input:
        #
        # old [out, N]
        # new [out, N + 6]
        #
        # This covers actor 408 -> 414.
        #
        # It may also cover critic if the critic receives the same
        # extra ball observation.
        # ------------------------------------------------------------
        is_input_expansion = (
            old_tensor.ndim == 2
            and new_tensor.ndim == 2
            and old_tensor.shape[0] == new_tensor.shape[0]
            and new_tensor.shape[1]
            == old_tensor.shape[1] + extra_actor_obs
        )

        if is_input_expansion:
            patched = new_tensor.clone()

            old_in = old_tensor.shape[1]

            patched[:, :old_in] = old_tensor.to(
                device=patched.device,
                dtype=patched.dtype,
            )

            # Important:
            # ball observation initially has zero influence.
            patched[:, old_in:] = 0.0

            new_state[name] = patched

            expanded_loaded.append(
                (
                    name,
                    tuple(old_tensor.shape),
                    tuple(new_tensor.shape),
                )
            )
            continue

        # ------------------------------------------------------------
        # 3. Anything else:
        # leave new initialization untouched.
        # ------------------------------------------------------------
        skipped_shape.append(
            (
                name,
                tuple(old_tensor.shape),
                tuple(new_tensor.shape),
            )
        )

    model.load_state_dict(
        new_state,
        strict=True,
    )

    if verbose:
        print(
            f"[X2 transfer] exact loaded     : "
            f"{len(exact_loaded)}"
        )
        print(
            f"[X2 transfer] expanded loaded  : "
            f"{len(expanded_loaded)}"
        )
        print(
            f"[X2 transfer] shape skipped    : "
            f"{len(skipped_shape)}"
        )
        print(
            f"[X2 transfer] new-only params  : "
            f"{len(missing_old)}"
        )

        if expanded_loaded:
            print("\n[X2 transfer] expanded layers:")
            for name, old_shape, new_shape in expanded_loaded:
                print(
                    f"  {name}: "
                    f"{old_shape} -> {new_shape}"
                )

        if skipped_shape:
            print("\n[X2 transfer] skipped mismatches:")
            for name, old_shape, new_shape in skipped_shape:
                print(
                    f"  {name}: "
                    f"{old_shape} -> {new_shape}"
                )

        if missing_old:
            print("\n[X2 transfer] parameters only in new model:")
            for name, shape in missing_old:
                print(
                    f"  {name}: {shape}"
                )

    print("\n[X2 transfer] model initialized successfully.")
    print("[X2 transfer] optimizer NOT restored.")
    print("[X2 transfer] iteration NOT restored.")
    print("=" * 88 + "\n")

    return {
        "exact": exact_loaded,
        "expanded": expanded_loaded,
        "skipped": skipped_shape,
        "missing": missing_old,
    }