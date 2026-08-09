"""Weight-soup interpolation between a base model and a steering expert.

Steering strength as a post-training dial: instead of blending losses, the
base model and a reward-fine-tuned expert are interpolated in parameter
space, so one expert training run serves every steering strength.
"""

from __future__ import annotations

import jax
from flax import nnx

from diffav.models.trajectory_diffusion import TrajectoryDiffusionModel


def make_weight_soup(
    base_model: TrajectoryDiffusionModel,
    target_model: TrajectoryDiffusionModel,
    interpolation_weight: float,
) -> TrajectoryDiffusionModel:
    """Interpolate model parameters: ``θ = (1 − λ)·θ_base + λ·θ_target``.

    Only ``nnx.Param`` leaves are interpolated; all remaining state (e.g.
    RNG state, which cannot be averaged) is taken from ``base_model``. The
    inputs are left untouched — a new model is returned.

    Args:
        base_model: The unsteered reference model (λ = 0 endpoint).
        target_model: The reward-fine-tuned expert (λ = 1 endpoint).
        interpolation_weight: λ ∈ [0.0, 1.0].

    Returns:
        A new model carrying the interpolated parameters.

    Raises:
        ValueError: If ``interpolation_weight`` is outside [0.0, 1.0].
    """
    if not 0.0 <= interpolation_weight <= 1.0:
        msg = f"interpolation_weight must be in [0.0, 1.0], got {interpolation_weight}"
        raise ValueError(msg)

    graphdef, base_params, rest = nnx.split(base_model, nnx.Param, ...)
    target_params = nnx.state(target_model, nnx.Param)
    souped_params = jax.tree.map(
        lambda base_leaf, target_leaf: (
            (1.0 - interpolation_weight) * base_leaf + interpolation_weight * target_leaf
        ),
        base_params,
        target_params,
    )
    return nnx.merge(graphdef, souped_params, rest)
