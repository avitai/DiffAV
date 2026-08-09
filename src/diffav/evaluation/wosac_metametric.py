"""WOSAC-style histogram log-likelihood realism metametric.

Scores simulated rollouts by estimating each feature's distribution across
rollouts and evaluating the logged scenario's feature values under it,
following the official 2025 sim-agents scoring flow. Two documented
divergences from the official metric:

- ``distance_to_nearest_object`` and ``collision_per_step`` use
  centre-to-centre distances (the trajectory state carries no box
  dimensions). The proxy is applied identically to log and sim features,
  so the distribution comparison remains internally consistent; only
  comparability with official box-aware numbers is lost.
- ``time_to_collision`` and ``traffic_light_violation`` are not computed
  (they need box geometry and traffic-light data respectively). The
  metametric is the reference-formula weighted sum over the eight active
  features (attainable maximum ``active_weight_sum`` = 0.85);
  ``normalized_metametric`` divides by that sum for [0, 1] comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import jax
import jax.numpy as jnp

from diffav.core.constants import (
    ROAD_EDGE_Z_STRETCH,
    WOSAC_2025_METAMETRIC_CONFIG,
    WosacMetametricConfig,
)
from diffav.core.geometry import RoadEdges, signed_distance_to_polylines
from diffav.evaluation.estimators import (
    log_likelihood_estimate_scenario_level,
    log_likelihood_estimate_timeseries,
)
from diffav.physics.kinematics import (
    compute_kinematic_features,
    compute_kinematic_validity,
)


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True, kw_only=True)
class MetametricFeatures:
    """Per-rollout metametric features for one scenario.

    Registered as a JAX pytree. The same container holds logged features
    (``num_rollouts`` = 1) and simulated features; ``valid`` is consulted
    on the log side only (simulated rollouts are always fully valid).

    Attributes:
        linear_speed: ``(num_rollouts, num_agents, num_steps)``.
        linear_acceleration: Same shape.
        angular_speed: Same shape.
        angular_acceleration: Same shape.
        distance_to_nearest_object: Centre-to-centre distance to the
            nearest other agent, same shape.
        collision_per_step: Boolean centre-distance collisions, same shape.
        distance_to_road_edge: Signed distance (positive off-road), same
            shape.
        offroad_per_step: Boolean off-road indications, same shape.
        valid: Logged per-step validity, ``(num_agents, num_steps)``.
    """

    linear_speed: jax.Array
    linear_acceleration: jax.Array
    angular_speed: jax.Array
    angular_acceleration: jax.Array
    distance_to_nearest_object: jax.Array
    collision_per_step: jax.Array
    distance_to_road_edge: jax.Array
    offroad_per_step: jax.Array
    valid: jax.Array


@dataclass(frozen=True, slots=True, kw_only=True)
class WosacMetametricResult:
    """Scores produced by :class:`WosacMetametric`.

    Attributes:
        per_feature: Likelihood in [0, 1] per active feature name.
        kinematic_score: Weight-normalized kinematic bucket score.
        interactive_score: Weight-normalized interactive bucket score
            (distance-to-nearest-object and collision features).
        map_score: Weight-normalized map bucket score (road-edge distance
            and offroad features).
        metametric: Reference-formula weighted sum over the active
            features; attainable maximum is ``active_weight_sum``.
        normalized_metametric: ``metametric / active_weight_sum``, in [0, 1].
        active_weight_sum: Total weight of the active features (0.85 for
            the 2025 configuration without time-to-collision and
            traffic-light features).
    """

    per_feature: dict[str, float]
    kinematic_score: float
    interactive_score: float
    map_score: float
    metametric: float
    normalized_metametric: float
    active_weight_sum: float


def compute_metametric_features(
    trajectories: jax.Array,
    valid: jax.Array,
    road_edges: RoadEdges,
    collision_threshold: float = 1.0,
    offroad_chunk_size: int | None = None,
) -> MetametricFeatures:
    """Assemble metametric features from rollout trajectories.

    Args:
        trajectories: Rollouts, shape ``(num_rollouts, num_agents,
            num_steps, 4)`` with state ``[x, y, heading, speed]``. Speeds
            and accelerations are derived from positions and headings via
            the WOSAC central differences at the scenario step rate.
        valid: Logged per-step validity, ``(num_agents, num_steps)``.
        road_edges: Oriented road-edge polylines.
        collision_threshold: Centre-to-centre distance (metres) below
            which two agents count as colliding.
        offroad_chunk_size: Optional polyline-chunk size bounding peak memory
            of the road-edge distance search (see
            :func:`~diffav.core.geometry.signed_distance_to_polylines`).
            The result is independent of it; set it (e.g. 8) when evaluating
            full-resolution road edges over many rollouts would otherwise
            exceed device memory. ``None`` (default) evaluates in one pass.

    Returns:
        The assembled :class:`MetametricFeatures`.
    """
    num_rollouts, num_agents, num_steps = trajectories.shape[:3]
    x = trajectories[..., 0]
    y = trajectories[..., 1]
    z = jnp.zeros_like(x)
    heading = trajectories[..., 2]
    linear_speed, linear_acceleration, angular_speed, angular_acceleration = (
        compute_kinematic_features(x, y, z, heading)
    )

    # Centre-to-centre nearest-object distances per rollout.
    xy = trajectories[..., :2]  # (K, N, T, 2)
    pairwise = jnp.linalg.norm(
        xy[:, :, jnp.newaxis, :, :] - xy[:, jnp.newaxis, :, :, :], axis=-1
    )  # (K, N, N, T)
    self_mask = jnp.eye(num_agents, dtype=bool)[jnp.newaxis, :, :, jnp.newaxis]
    # cast: the jax 0.9 stubs annotate three-argument jnp.where with the
    # one-argument tuple return included.
    pairwise = cast(jax.Array, jnp.where(self_mask, jnp.inf, pairwise))
    distance_to_nearest_object = jnp.min(pairwise, axis=2)  # (K, N, T)
    collision_per_step = distance_to_nearest_object < collision_threshold

    # Signed distance to road edges (z = 0 queries).
    points = jnp.concatenate(
        [xy.reshape(-1, 2), jnp.zeros((num_rollouts * num_agents * num_steps, 1))],
        axis=-1,
    )
    distance_to_road_edge = signed_distance_to_polylines(
        points,
        road_edges.polylines,
        road_edges.is_cyclic,
        z_stretch=ROAD_EDGE_Z_STRETCH,
        chunk_size=offroad_chunk_size,
    ).reshape(num_rollouts, num_agents, num_steps)
    offroad_per_step = distance_to_road_edge > 0.0

    return MetametricFeatures(
        linear_speed=linear_speed,
        linear_acceleration=linear_acceleration,
        angular_speed=angular_speed,
        angular_acceleration=angular_acceleration,
        distance_to_nearest_object=distance_to_nearest_object,
        collision_per_step=collision_per_step,
        distance_to_road_edge=distance_to_road_edge,
        offroad_per_step=offroad_per_step,
        valid=valid,
    )


def _masked_mean(values: jax.Array, validity: jax.Array) -> jax.Array:
    """Mean of ``values`` over the True entries of ``validity``."""
    total = jnp.sum(jnp.where(validity, values, 0.0))
    return total / jnp.maximum(jnp.sum(validity.astype(jnp.float32)), 1.0)


class WosacMetametric:
    """WOSAC-style realism metametric over simulated rollouts.

    Follows the official scoring flow: per-feature log-likelihood
    estimation of the logged scenario under the simulated distribution,
    validity-masked averaging, exponentiation to per-feature likelihoods,
    and the weighted metametric. See the module docstring for the two
    documented divergences from the official metric.

    Args:
        config: Per-feature estimator specs; defaults to the 2025
            challenge configuration.
    """

    _KINEMATIC = (
        "linear_speed",
        "linear_acceleration",
        "angular_speed",
        "angular_acceleration",
    )
    _INTERACTIVE = ("distance_to_nearest_object", "collision_indication")
    _MAP = ("distance_to_road_edge", "offroad_indication")

    def __init__(self, config: WosacMetametricConfig = WOSAC_2025_METAMETRIC_CONFIG) -> None:
        """Initialise with the metametric configuration."""
        self._config = config

    def compute(
        self,
        log_features: MetametricFeatures,
        sim_features: MetametricFeatures,
    ) -> WosacMetametricResult:
        """Score simulated rollouts against the logged scenario.

        Args:
            log_features: Logged-scenario features (``num_rollouts`` = 1).
            sim_features: Simulated-rollout features.

        Returns:
            The per-feature, per-bucket, and combined scores.
        """
        config = self._config
        valid = log_features.valid
        speed_validity, acceleration_validity = compute_kinematic_validity(valid)

        def timeseries_likelihood(
            feature_name: str, log_array: jax.Array, sim_array: jax.Array, validity: jax.Array
        ) -> float:
            log_likelihood = log_likelihood_estimate_timeseries(
                getattr(config, feature_name), log_array[0], sim_array
            )
            return float(jnp.exp(_masked_mean(log_likelihood, validity)))

        def indication_likelihood(
            feature_name: str, log_per_step: jax.Array, sim_per_step: jax.Array
        ) -> float:
            log_indication = jnp.any(log_per_step & valid[jnp.newaxis], axis=-1)
            sim_indication = jnp.any(sim_per_step & valid[jnp.newaxis], axis=-1)
            score = log_likelihood_estimate_scenario_level(
                getattr(config, feature_name), log_indication[0], sim_indication
            )
            return float(jnp.exp(jnp.mean(score)))

        per_feature = {
            "linear_speed": timeseries_likelihood(
                "linear_speed",
                log_features.linear_speed,
                sim_features.linear_speed,
                speed_validity,
            ),
            "linear_acceleration": timeseries_likelihood(
                "linear_acceleration",
                log_features.linear_acceleration,
                sim_features.linear_acceleration,
                acceleration_validity,
            ),
            "angular_speed": timeseries_likelihood(
                "angular_speed",
                log_features.angular_speed,
                sim_features.angular_speed,
                speed_validity,
            ),
            "angular_acceleration": timeseries_likelihood(
                "angular_acceleration",
                log_features.angular_acceleration,
                sim_features.angular_acceleration,
                acceleration_validity,
            ),
            "distance_to_nearest_object": timeseries_likelihood(
                "distance_to_nearest_object",
                log_features.distance_to_nearest_object,
                sim_features.distance_to_nearest_object,
                valid,
            ),
            "collision_indication": indication_likelihood(
                "collision_indication",
                log_features.collision_per_step,
                sim_features.collision_per_step,
            ),
            "distance_to_road_edge": timeseries_likelihood(
                "distance_to_road_edge",
                log_features.distance_to_road_edge,
                sim_features.distance_to_road_edge,
                valid,
            ),
            "offroad_indication": indication_likelihood(
                "offroad_indication",
                log_features.offroad_per_step,
                sim_features.offroad_per_step,
            ),
        }

        def weight_of(feature_name: str) -> float:
            return getattr(config, feature_name).metametric_weight

        def bucket_score(feature_names: tuple[str, ...]) -> float:
            weighted = sum(weight_of(name) * per_feature[name] for name in feature_names)
            weight_sum = sum(weight_of(name) for name in feature_names)
            return weighted / weight_sum

        active_weight_sum = sum(weight_of(name) for name in per_feature)
        metametric = sum(weight_of(name) * score for name, score in per_feature.items())

        return WosacMetametricResult(
            per_feature=per_feature,
            kinematic_score=bucket_score(self._KINEMATIC),
            interactive_score=bucket_score(self._INTERACTIVE),
            map_score=bucket_score(self._MAP),
            metametric=metametric,
            normalized_metametric=metametric / active_weight_sum,
            active_weight_sum=active_weight_sum,
        )
