"""Tests for canonical WOMD/WOSAC constants.

Expected values mirror the official waymo-open-dataset references:
``utils/sim_agents/submission_specs.py`` (timing and rollout dims) and
``wdl_limited/sim_agents_metrics/challenge_2025_sim_agents_config.textproto``
(the kinematic-feature histogram support ranges).
"""

from __future__ import annotations

import pytest

from simulacrax.core.constants import (
    DEFAULT_NUM_BLOCKS,
    DEFAULT_NUM_SOCIAL_LAYERS,
    DEFAULT_NUM_TEMPORAL_LAYERS,
    FOG_EXTINCTION_COEFFICIENT,
    KinematicEnvelope,
    LIDAR_HIT_WEIGHT_THRESHOLD_FACTOR,
    MAX_ROAD_EDGE_POINTS,
    MAX_ROAD_EDGE_POLYLINES,
    MODALITY_EMBEDDING_KEYS,
    MODALITY_TOKEN_VALID_KEYS,
    NERF_CAMERA_HORIZONTAL_FOV_RAD,
    ROAD_EDGE_TYPES,
    ROADGRAPH_TYPE_ROAD_EDGE_BOUNDARY,
    ROADGRAPH_TYPE_ROAD_EDGE_MEDIAN,
    SCENARIO_ID,
    SCENE_BACKBONE_ARCHITECTURE_VERSION,
    SCENE_CONTEXT,
    SCENE_EMBEDDING,
    SCENE_TOKEN_VALID,
    STATE_ID,
    STATE_OBJECTS_OF_INTEREST,
    STATE_TIMESTAMP_MICROS,
    STATE_TRACKS_TO_PREDICT,
    STATE_TYPE,
    STATE_Z,
    WOD_CURRENT_TIME_INDEX,
    WOD_FULL_SCENARIO_STEPS,
    WOD_FUTURE_STEPS,
    WOD_HISTORY_STEPS,
    WOD_STEP_DURATION_SECONDS,
    WOSAC_2025_KINEMATIC_ENVELOPE,
    WOSAC_2025_METAMETRIC_CONFIG,
    WOSAC_N_ROLLOUTS,
)


class TestScenarioTiming:
    """WOMD scenario timing constants match the submission specs."""

    def test_step_duration_is_10hz(self) -> None:
        """Scenarios are sampled at 10 Hz."""
        assert WOD_STEP_DURATION_SECONDS == 0.1

    def test_current_time_index(self) -> None:
        """The current step is the 11th (index 10)."""
        assert WOD_CURRENT_TIME_INDEX == 10

    def test_history_covers_indices_through_current(self) -> None:
        """History spans indices 0..current inclusive."""
        assert WOD_HISTORY_STEPS == WOD_CURRENT_TIME_INDEX + 1
        assert WOD_HISTORY_STEPS == 11

    def test_future_steps_match_sim_agents_submission(self) -> None:
        """A sim-agents submission simulates 80 future steps."""
        assert WOD_FUTURE_STEPS == 80

    def test_full_scenario_is_history_plus_future(self) -> None:
        """The 91-step scenario decomposes into history + future."""
        assert WOD_FULL_SCENARIO_STEPS == WOD_HISTORY_STEPS + WOD_FUTURE_STEPS
        assert WOD_FULL_SCENARIO_STEPS == 91


class TestWosacDims:
    """WOSAC submission dimensions."""

    def test_n_rollouts(self) -> None:
        """A valid submission carries 32 parallel rollouts."""
        assert WOSAC_N_ROLLOUTS == 32


class TestRoadEdgeTypes:
    """Roadgraph road-edge type codes match the official mapping."""

    def test_boundary_code(self) -> None:
        """ROAD_EDGE_BOUNDARY is roadgraph type 15."""
        assert ROADGRAPH_TYPE_ROAD_EDGE_BOUNDARY == 15

    def test_median_code(self) -> None:
        """ROAD_EDGE_MEDIAN is roadgraph type 16."""
        assert ROADGRAPH_TYPE_ROAD_EDGE_MEDIAN == 16

    def test_road_edge_types_exclude_stop_signs(self) -> None:
        """The map metric uses boundary and median only (17 is STOP_SIGN)."""
        assert ROAD_EDGE_TYPES == (15, 16)
        assert 17 not in ROAD_EDGE_TYPES

    def test_fixed_shape_caps_cover_measured_distribution(self) -> None:
        """The jit-batched road-edge caps exceed the measured WOD maxima."""
        assert MAX_ROAD_EDGE_POLYLINES == 128
        assert MAX_ROAD_EDGE_POINTS == 768


class TestKinematicEnvelope:
    """2025 challenge kinematic histogram support ranges."""

    def test_is_frozen(self) -> None:
        """The envelope is an immutable value object."""
        with pytest.raises(AttributeError):
            WOSAC_2025_KINEMATIC_ENVELOPE.linear_speed_max = 1.0  # type: ignore[misc]

    def test_linear_speed_range(self) -> None:
        """linear_speed histogram spans [0, 25] m/s."""
        assert WOSAC_2025_KINEMATIC_ENVELOPE.linear_speed_min == 0.0
        assert WOSAC_2025_KINEMATIC_ENVELOPE.linear_speed_max == 25.0

    def test_linear_acceleration_range(self) -> None:
        """linear_acceleration histogram spans [-12, 12] m/s^2."""
        assert WOSAC_2025_KINEMATIC_ENVELOPE.linear_acceleration_min == -12.0
        assert WOSAC_2025_KINEMATIC_ENVELOPE.linear_acceleration_max == 12.0

    def test_angular_speed_range(self) -> None:
        """angular_speed histogram spans [-0.628, 0.628] rad/s."""
        assert WOSAC_2025_KINEMATIC_ENVELOPE.angular_speed_min == -0.628
        assert WOSAC_2025_KINEMATIC_ENVELOPE.angular_speed_max == 0.628

    def test_angular_acceleration_range(self) -> None:
        """angular_acceleration histogram spans [-3.14, 3.14] rad/s^2."""
        assert WOSAC_2025_KINEMATIC_ENVELOPE.angular_acceleration_min == -3.14
        assert WOSAC_2025_KINEMATIC_ENVELOPE.angular_acceleration_max == 3.14

    def test_envelope_type(self) -> None:
        """The module-level envelope is a KinematicEnvelope instance."""
        assert isinstance(WOSAC_2025_KINEMATIC_ENVELOPE, KinematicEnvelope)


class TestWosac2025MetametricConfig:
    """Per-feature 2025 estimator specs match the challenge textproto."""

    def test_kinematic_histograms(self) -> None:
        """Kinematic feature histograms: ranges, bins, pseudocount, weights."""
        cfg = WOSAC_2025_METAMETRIC_CONFIG
        for name, lo, hi, bins in (
            ("linear_speed", 0.0, 25.0, 10),
            ("linear_acceleration", -12.0, 12.0, 11),
            ("angular_speed", -0.628, 0.628, 11),
            ("angular_acceleration", -3.14, 3.14, 11),
        ):
            feature = getattr(cfg, name)
            assert feature.histogram is not None
            assert feature.histogram.min_val == lo
            assert feature.histogram.max_val == hi
            assert feature.histogram.num_bins == bins
            assert feature.histogram.additive_smoothing_pseudocount == 0.1
            assert feature.independent_timesteps is True
            assert feature.metametric_weight == 0.05

    def test_interactive_and_map_features(self) -> None:
        """Interaction and map feature specs match the textproto."""
        cfg = WOSAC_2025_METAMETRIC_CONFIG
        dno = cfg.distance_to_nearest_object
        assert dno.histogram is not None
        assert (dno.histogram.min_val, dno.histogram.max_val) == (-5.0, 40.0)
        assert dno.histogram.num_bins == 10
        assert dno.metametric_weight == 0.1

        dre = cfg.distance_to_road_edge
        assert dre.histogram is not None
        assert (dre.histogram.min_val, dre.histogram.max_val) == (-20.0, 40.0)
        assert dre.histogram.num_bins == 10
        assert dre.metametric_weight == 0.05

        ttc = cfg.time_to_collision
        assert ttc.histogram is not None
        assert (ttc.histogram.min_val, ttc.histogram.max_val) == (0.0, 5.0)
        assert ttc.histogram.num_bins == 10
        assert ttc.metametric_weight == 0.1

    def test_bernoulli_indications(self) -> None:
        """Indication features are bernoulli with the challenge weights."""
        cfg = WOSAC_2025_METAMETRIC_CONFIG
        for name, weight in (
            ("collision_indication", 0.25),
            ("offroad_indication", 0.25),
            ("traffic_light_violation", 0.05),
        ):
            feature = getattr(cfg, name)
            assert feature.histogram is None
            assert feature.metametric_weight == weight

    def test_weights_sum_to_one(self) -> None:
        """All ten 2025 feature weights sum to 1.0."""
        cfg = WOSAC_2025_METAMETRIC_CONFIG
        total = sum(
            getattr(cfg, name).metametric_weight
            for name in (
                "linear_speed",
                "linear_acceleration",
                "angular_speed",
                "angular_acceleration",
                "distance_to_nearest_object",
                "collision_indication",
                "time_to_collision",
                "distance_to_road_edge",
                "offroad_indication",
                "traffic_light_violation",
            )
        )
        assert total == pytest.approx(1.0)


class TestScenarioDictKeys:
    """Raw WOD scenario-dict key constants match the Waymax feature layout."""

    def test_temporal_state_keys(self) -> None:
        """Aggregated temporal keys live under state/all/."""
        assert STATE_Z == "state/all/z"
        assert STATE_TIMESTAMP_MICROS == "state/all/timestamp_micros"

    def test_object_metadata_keys(self) -> None:
        """Per-object metadata keys live directly under state/."""
        assert STATE_ID == "state/id"
        assert STATE_TYPE == "state/type"
        assert STATE_TRACKS_TO_PREDICT == "state/tracks_to_predict"
        assert STATE_OBJECTS_OF_INTEREST == "state/objects_of_interest"

    def test_scenario_id_key(self) -> None:
        """The scenario identifier key matches the TFRecord feature name."""
        assert SCENARIO_ID == "scenario/id"


class TestTokenizationKeys:
    """Field keys for the diffusion model's per-agent conditioning."""

    def test_scene_context_key(self) -> None:
        """Per-agent context rows are keyed by 'scene_context'."""
        assert SCENE_CONTEXT == "scene_context"

    def test_scene_context_distinct_from_scene_embedding(self) -> None:
        """The per-agent context key differs from the fused-token key."""
        assert SCENE_CONTEXT != SCENE_EMBEDDING

    def test_scene_token_valid_key(self) -> None:
        """The fused-token validity vector is keyed by 'scene_token_valid'."""
        assert SCENE_TOKEN_VALID == "scene_token_valid"

    def test_token_valid_keys_cover_every_embedding_modality(self) -> None:
        """Each fused modality has a parallel per-token validity key."""
        assert set(MODALITY_TOKEN_VALID_KEYS) == set(MODALITY_EMBEDDING_KEYS)
        assert SCENE_TOKEN_VALID not in set(MODALITY_TOKEN_VALID_KEYS.values())


class TestSensorConstants:
    """Sensor simulation constants carry their documented values."""

    def test_lidar_hit_weight_threshold_factor(self) -> None:
        """A hit needs half the configured density threshold in weight."""
        assert LIDAR_HIT_WEIGHT_THRESHOLD_FACTOR == 0.5

    def test_fog_extinction_coefficient(self) -> None:
        """Full-intensity fog transmits exp(-3) (~5%)."""
        assert FOG_EXTINCTION_COEFFICIENT == 3.0

    def test_nerf_camera_fov_is_sixty_degrees(self) -> None:
        """The pinhole camera model uses a 60-degree horizontal FoV."""
        import math

        assert NERF_CAMERA_HORIZONTAL_FOV_RAD == pytest.approx(math.pi / 3.0)


class TestMinerNormalizationConstants:
    """Miner normalization coefficients match the template family."""

    def test_offsets_are_zero(self) -> None:
        from simulacrax.core.constants import MINER_STATE_OFFSETS

        assert MINER_STATE_OFFSETS == (0.0, 0.0, 0.0, 0.0)

    def test_scales_positive_and_state_dim_length(self) -> None:
        from simulacrax.core.constants import MINER_STATE_SCALES

        assert len(MINER_STATE_SCALES) == 4
        assert all(scale > 0 for scale in MINER_STATE_SCALES)


class TestAgentLocalNormalizationConstants:
    """Per-agent local-frame diffusion normalization coefficients."""

    def test_offsets_are_zero(self) -> None:
        """The per-agent frame is already origin-centred, so offsets are zero."""
        from simulacrax.core.constants import AGENT_LOCAL_STATE_OFFSETS

        assert AGENT_LOCAL_STATE_OFFSETS == (0.0, 0.0, 0.0, 0.0)

    def test_scales_positive_and_state_dim_length(self) -> None:
        from simulacrax.core.constants import AGENT_LOCAL_STATE_SCALES

        assert len(AGENT_LOCAL_STATE_SCALES) == 4
        assert all(scale > 0 for scale in AGENT_LOCAL_STATE_SCALES)

    def test_scales_are_unit_variance_std_derived(self) -> None:
        """The scales are the local-frame target distribution's per-dim standard
        deviation, so the diffusion target has ~unit variance. Using the p99 tail
        maxima instead left the target at ~0.15 std, starving the noise schedule
        (only the top few percent of timesteps carried learnable signal)."""
        from simulacrax.core.constants import AGENT_LOCAL_STATE_SCALES

        forward, lateral, heading, speed = AGENT_LOCAL_STATE_SCALES
        # Agents move mostly forward, so the forward std exceeds the lateral std.
        assert forward > lateral
        # Forward is a tens-of-metres 8 s displacement std, not the ~150 m tail max.
        assert 10.0 < forward < 60.0
        # Heading is a small angular std (~0.3 rad), far below the wrapped range pi.
        assert 0.1 < heading < 1.0
        # Speed std is a few m/s.
        assert 3.0 < speed < 8.0


class TestSceneBackboneArchitecture:
    """Factorized scene-diffusion backbone architecture constants."""

    def test_architecture_version(self) -> None:
        """x0-prediction is revision 3 (epsilon-prediction weights invalid)."""
        assert SCENE_BACKBONE_ARCHITECTURE_VERSION == 3

    def test_ctgpp_layer_counts(self) -> None:
        """Default block/layer counts mirror the CTG++ T2.S1 factorization."""
        assert DEFAULT_NUM_BLOCKS == 2
        assert DEFAULT_NUM_TEMPORAL_LAYERS == 2
        assert DEFAULT_NUM_SOCIAL_LAYERS == 1
