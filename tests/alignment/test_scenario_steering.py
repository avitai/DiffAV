"""Tests for ranked-DPO scenario steering.

Covers ScenarioSteeringConfig validation, compute_scenario_reward behaviour,
and the ScenarioSteeringTrainer built on model-generated ranked pairs.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from simulacrax.alignment.dpo_trainer import (
    create_reference_model,
    DPOAlignmentConfig,
    DPOAlignmentMetrics,
    DPOAlignmentTrainer,
)
from simulacrax.alignment.scenario_steering import (
    compute_scenario_reward,
    ScenarioSteeringConfig,
    ScenarioSteeringTrainer,
    SteeringStrategy,
)
from simulacrax.models.trajectory_diffusion import (
    TrajectoryDiffusionModel,
)
from tests import support
from tests.alignment.helpers import CONTEXT_DIM, FUTURE_STEPS, NUM_AGENTS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_optimizer(model: TrajectoryDiffusionModel) -> nnx.Optimizer:
    """Create a small optimizer for testing."""
    return support.make_adam_optimizer(model)


def _make_dpo_trainer(model: TrajectoryDiffusionModel) -> DPOAlignmentTrainer:
    """Create a reference-free DPO trainer for testing."""
    optimizer = _make_optimizer(model)
    ref = create_reference_model(model)
    return DPOAlignmentTrainer(
        model=model,
        optimizer=optimizer,
        config=DPOAlignmentConfig(reference_free=True),
        reference_model=ref,
    )


def _make_model(seed: int = 0) -> TrajectoryDiffusionModel:
    cfg = support.make_diffusion_config(
        future_steps=FUTURE_STEPS,
        num_agents_max=NUM_AGENTS + 1,
        context_dim=CONTEXT_DIM,
    )
    return support.make_diffusion_model(cfg, seed=seed)


def _probe_contexts(num_scenes: int = 2) -> jax.Array:
    return jnp.ones((num_scenes, NUM_AGENTS, CONTEXT_DIM))


# ---------------------------------------------------------------------------
# TestScenarioSteeringConfig
# ---------------------------------------------------------------------------


class TestScenarioSteeringConfig:
    """Tests for ScenarioSteeringConfig validation and defaults."""

    def test_default_config_values(self) -> None:
        """Default config has expected values for all optional fields."""
        cfg = ScenarioSteeringConfig(target_scenario="forward")
        assert cfg.target_scenario == "forward"
        assert cfg.strategy is SteeringStrategy.RANKED_DPO
        assert cfg.steering_strength == pytest.approx(0.3)
        assert cfg.num_steering_steps == 1
        assert cfg.num_candidates == 8
        assert cfg.feasibility_threshold == pytest.approx(0.0)

    def test_config_frozen(self) -> None:
        """ScenarioSteeringConfig raises on mutation."""
        cfg = ScenarioSteeringConfig(target_scenario="forward")
        with pytest.raises(AttributeError):
            cfg.steering_strength = 0.5  # type: ignore[misc]

    def test_invalid_steering_strength_raises(self) -> None:
        """steering_strength outside [0.0, 1.0] raises ValueError."""
        with pytest.raises(ValueError, match="steering_strength"):
            ScenarioSteeringConfig(target_scenario="forward", steering_strength=1.5)
        with pytest.raises(ValueError, match="steering_strength"):
            ScenarioSteeringConfig(target_scenario="forward", steering_strength=-0.1)

    def test_invalid_num_steps_raises(self) -> None:
        """num_steering_steps < 1 raises ValueError."""
        with pytest.raises(ValueError, match="num_steering_steps"):
            ScenarioSteeringConfig(target_scenario="forward", num_steering_steps=0)

    def test_invalid_num_candidates_raises(self) -> None:
        with pytest.raises(ValueError, match="num_candidates"):
            ScenarioSteeringConfig(target_scenario="forward", num_candidates=1)

    def test_invalid_feasibility_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="feasibility_threshold"):
            ScenarioSteeringConfig(target_scenario="forward", feasibility_threshold=-0.1)


# ---------------------------------------------------------------------------
# TestComputeScenarioReward
# ---------------------------------------------------------------------------


class TestComputeScenarioReward:
    """Tests for the compute_scenario_reward function."""

    def _make_trajectories(self, lateral_motion: bool = False) -> jax.Array:
        """Build simple (num_agents, T, 4) trajectories for tests."""
        dt = 0.1
        t = jnp.arange(FUTURE_STEPS) * dt
        x = t * 10.0  # 10 m/s forward
        if lateral_motion:
            y = t * 3.0  # lateral drift
        else:
            y = jnp.zeros(FUTURE_STEPS)
        heading = jnp.zeros(FUTURE_STEPS)
        velocity = jnp.full(FUTURE_STEPS, 10.0)
        single = jnp.stack([x, y, heading, velocity], axis=-1)
        return jnp.broadcast_to(single[None], (NUM_AGENTS, FUTURE_STEPS, 4))

    def test_reward_scalar_output(self) -> None:
        """compute_scenario_reward returns a scalar (shape ())."""
        reward = compute_scenario_reward(self._make_trajectories(), "forward", reference_speed=10.0)
        assert reward.shape == ()

    def test_reward_bounded(self) -> None:
        """Reward is in [-1.0, 1.0] for valid inputs."""
        trajectories = self._make_trajectories()
        for scenario in ["forward", "lane_change", "unknown_type"]:
            reward = compute_scenario_reward(trajectories, scenario, reference_speed=10.0)
            assert float(reward) >= -1.0
            assert float(reward) <= 1.0

    def test_forward_scenario_reward_positive(self) -> None:
        """Straight-line high-velocity trajectories get positive reward for 'forward'."""
        reward = compute_scenario_reward(
            self._make_trajectories(lateral_motion=False), "forward", reference_speed=10.0
        )
        assert float(reward) > 0.0

    def test_reward_jit_compatible(self) -> None:
        """compute_scenario_reward runs without error under jax.jit."""
        jitted = jax.jit(lambda t: compute_scenario_reward(t, "forward", reference_speed=10.0))
        reward = jitted(self._make_trajectories())
        assert reward.shape == ()
        assert jnp.isfinite(reward)


# ---------------------------------------------------------------------------
# TestScenarioSteeringTrainer
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestScenarioSteeringTrainer:
    """Tests for the ranked-DPO ScenarioSteeringTrainer."""

    def test_steer_step_returns_metrics(
        self, small_diffusion_model: TrajectoryDiffusionModel
    ) -> None:
        """steer_step returns a DPOAlignmentMetrics instance."""
        dpo_trainer = _make_dpo_trainer(small_diffusion_model)
        config = ScenarioSteeringConfig(
            target_scenario="forward", steering_strength=0.5, num_candidates=4
        )
        trainer = ScenarioSteeringTrainer(dpo_trainer=dpo_trainer, config=config)
        metrics = trainer.steer_step(_probe_contexts(), jax.random.key(0), num_agents=NUM_AGENTS)
        assert isinstance(metrics, DPOAlignmentMetrics)
        assert isinstance(metrics.dpo_loss, float)

    def test_batch_pairs_carry_preference_signal(
        self, small_diffusion_model: TrajectoryDiffusionModel
    ) -> None:
        """Chosen candidates outscore rejected ones (SRC-5's missing signal)."""
        dpo_trainer = _make_dpo_trainer(small_diffusion_model)
        config = ScenarioSteeringConfig(
            target_scenario="forward", steering_strength=1.0, num_candidates=6
        )
        trainer = ScenarioSteeringTrainer(dpo_trainer=dpo_trainer, config=config)
        _, margins = trainer.build_step_batch(
            _probe_contexts(3), jax.random.key(1), num_agents=NUM_AGENTS
        )
        assert float(jnp.min(margins)) > 0.0

    def test_steer_step_updates_model(
        self, small_diffusion_model: TrajectoryDiffusionModel
    ) -> None:
        """Model parameters change after one steer_step (non-zero gradients)."""
        dpo_trainer = _make_dpo_trainer(small_diffusion_model)
        config = ScenarioSteeringConfig(
            target_scenario="forward", steering_strength=0.5, num_candidates=4
        )
        trainer = ScenarioSteeringTrainer(dpo_trainer=dpo_trainer, config=config)

        params_before = [
            p.copy() for p in jax.tree_util.tree_leaves(nnx.state(dpo_trainer.model, nnx.Param))
        ]
        trainer.steer_step(_probe_contexts(), jax.random.key(0), num_agents=NUM_AGENTS)
        params_after = jax.tree_util.tree_leaves(nnx.state(dpo_trainer.model, nnx.Param))

        any_changed = any(not jnp.allclose(p, a) for p, a in zip(params_before, params_after))
        assert any_changed

    def test_zero_pressure_matches_plain_dpo(self) -> None:
        """steering_strength=0.0 equals plain DPO on the same random pairs."""
        model_steer = _make_model(seed=0)
        model_dpo = _make_model(seed=0)

        config = ScenarioSteeringConfig(
            target_scenario="forward", steering_strength=0.0, num_candidates=4
        )
        key = jax.random.key(42)
        sample_key, loss_key = jax.random.split(key)

        steer_trainer = ScenarioSteeringTrainer(
            dpo_trainer=_make_dpo_trainer(model_steer), config=config
        )
        metrics_steer = steer_trainer.steer_step(_probe_contexts(), key, num_agents=NUM_AGENTS)

        dpo_trainer = _make_dpo_trainer(model_dpo)
        reference_trainer = ScenarioSteeringTrainer(dpo_trainer=dpo_trainer, config=config)
        batch, _ = reference_trainer.build_step_batch(
            _probe_contexts(), sample_key, num_agents=NUM_AGENTS
        )
        metrics_dpo = dpo_trainer.train_step(batch, loss_key)

        assert metrics_steer.dpo_loss == pytest.approx(metrics_dpo.dpo_loss, rel=1e-4)

    def test_reward_target_shapes_the_update(self) -> None:
        """Different reward targets produce different parameter updates.

        This is SRC-1's kill-shot: pre-fix, the reward entered the loss as a
        constant, so the gradient was identical no matter what the reward
        said — steering toward "forward" and "lane_change" updated the model
        identically. Post-fix, the target ranks the candidate pairs, so the
        updates differ.
        """

        def updated_params(target: str) -> list[jax.Array]:
            model = _make_model(seed=7)
            config = ScenarioSteeringConfig(
                target_scenario=target, steering_strength=1.0, num_candidates=6
            )
            trainer = ScenarioSteeringTrainer(dpo_trainer=_make_dpo_trainer(model), config=config)
            trainer.steer_step(_probe_contexts(2), jax.random.key(5), num_agents=NUM_AGENTS)
            return jax.tree_util.tree_leaves(nnx.state(model, nnx.Param))

        forward_params = updated_params("forward")
        lane_change_params = updated_params("lane_change")
        any_different = any(
            not jnp.allclose(a, b) for a, b in zip(forward_params, lane_change_params)
        )
        assert any_different

    def test_steer_step_nan_safe(self, small_diffusion_model: TrajectoryDiffusionModel) -> None:
        """NaN contexts do not raise; the NaN-safe path returns metrics."""
        dpo_trainer = _make_dpo_trainer(small_diffusion_model)
        config = ScenarioSteeringConfig(
            target_scenario="forward", steering_strength=0.5, num_candidates=4
        )
        trainer = ScenarioSteeringTrainer(dpo_trainer=dpo_trainer, config=config)
        nan_contexts = jnp.full((2, NUM_AGENTS, CONTEXT_DIM), float("nan"))
        metrics = trainer.steer_step(nan_contexts, jax.random.key(0), num_agents=NUM_AGENTS)
        assert isinstance(metrics, DPOAlignmentMetrics)
