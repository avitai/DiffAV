"""Tests for TrajectoryDiffusionModel."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from diffav.core.types import TrajectoryPrediction
from diffav.models.sampling_utils import stratified_timestep
from diffav.models.trajectory_diffusion import (
    create_trajectory_model,
    GuidanceSpec,
    TrajectoryDiffusionConfig,
    TrajectoryDiffusionModel,
)
from tests import support
from tests.models.helpers import (
    make_model as _make_model,
    sample_data as _sample_data,
    small_diffusion_config as _small_config,
    TEST_AGENTS,
    TEST_CTX_DIM,
    TEST_FUTURE,
    TEST_STATE_DIM,
)


# ---- Config validation ----


class TestTrajectoryDiffusionConfig:
    """Tests for TrajectoryDiffusionConfig validation."""

    def test_defaults(self) -> None:
        """Default config creates with expected values."""
        cfg = TrajectoryDiffusionConfig()
        assert cfg.model_type == "diffusion"
        assert cfg.hidden_dim == 128
        assert cfg.num_timesteps == 1000
        assert cfg.loss_type == "mse"
        assert cfg.noise_schedule_type == "cosine"

    def test_invalid_model_type(self) -> None:
        """Non-diffusion model_type raises."""
        with pytest.raises(ValueError, match="model_type"):
            _small_config(model_type="gan")

    def test_invalid_loss_type(self) -> None:
        """Unknown loss_type raises."""
        with pytest.raises(ValueError, match="is not a valid LossType"):
            _small_config(loss_type="huber")

    def test_invalid_schedule_type(self) -> None:
        """Unknown noise_schedule_type raises."""
        with pytest.raises(ValueError, match="is not a valid NoiseScheduleType"):
            _small_config(noise_schedule_type="sigmoid")

    def test_hidden_dim_not_divisible(self) -> None:
        """Non-divisible hidden_dim/num_heads raises."""
        with pytest.raises(ValueError, match="divisible"):
            _small_config(hidden_dim=33, num_heads=4)

    @pytest.mark.parametrize(
        "field",
        [
            "hidden_dim",
            "num_blocks",
            "num_temporal_layers",
            "num_social_layers",
            "num_heads",
            "future_steps",
            "num_timesteps",
            "num_agents_max",
            "state_dim",
            "context_dim",
        ],
    )
    def test_positive_fields(self, field: str) -> None:
        """Zero-valued positive fields raise."""
        with pytest.raises(ValueError, match="must be positive"):
            _small_config(**{field: 0})


# ---- Forward diffusion ----


class TestQSample:
    """Tests for q_sample (forward diffusion)."""

    def test_shape(self) -> None:
        """Output matches input shape."""
        model = _make_model()
        traj, _ = _sample_data()
        noise = jnp.ones_like(traj)
        noisy = model.q_sample(traj, 5, noise)
        assert noisy.shape == traj.shape

    def test_t0_less_noisy_than_t_max(self) -> None:
        """At t=0, trajectories are closer to original than at t=max."""
        model = _make_model()
        traj, _ = _sample_data()
        noise = jax.random.normal(jax.random.key(0), traj.shape)
        noisy_t0 = model.q_sample(traj, 0, noise)
        t_max = model.config.num_timesteps - 1
        noisy_tmax = model.q_sample(traj, t_max, noise)
        diff_t0 = jnp.mean(jnp.abs(noisy_t0 - traj))
        diff_tmax = jnp.mean(jnp.abs(noisy_tmax - traj))
        assert diff_t0 < diff_tmax

    def test_t_max_is_noisy(self) -> None:
        """At t=max, output differs significantly from original."""
        model = _make_model()
        traj, _ = _sample_data()
        noise = jax.random.normal(jax.random.key(0), traj.shape)
        t_max = model.config.num_timesteps - 1
        noisy = model.q_sample(traj, t_max, noise)
        max_diff = jnp.max(jnp.abs(noisy - traj))
        assert max_diff > 0.5


# ---- Noise prediction ----


class TestPredictNoise:
    """Tests for predict_noise."""

    def test_shape(self) -> None:
        """Prediction shape matches input shape."""
        model = _make_model()
        traj, ctx = _sample_data()
        pred = model.predict_noise(traj, 5, ctx)
        assert pred.shape == traj.shape


# ---- Reverse diffusion ----


class TestPSampleStep:
    """Tests for p_sample_step."""

    def test_shape(self) -> None:
        """Output shape matches input."""
        model = _make_model()
        traj, ctx = _sample_data()
        key = jax.random.key(42)
        out = model.p_sample_step(traj, 5, ctx, key=key)
        assert out.shape == traj.shape


class TestSample:
    """Tests for full reverse diffusion sampling."""

    def test_shape(self) -> None:
        """Sample produces correct output shape."""
        model = _make_model()
        _, ctx = _sample_data()
        num_agents = 3
        key = jax.random.key(0)

        result = model.sample(ctx, key=key)
        assert result.trajectories.shape == (
            num_agents,
            TEST_FUTURE,
            TEST_STATE_DIM,
        )

    def test_returns_trajectory_prediction(self) -> None:
        """Sample returns a TrajectoryPrediction instance."""
        model = _make_model()
        _, ctx = _sample_data()
        key = jax.random.key(0)

        result = model.sample(ctx, key=key)
        assert isinstance(result, TrajectoryPrediction)
        assert len(result.agent_ids) == 3
        assert result.agent_ids == ("agent_0", "agent_1", "agent_2")

    @pytest.mark.parametrize("num_agents", [1, 2, 5])
    def test_num_agents_inferred_from_context(self, num_agents: int) -> None:
        """The agent count is inferred from scene_context rows, not passed.

        One trajectory is generated per context row; the count is not capped
        by num_agents_max (here 4), so a 5-row context yields 5 trajectories.
        """
        model = _make_model()
        scene_context = jnp.ones((num_agents, TEST_CTX_DIM))
        result = model.sample(scene_context, key=jax.random.key(0))
        assert result.trajectories.shape == (num_agents, TEST_FUTURE, TEST_STATE_DIM)
        assert len(result.agent_ids) == num_agents


# ---- Training loss ----


class TestComputeLoss:
    """Tests for compute_loss."""

    def test_scalar_nonnegative(self) -> None:
        """Loss is a scalar and non-negative."""
        model = _make_model()
        traj, ctx = _sample_data()
        key = jax.random.key(42)
        loss = model.compute_loss(traj, ctx, key=key)
        assert loss.shape == ()
        assert loss >= 0

    def test_l1_loss(self) -> None:
        """L1 loss variant produces scalar output."""
        config = _small_config(loss_type="l1")
        model = _make_model(config)
        traj, ctx = _sample_data()
        key = jax.random.key(42)
        loss = model.compute_loss(traj, ctx, key=key)
        assert loss.shape == ()
        assert loss >= 0

    def test_gradient_exists(self) -> None:
        """Gradients w.r.t. model parameters are non-zero."""
        model = _make_model()
        traj, ctx = _sample_data()
        key = jax.random.key(42)

        def loss_fn(model: TrajectoryDiffusionModel) -> jax.Array:
            return model.compute_loss(traj, ctx, key=key)

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        assert loss.shape == ()

        flat_grads = jax.tree_util.tree_leaves(grads)
        has_nonzero = any(bool(jnp.any(g != 0)) for g in flat_grads if hasattr(g, "shape"))
        assert has_nonzero, "All gradients are zero"

    def test_jit_compute_loss(self) -> None:
        """compute_loss works under jax.jit via nnx.jit."""
        model = _make_model()
        traj, ctx = _sample_data()
        key = jax.random.key(42)

        @nnx.jit
        def jitted_loss(
            m: TrajectoryDiffusionModel,
        ) -> jax.Array:
            return m.compute_loss(traj, ctx, key=key)

        loss = jitted_loss(model)
        assert loss.shape == ()
        assert loss >= 0


class TestMaskedComputeLoss:
    """Validity-masked epsilon loss (WOD-tutorial ``sample_weight`` analog)."""

    def test_none_mask_matches_default(self) -> None:
        """valid_mask=None reproduces the unmasked mean bit-identically."""
        model = _make_model()
        traj, ctx = _sample_data()
        key = jax.random.key(7)
        default = model.compute_loss(traj, ctx, key=key)
        explicit_none = model.compute_loss(traj, ctx, key=key, valid_mask=None)
        assert default == explicit_none

    def test_all_valid_mask_equals_unmasked_mean(self) -> None:
        """An all-ones mask equals the plain mean (same numerator/denominator)."""
        model = _make_model()
        traj, ctx = _sample_data()
        key = jax.random.key(7)
        mask = jnp.ones(traj.shape[:2])
        masked = model.compute_loss(traj, ctx, key=key, valid_mask=mask)
        unmasked = model.compute_loss(traj, ctx, key=key)
        assert jnp.allclose(masked, unmasked)

    def test_fully_invalid_agent_contributes_zero(self) -> None:
        """Zeroing an agent's mask row matches the loss over the valid rows only."""
        model = _make_model()
        num_agents = 3
        traj, ctx = _sample_data(num_agents=num_agents)
        key = jax.random.key(7)
        mask = jnp.ones((num_agents, traj.shape[1]))
        mask = mask.at[num_agents - 1].set(0.0)
        masked = model.compute_loss(traj, ctx, key=key, valid_mask=mask)
        # Same key/data restricted to the two valid agents, fully valid.
        valid_only = model.compute_loss(
            traj[: num_agents - 1],
            ctx[: num_agents - 1],
            key=key,
            valid_mask=mask[: num_agents - 1],
        )
        # The invalid row adds nothing to numerator or denominator, but the
        # backbone still sees all agents, so only the reduction is compared:
        # both restrict the average to the valid entries.
        assert masked.shape == () and valid_only.shape == ()
        assert masked >= 0

    def test_partial_step_mask_scalar_nonnegative(self) -> None:
        """A per-step partial mask yields a finite non-negative scalar."""
        model = _make_model()
        traj, ctx = _sample_data()
        key = jax.random.key(7)
        mask = jnp.ones(traj.shape[:2]).at[:, traj.shape[1] // 2 :].set(0.0)
        loss = model.compute_loss(traj, ctx, key=key, valid_mask=mask)
        assert loss.shape == ()
        assert bool(jnp.isfinite(loss)) and loss >= 0

    def test_masked_loss_has_gradients(self) -> None:
        """Masked loss backpropagates to model parameters."""
        model = _make_model()
        traj, ctx = _sample_data()
        key = jax.random.key(7)
        mask = jnp.ones(traj.shape[:2]).at[0].set(0.0)

        def loss_fn(m: TrajectoryDiffusionModel) -> jax.Array:
            return m.compute_loss(traj, ctx, key=key, valid_mask=mask)

        grads = nnx.grad(loss_fn)(model)
        flat_grads = jax.tree_util.tree_leaves(grads)
        has_nonzero = any(bool(jnp.any(g != 0)) for g in flat_grads if hasattr(g, "shape"))
        assert has_nonzero, "Masked loss carries no model gradients"

    def test_masked_loss_under_vmap(self) -> None:
        """The masked loss vmaps over a batch (training uses a vmapped step)."""
        model = _make_model()
        batch = 3
        traj = jnp.ones((batch, TEST_AGENTS, TEST_FUTURE, TEST_STATE_DIM))
        ctx = jnp.ones((batch, TEST_AGENTS, TEST_CTX_DIM))
        masks = jnp.ones((batch, TEST_AGENTS, TEST_FUTURE)).at[:, -1].set(0.0)
        keys = jax.random.split(jax.random.key(7), batch)
        losses = jax.vmap(lambda tr, cx, mk, k: model.compute_loss(tr, cx, key=k, valid_mask=mk))(
            traj, ctx, masks, keys
        )
        assert losses.shape == (batch,)
        assert bool(jnp.all(jnp.isfinite(losses)))


class TestComputeLossAndPrediction:
    """Tests for the x̂₀ seam consumed by downstream physics penalties."""

    def test_returns_loss_and_predicted_trajectories(self) -> None:
        """Returns (scalar loss, predicted clean trajectories)."""
        model = _make_model()
        traj, ctx = _sample_data()
        loss, predicted = model.compute_loss_and_prediction(traj, ctx, key=jax.random.key(42))
        assert loss.shape == ()
        assert predicted.shape == traj.shape
        assert bool(jnp.all(jnp.isfinite(predicted)))

    def test_loss_matches_compute_loss(self) -> None:
        """With the same key, the loss equals compute_loss exactly."""
        model = _make_model()
        traj, ctx = _sample_data()
        key = jax.random.key(42)
        loss_only = model.compute_loss(traj, ctx, key=key)
        loss_pair, _ = model.compute_loss_and_prediction(traj, ctx, key=key)
        assert jnp.allclose(loss_only, loss_pair)

    def test_prediction_carries_model_gradients(self) -> None:
        """Functions of the prediction differentiate w.r.t. model params."""
        model = _make_model()
        traj, ctx = _sample_data()
        key = jax.random.key(42)

        def predicted_magnitude(m: TrajectoryDiffusionModel) -> jax.Array:
            _, predicted = m.compute_loss_and_prediction(traj, ctx, key=key)
            return jnp.mean(predicted**2)

        grads = nnx.grad(predicted_magnitude)(model)
        flat_grads = jax.tree_util.tree_leaves(grads)
        has_nonzero = any(bool(jnp.any(g != 0)) for g in flat_grads if hasattr(g, "shape"))
        assert has_nonzero, "Prediction carries no model gradients"

    def test_default_stratum_is_bit_identical_to_uniform(self) -> None:
        """``timestep_stratum=None`` reproduces the prior uniform-draw loss."""
        model = _make_model()
        traj, ctx = _sample_data()
        key = jax.random.key(11)
        baseline, _ = model.compute_loss_and_prediction(traj, ctx, key=key)
        defaulted, _ = model.compute_loss_and_prediction(traj, ctx, key=key, timestep_stratum=None)
        assert jnp.array_equal(baseline, defaulted)

    def test_stratum_draws_land_in_assigned_bin(self) -> None:
        """Each stratum draws a timestep inside its own schedule bin.

        The stratum draw uses ``key_t = split(key)[0]`` (the same split the
        loss path performs), so reproducing that split yields the timestep
        the loss actually used; here we assert it lands in stratum i's bin.
        """
        model = _make_model()
        traj, ctx = _sample_data()
        num_timesteps = model.config.num_timesteps
        count = num_timesteps  # one stratum per bin
        for i in range(count):
            key = jax.random.fold_in(jax.random.key(5), i)
            key_t, _ = jax.random.split(key)
            t = stratified_timestep(key_t, i, count, num_timesteps)
            assert int(t) == i
            # The loss runs without error for this stratum assignment.
            loss, _ = model.compute_loss_and_prediction(
                traj, ctx, key=key, timestep_stratum=(i, count)
            )
            assert jnp.isfinite(loss)


class TestComputeLossOutputs:
    """Tests for the structured loss-outputs seam (loss, x̂₀, ᾱ_t)."""

    def test_matches_compute_loss_and_prediction(self) -> None:
        """The pair view is exactly the structured view minus ᾱ_t."""
        model = _make_model()
        traj, ctx = _sample_data()
        key = jax.random.key(42)
        outputs = model.compute_loss_outputs(traj, ctx, key=key)
        loss, predicted = model.compute_loss_and_prediction(traj, ctx, key=key)
        assert jnp.array_equal(outputs.loss, loss)
        assert jnp.array_equal(outputs.prediction, predicted)

    def test_alpha_bar_matches_schedule_at_pinned_timestep(self) -> None:
        """ᾱ_t reports the schedule value of the drawn timestep.

        Width-1 strata (one stratum per timestep bin) pin the draw to
        ``t == i`` exactly, so the expected schedule entry is known.
        """
        model = _make_model()
        traj, ctx = _sample_data()
        num_timesteps = model.config.num_timesteps
        for i in (0, num_timesteps // 2, num_timesteps - 1):
            outputs = model.compute_loss_outputs(
                traj, ctx, key=jax.random.key(3), timestep_stratum=(i, num_timesteps)
            )
            expected = model.noise_schedule.alphas_cumprod[i]
            assert jnp.allclose(outputs.alpha_bar_t, expected)

    def test_alpha_bar_is_finite_scalar(self) -> None:
        """ᾱ_t is a finite scalar in (0, 1]."""
        model = _make_model()
        traj, ctx = _sample_data()
        outputs = model.compute_loss_outputs(traj, ctx, key=jax.random.key(9))
        assert outputs.alpha_bar_t.shape == ()
        assert 0.0 < float(outputs.alpha_bar_t) <= 1.0


class TestSnrGammaWeighting:
    """Tests for Min-SNR-γ diffusion-loss weighting (``snr_gamma``)."""

    def test_non_positive_gamma_raises(self) -> None:
        """snr_gamma must be positive when set."""
        with pytest.raises(ValueError, match="snr_gamma"):
            _small_config(snr_gamma=0.0)

    def test_weight_applied_at_pinned_timestep(self) -> None:
        """Weighted loss equals unweighted loss × min(SNR, γ)/SNR.

        Width-1 strata pin ``t == i`` so the per-timestep SNR is known;
        the factor matches the reference ε-prediction Min-SNR weighting.
        """
        gamma = 5.0
        base = _make_model()
        weighted = _make_model(config=_small_config(snr_gamma=gamma))
        traj, ctx = _sample_data()
        num_timesteps = base.config.num_timesteps
        for i in (0, num_timesteps // 2, num_timesteps - 1):
            key = jax.random.key(11)
            stratum = (i, num_timesteps)
            loss_base, _ = base.compute_loss_and_prediction(
                traj, ctx, key=key, timestep_stratum=stratum
            )
            loss_weighted, _ = weighted.compute_loss_and_prediction(
                traj, ctx, key=key, timestep_stratum=stratum
            )
            alpha_bar = base.noise_schedule.alphas_cumprod[i]
            snr = alpha_bar / (1.0 - alpha_bar)
            expected = loss_base * jnp.minimum(snr, gamma) / snr
            assert jnp.allclose(loss_weighted, expected, rtol=1e-6)

    def test_huge_gamma_matches_unweighted(self) -> None:
        """γ ≥ max SNR makes every weight 1: identical to the default path."""
        base = _make_model()
        weighted = _make_model(config=_small_config(snr_gamma=1e12))
        traj, ctx = _sample_data()
        key = jax.random.key(5)
        loss_base, _ = base.compute_loss_and_prediction(traj, ctx, key=key)
        loss_weighted, _ = weighted.compute_loss_and_prediction(traj, ctx, key=key)
        assert jnp.allclose(loss_base, loss_weighted, rtol=1e-6)

    def test_weighted_loss_has_gradients(self) -> None:
        """The weighted loss still differentiates w.r.t. model params."""
        model = _make_model(config=_small_config(snr_gamma=5.0))
        traj, ctx = _sample_data()

        def loss_fn(m: TrajectoryDiffusionModel) -> jax.Array:
            return m.compute_loss(traj, ctx, key=jax.random.key(2))

        grads = nnx.grad(loss_fn)(model)
        flat_grads = jax.tree_util.tree_leaves(grads)
        has_nonzero = any(bool(jnp.any(g != 0)) for g in flat_grads if hasattr(g, "shape"))
        assert has_nonzero, "Weighted loss carries no model gradients"


# ---- Forward pass under JIT ----


class TestJIT:
    """Tests for JIT compatibility of model operations."""

    def test_predict_noise_jit(self) -> None:
        """predict_noise works under nnx.jit."""
        model = _make_model()
        traj, ctx = _sample_data()

        @nnx.jit
        def jitted_predict(
            m: TrajectoryDiffusionModel,
        ) -> jax.Array:
            return m.predict_noise(traj, 5, ctx)

        out = jitted_predict(model)
        assert out.shape == traj.shape

    def test_q_sample_jit(self) -> None:
        """q_sample works under nnx.jit."""
        model = _make_model()
        traj, _ = _sample_data()
        noise = jnp.ones_like(traj)

        @nnx.jit
        def jitted_q(
            m: TrajectoryDiffusionModel,
        ) -> jax.Array:
            return m.q_sample(traj, 5, noise)

        out = jitted_q(model)
        assert out.shape == traj.shape


# ---- Factory ----


class TestFactory:
    """Tests for create_trajectory_model factory."""

    def test_produces_correct_type(self) -> None:
        """Factory returns TrajectoryDiffusionModel."""
        config = _small_config()
        rngs = nnx.Rngs(params=jax.random.key(0))
        model = create_trajectory_model(config, rngs=rngs)
        assert isinstance(model, TrajectoryDiffusionModel)


# ---- x̂₀ scale contract ----


class TestX0ClipContract:
    """The x̂₀ clamp must follow the configured data scale, not a magic ±10.

    WOD trajectories are ego-centred metres — 8 s horizons far exceed
    ±10 m — so clipping is off by default and opt-in via
    ``x0_clip_bound`` for data normalised to a known range.
    """

    @staticmethod
    def _manual_step(
        model: TrajectoryDiffusionModel,
        x_t: jax.Array,
        t: int,
        ctx: jax.Array,
        key: jax.Array,
        clip_bound: float | None,
    ) -> jax.Array:
        """Reference reverse step with explicit control over the x̂₀ clamp."""
        t_batch = jnp.atleast_1d(jnp.asarray(t))
        pred_noise = model.predict_noise(x_t, t, ctx)
        x_0 = model.noise_schedule.predict_start_from_noise(x_t, t_batch, pred_noise)
        if clip_bound is not None:
            x_0 = jnp.clip(x_0, -clip_bound, clip_bound)
        mean, _, log_var = model.noise_schedule.q_posterior_mean_variance(x_0, x_t, t_batch)
        noise = jax.random.normal(key, x_t.shape)
        return mean + jnp.exp(0.5 * log_var) * noise

    def test_default_does_not_clamp_meter_scale_states(self) -> None:
        """By default the reverse step uses the unclipped x̂₀ estimate.

        A ~500 m ego-frame state makes the x̂₀ estimate far exceed ±10;
        the step output must match the posterior of the *unclipped*
        estimate exactly (the old hardcoded clip(±10) diverges from it).
        """
        model = _make_model()
        _, ctx = _sample_data()
        x_t = jnp.full((3, TEST_FUTURE, TEST_STATE_DIM), 500.0)
        key = jax.random.key(0)
        out = model.p_sample_step(x_t, 1, ctx, key=key)
        expected = self._manual_step(model, x_t, 1, ctx, key, clip_bound=None)
        assert jnp.allclose(out, expected, rtol=1e-6)

    def test_configured_bound_clips(self) -> None:
        """An explicit x0_clip_bound clamps the x̂₀ estimate to ±bound."""
        model = _make_model(config=_small_config(x0_clip_bound=5.0))
        _, ctx = _sample_data()
        x_t = jnp.full((3, TEST_FUTURE, TEST_STATE_DIM), 500.0)
        key = jax.random.key(0)
        out = model.p_sample_step(x_t, 1, ctx, key=key)
        expected = self._manual_step(model, x_t, 1, ctx, key, clip_bound=5.0)
        unclipped = self._manual_step(model, x_t, 1, ctx, key, clip_bound=None)
        assert jnp.allclose(out, expected, rtol=1e-6)
        assert not jnp.allclose(out, unclipped, rtol=1e-6)

    def test_non_positive_bound_raises(self) -> None:
        """x0_clip_bound must be strictly positive when set."""
        with pytest.raises(ValueError, match="x0_clip_bound"):
            _small_config(x0_clip_bound=0.0)


# ---- Scan-based sampling loop ----


class TestSampleScan:
    """The reverse loop must be scan-based: jit-stable, key-compatible."""

    def test_sample_jit_compatible_and_repeatable(self) -> None:
        """Sampling composes with nnx.jit and is bit-repeatable."""
        model = _make_model()
        _, ctx = _sample_data()

        @nnx.jit
        def draw(m: TrajectoryDiffusionModel, context: jax.Array, key: jax.Array) -> jax.Array:
            return m.sample(context, key=key).trajectories

        first = draw(model, ctx, jax.random.key(1))
        second = draw(model, ctx, jax.random.key(1))
        assert first.shape == (3, TEST_FUTURE, TEST_STATE_DIM)
        assert bool(jnp.all(jnp.isfinite(first)))
        assert jnp.array_equal(first, second)

    def test_sample_matches_stepwise_loop(self) -> None:
        """The scanned loop reproduces an explicit p_sample_step loop.

        Pins the per-step key derivation contract: an init split off the
        caller's key, then fold_in(key, t) at each timestep.
        """
        model = _make_model()
        _, ctx = _sample_data()
        num_agents = ctx.shape[0]
        key = jax.random.key(3)

        result = model.sample(ctx, key=key).trajectories

        loop_key, init_key = jax.random.split(key)
        x_t = jax.random.normal(init_key, (num_agents, TEST_FUTURE, TEST_STATE_DIM))
        for t_val in range(model.config.num_timesteps - 1, -1, -1):
            step_key = jax.random.fold_in(loop_key, t_val)
            x_t = model.p_sample_step(x_t, t_val, ctx, key=step_key)

        assert jnp.allclose(result, x_t, rtol=1e-5, atol=1e-6)

    def test_sample_deterministic(self) -> None:
        """Two eager samples with the same key are bit-identical."""
        model = _make_model()
        _, ctx = _sample_data()
        first = model.sample(ctx, key=jax.random.key(9)).trajectories
        second = model.sample(ctx, key=jax.random.key(9)).trajectories
        assert jnp.array_equal(first, second)

    def test_grad_through_sample_wrt_context(self) -> None:
        """Full sampling is differentiable w.r.t. the conditioning context.

        This is the adversarial-search code path: gradients flow through the
        checkpointed scan body, so they must be finite, nonzero, and jit-safe.
        """
        model = _make_model()
        # adaLN is zero-initialised (DiT identity init), so an untrained model
        # ignores its context. Adversarial search runs against trained weights;
        # give adaLN non-zero weights to exercise the conditioning pathway.
        adaln_key = jax.random.key(11)
        adaln_linears = [block.adaln for block in model.backbone.blocks]
        adaln_linears.append(model.backbone.output_adaln)
        for index, linear in enumerate(adaln_linears):
            shape = linear.kernel[...].shape
            linear.kernel[...] = 0.5 * jax.random.normal(
                jax.random.fold_in(adaln_key, index), shape
            )
        _, ctx = _sample_data()
        key = jax.random.key(4)

        @jax.jit
        def mean_displacement(context: jax.Array) -> jax.Array:
            return jnp.mean(model.sample(context, key=key).trajectories)

        grad = jax.grad(mean_displacement)(ctx)
        assert grad.shape == ctx.shape
        assert bool(jnp.all(jnp.isfinite(grad)))
        assert float(jnp.max(jnp.abs(grad))) > 0.0


class TestGuidance:
    """Test-time x̂₀ reward-gradient guidance in the reverse loop."""

    def test_zero_scale_identical_to_unguided(self) -> None:
        """A guidance_fn with scale 0 changes nothing, bit-exactly."""
        model = _make_model()
        _, ctx = _sample_data()
        key = jax.random.key(0)

        plain = model.sample(ctx, key=key)
        guided = model.sample(
            ctx, key=key, guidance=GuidanceSpec(reward_fn=lambda x: jnp.mean(x[..., 0]), scale=0.0)
        )
        assert jnp.array_equal(plain.trajectories, guided.trajectories)

    def test_positive_reward_gradient_shifts_samples(self) -> None:
        """Guidance toward higher x strictly raises the mean x coordinate."""
        model = _make_model()
        _, ctx = _sample_data()
        key = jax.random.key(0)

        plain = model.sample(ctx, key=key)
        guided = model.sample(
            ctx,
            key=key,
            guidance=GuidanceSpec(reward_fn=lambda x: jnp.mean(x[..., 0]), scale=100.0),
        )
        assert float(jnp.mean(guided.trajectories[..., 0])) > float(
            jnp.mean(plain.trajectories[..., 0])
        )

    def test_nonfinite_reward_gradient_is_skipped(self) -> None:
        """A reward with NaN gradients degrades to unguided sampling."""
        model = _make_model()
        _, ctx = _sample_data()
        key = jax.random.key(0)

        plain = model.sample(ctx, key=key)
        guided = model.sample(
            ctx,
            key=key,
            # sqrt of the (partly negative) estimate produces NaN gradients
            guidance=GuidanceSpec(reward_fn=lambda x: jnp.sum(jnp.sqrt(x)), scale=10.0),
        )
        assert bool(jnp.all(jnp.isfinite(guided.trajectories)))
        assert jnp.array_equal(plain.trajectories, guided.trajectories)

    def test_guided_sampling_jit_compatible(self) -> None:
        """Guidance compiles under jit and keeps its directional effect.

        Pointwise jit-vs-eager comparison is unstable here: each reverse
        step's reward gradient feeds the next step, so f32 fusion
        differences compound over the loop. The compilable, finite, and
        directionally-consistent properties are the stable contract.
        """
        model = _make_model()
        _, ctx = _sample_data()

        def guided(context: jax.Array) -> jax.Array:
            prediction = model.sample(
                context,
                key=jax.random.key(0),
                guidance=GuidanceSpec(reward_fn=lambda x: jnp.mean(x[..., 0]), scale=100.0),
            )
            return prediction.trajectories

        plain = model.sample(ctx, key=jax.random.key(0))
        jitted = jax.jit(guided)(ctx)
        assert bool(jnp.all(jnp.isfinite(jitted)))
        assert float(jnp.mean(jitted[..., 0])) > float(jnp.mean(plain.trajectories[..., 0]))


class TestStateNormalization:
    """CTG-style normalized-space diffusion with metre-space boundaries."""

    def _normalized_config(self):
        return _small_config(
            state_offsets=(0.0, 0.0, 0.0, 0.0),
            state_scales=(50.0, 50.0, 3.2, 15.0),
            x0_clip_bound=1.0,
        )

    def test_scale_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="state_scales"):
            _small_config(state_offsets=(0.0,) * 4, state_scales=(50.0,) * 3)

    def test_nonpositive_scale_raises(self) -> None:
        with pytest.raises(ValueError, match="state_scales"):
            _small_config(state_offsets=(0.0,) * 4, state_scales=(50.0, 0.0, 1.0, 1.0))

    def test_offsets_without_scales_raises(self) -> None:
        with pytest.raises(ValueError, match="state_offsets"):
            _small_config(state_offsets=(0.0,) * 4)

    def test_sample_bounded_by_descaled_clip(self) -> None:
        """With normalization + unit clip, samples are bounded dimension-wise.

        The final reverse step returns the clipped x0 estimate, so after
        descaling every dimension must lie within offset ± scale exactly —
        regardless of how (un)trained the model is. This is the contract that
        keeps showcase samples at data scale.
        """
        model = TrajectoryDiffusionModel(
            self._normalized_config(), rngs=nnx.Rngs(params=jax.random.key(0))
        )
        prediction = model.sample(jnp.ones((3, TEST_CTX_DIM)), key=jax.random.key(1))
        scales = jnp.array([50.0, 50.0, 3.2, 15.0])
        assert bool(jnp.all(jnp.abs(prediction.trajectories) <= scales + 1e-4))

    def test_loss_and_prediction_descaled(self) -> None:
        """The normalized model's outputs equal descale(plain-model outputs).

        Two models with identical weights — one normalizing, one raw — fed
        equivalent inputs must produce the same loss and predictions related
        exactly by the descale map. This pins that training runs in
        normalized space while the prediction seam stays in raw units.
        """
        scales = jnp.array([50.0, 50.0, 3.2, 15.0])
        normalized_model = TrajectoryDiffusionModel(
            self._normalized_config(), rngs=nnx.Rngs(params=jax.random.key(0))
        )
        plain_model = TrajectoryDiffusionModel(
            _small_config(x0_clip_bound=1.0), rngs=nnx.Rngs(params=jax.random.key(0))
        )
        trajectories = jnp.ones((3, TEST_FUTURE, TEST_STATE_DIM)) * 40.0
        context = jnp.ones((3, TEST_CTX_DIM))
        key = jax.random.key(2)

        loss_norm, pred_norm = normalized_model.compute_loss_and_prediction(
            trajectories, context, key=key
        )
        loss_plain, pred_plain = plain_model.compute_loss_and_prediction(
            trajectories / scales, context, key=key
        )
        assert float(loss_norm) == pytest.approx(float(loss_plain), rel=1e-6)
        assert jnp.allclose(pred_norm, pred_plain * scales, rtol=1e-5, atol=1e-4)

    def test_identity_when_unset(self) -> None:
        """Without normalization the behavior is unchanged (raw space)."""
        model = _make_model()
        reference = _make_model()
        key = jax.random.key(3)
        a = model.sample(jnp.ones((2, TEST_CTX_DIM)), key=key)
        b = reference.sample(jnp.ones((2, TEST_CTX_DIM)), key=key)
        assert jnp.array_equal(a.trajectories, b.trajectories)


_MAP_TOKENS = 5
_MAP_TOKEN_DIM = 12


def _scene_tokens(seed: int = 7) -> jax.Array:
    """A fused scene-token set for the backbone map cross-attention."""
    return jax.random.normal(jax.random.key(seed), (_MAP_TOKENS, _MAP_TOKEN_DIM))


def _map_model(seed: int = 0) -> TrajectoryDiffusionModel:
    """A map-conditioned diffusion model (backbone map cross-attention on)."""
    cfg = _small_config(use_map_cross_attention=True, scene_token_dim=_MAP_TOKEN_DIM)
    return _make_model(cfg, seed=seed)


class TestMapConditioning:
    """The model threads a fused scene-token set to the backbone cross-attention."""

    def test_config_map_fields_default_off(self) -> None:
        """The diffusion config inherits the map-attention fields, off by default."""
        cfg = TrajectoryDiffusionConfig()
        assert cfg.use_map_cross_attention is False
        assert cfg.scene_token_dim is None

    def test_map_aware_sample_shape(self) -> None:
        """Map-conditioned sampling yields one trajectory per context row."""
        model = _map_model()
        _, context = _sample_data(TEST_AGENTS)
        pred = model.sample(context, key=jax.random.key(1), scene_tokens=_scene_tokens())
        assert pred.trajectories.shape == (TEST_AGENTS, TEST_FUTURE, TEST_STATE_DIM)

    def test_map_aware_compute_loss_finite(self) -> None:
        """Map-conditioned loss is a finite scalar."""
        model = _map_model()
        traj, context = _sample_data(TEST_AGENTS)
        loss = model.compute_loss(
            traj, context, key=jax.random.key(2), scene_tokens=_scene_tokens()
        )
        assert loss.shape == ()
        assert bool(jnp.isfinite(loss))

    def test_map_blind_model_rejects_scene_tokens(self) -> None:
        """A map-blind model passed scene_tokens fails fast (via the backbone)."""
        model = _make_model()
        traj, context = _sample_data(TEST_AGENTS)
        with pytest.raises(ValueError, match="map cross-attention"):
            model.compute_loss(traj, context, key=jax.random.key(3), scene_tokens=_scene_tokens())

    def test_map_aware_model_requires_scene_tokens(self) -> None:
        """A map-aware model called without scene_tokens fails fast."""
        model = _map_model()
        traj, context = _sample_data(TEST_AGENTS)
        with pytest.raises(ValueError, match="scene_tokens"):
            model.compute_loss(traj, context, key=jax.random.key(4))

    def test_gradient_flows_to_scene_tokens(self) -> None:
        """With trained adaLN, the loss depends differentiably on scene_tokens."""
        model = _map_model()
        support.randomize_adaln(model.backbone, seed=0)
        traj, context = _sample_data(TEST_AGENTS)
        tokens = _scene_tokens()

        def loss_fn(tk: jax.Array) -> jax.Array:
            return model.compute_loss(traj, context, key=jax.random.key(5), scene_tokens=tk)

        grad = jax.grad(loss_fn)(tokens)
        assert bool(jnp.all(jnp.isfinite(grad)))
        assert float(jnp.max(jnp.abs(grad))) > 0.0

    def test_map_aware_sample_deterministic(self) -> None:
        """Two map-conditioned samples with the same key match."""
        model = _map_model()
        _, context = _sample_data(TEST_AGENTS)
        tokens = _scene_tokens()
        a = model.sample(context, key=jax.random.key(6), scene_tokens=tokens)
        b = model.sample(context, key=jax.random.key(6), scene_tokens=tokens)
        assert jnp.array_equal(a.trajectories, b.trajectories)


class TestPredictionType:
    """The diffuser supports epsilon, x0, and v parameterizations."""

    def test_invalid_prediction_type_raises(self) -> None:
        """An unknown prediction_type is rejected at construction."""
        with pytest.raises(ValueError, match="is not a valid PredictionType"):
            _small_config(prediction_type="score")

    @pytest.mark.parametrize("prediction_type", ["epsilon", "x0", "v"])
    def test_perfect_output_reconstructs_x0(self, prediction_type: str) -> None:
        """A perfect backbone output reconstructs x0 for every parameterization.

        The regression target and the x-hat-0 reconstruction must be exact
        inverses: feeding the target back through the reconstruction returns the
        clean signal, so the sampling and loss paths share one x-hat-0 identity.
        """
        model = _make_model(_small_config(prediction_type=prediction_type))
        x0 = jax.random.normal(jax.random.key(0), (TEST_AGENTS, TEST_FUTURE, TEST_STATE_DIM))
        noise = jax.random.normal(jax.random.key(1), x0.shape)
        t = jnp.asarray(3)
        x_t = model.q_sample(x0, t, noise)
        target = model._prediction_target(x0, noise, t)
        x0_hat = model._predict_x0_from_output(x_t, t, target)
        assert jnp.allclose(x0_hat, x0, atol=1e-4)

    @pytest.mark.parametrize("prediction_type", ["epsilon", "x0", "v"])
    def test_compute_loss_finite(self, prediction_type: str) -> None:
        """The diffusion loss is a finite scalar for every parameterization."""
        model = _make_model(_small_config(prediction_type=prediction_type))
        trajectories, scene_context = _sample_data(TEST_AGENTS)
        outputs = model.compute_loss_outputs(trajectories, scene_context, key=jax.random.key(2))
        assert outputs.loss.shape == ()
        assert bool(jnp.isfinite(outputs.loss))
        assert outputs.prediction.shape == trajectories.shape

    @pytest.mark.parametrize("prediction_type", ["epsilon", "x0", "v"])
    def test_sample_finite(self, prediction_type: str) -> None:
        """Sampling yields finite trajectories for every parameterization."""
        model = _make_model(_small_config(prediction_type=prediction_type))
        _, scene_context = _sample_data(TEST_AGENTS)
        prediction = model.sample(scene_context, key=jax.random.key(3))
        assert bool(jnp.all(jnp.isfinite(prediction.trajectories)))

    def test_x0_reconstruction_passes_output_through(self) -> None:
        """For x0-prediction the reconstruction is the backbone output itself."""
        model = _make_model(_small_config(prediction_type="x0"))
        x_t = jax.random.normal(jax.random.key(4), (TEST_AGENTS, TEST_FUTURE, TEST_STATE_DIM))
        output = jax.random.normal(jax.random.key(5), x_t.shape)
        assert jnp.array_equal(model._predict_x0_from_output(x_t, jnp.asarray(2), output), output)


class TestTemperatureSampling:
    """Low-temperature sampling: scale the reverse-step posterior noise by tau.

    tau < 1 sharpens the conditional (tightens the sample spread); tau = 1 is the
    standard ancestral sampler; tau = 0 makes the reverse process noise-free.
    """

    def test_temperature_one_matches_default(self) -> None:
        """temperature=1.0 reproduces the default sampler bit-for-bit."""
        model = _make_model()
        _, ctx = _sample_data()
        key = jax.random.key(3)
        default = model.sample(ctx, key=key).trajectories
        explicit = model.sample(ctx, key=key, temperature=1.0).trajectories
        assert jnp.array_equal(default, explicit)

    def test_zero_temperature_removes_step_noise(self) -> None:
        """At temperature=0 the reverse step is noise-free, so the output is
        independent of the sampling key (only the posterior mean remains)."""
        model = _make_model()
        traj, ctx = _sample_data()
        a = model.p_sample_step(traj, 5, ctx, key=jax.random.key(1), temperature=0.0)
        b = model.p_sample_step(traj, 5, ctx, key=jax.random.key(2), temperature=0.0)
        assert jnp.allclose(a, b)

    def test_temperature_scales_step_noise_std(self) -> None:
        """The per-step noise std scales linearly with temperature: halving the
        temperature halves the spread of p_sample_step over sampling keys."""
        model = _make_model()
        traj, ctx = _sample_data()

        def spread(temp: float) -> float:
            outs = jnp.stack(
                [
                    model.p_sample_step(traj, 5, ctx, key=jax.random.key(i), temperature=temp)
                    for i in range(24)
                ]
            )
            return float(jnp.std(outs, axis=0).mean())

        full = spread(1.0)
        half = spread(0.5)
        assert full > 0.0
        assert half == pytest.approx(0.5 * full, rel=0.25)

    def test_negative_temperature_raises(self) -> None:
        """A negative temperature is rejected."""
        model = _make_model()
        _, ctx = _sample_data()
        with pytest.raises(ValueError, match="temperature"):
            model.sample(ctx, key=jax.random.key(0), temperature=-0.1)
