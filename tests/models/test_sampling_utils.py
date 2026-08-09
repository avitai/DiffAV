"""Tests for the shared stratified diffusion-timestep sampler."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from diffav.models.sampling_utils import stratified_timestep


class TestStratifiedTimesteps:
    """Tests for the stratified Monte Carlo timestep sampler.

    A K-draw estimator partitions the timestep range into K equal strata
    and draws once per stratum (stratified sampling), so the estimate keeps
    the uniform marginal (unbiased) with strictly lower variance than K
    independent uniform draws.
    """

    def test_each_draw_falls_in_its_stratum(self) -> None:
        """With K dividing T, draw i must land in bin [i*T/K, (i+1)*T/K)."""
        num_samples, num_timesteps = 10, 100
        bin_width = num_timesteps // num_samples
        for seed in range(5):
            key = jax.random.key(seed)
            for index in range(num_samples):
                t_key, _ = jax.random.split(jax.random.fold_in(key, index))
                t = stratified_timestep(t_key, index, num_samples, num_timesteps)
                assert int(t) // bin_width == index

    def test_draws_cover_distinct_strata(self) -> None:
        """The K draws of one estimator call occupy K distinct bins."""
        num_samples, num_timesteps = 8, 96
        bin_width = num_timesteps // num_samples
        key = jax.random.key(123)
        bins = set()
        for index in range(num_samples):
            t_key, _ = jax.random.split(jax.random.fold_in(key, index))
            t = stratified_timestep(t_key, index, num_samples, num_timesteps)
            bins.add(int(t) // bin_width)
        assert bins == set(range(num_samples))

    def test_in_range_and_integer(self) -> None:
        """Draws are integer timesteps in [0, num_timesteps)."""
        num_samples, num_timesteps = 16, 4  # more samples than timesteps
        for seed in range(3):
            for index in range(num_samples):
                t_key = jax.random.fold_in(jax.random.key(seed), index)
                t = stratified_timestep(t_key, index, num_samples, num_timesteps)
                assert jnp.issubdtype(t.dtype, jnp.integer)
                assert 0 <= int(t) < num_timesteps

    def test_single_sample_marginal_covers_full_range(self) -> None:
        """K=1 degenerates to a plain uniform draw over the full range."""
        num_timesteps = 100
        draws = [
            int(stratified_timestep(jax.random.key(seed), 0, 1, num_timesteps))
            for seed in range(200)
        ]
        assert min(draws) < 10
        assert max(draws) >= 90

    def test_stratified_mean_has_lower_variance_than_uniform(self) -> None:
        """Variance of the K-draw mean timestep drops versus i.i.d. uniform."""
        num_samples, num_timesteps, num_trials = 8, 100, 300

        stratified_means = []
        uniform_means = []
        for seed in range(num_trials):
            key = jax.random.key(seed)
            strat = [
                stratified_timestep(
                    jax.random.fold_in(key, index), index, num_samples, num_timesteps
                )
                for index in range(num_samples)
            ]
            stratified_means.append(float(jnp.mean(jnp.stack(strat))))
            unif = jax.random.randint(key, (num_samples,), 0, num_timesteps)
            uniform_means.append(float(jnp.mean(unif)))

        var_stratified = float(jnp.var(jnp.array(stratified_means)))
        var_uniform = float(jnp.var(jnp.array(uniform_means)))
        assert var_stratified < var_uniform / 4
