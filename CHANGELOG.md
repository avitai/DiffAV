# Changelog

All notable changes to DiffAV are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Scenario steering strategies**: ranked-DPO training on reward-ranked,
  feasibility-gated pairs sampled from the current policy; weight-soup
  parameter interpolation between the base model and a fine-tuned expert;
  and test-time reward-gradient guidance in the diffusion model's
  clean-estimate space — all selectable from one `steer()` configuration.
- **WOSAC-style realism metametric**: per-feature histogram log-likelihood
  scoring against the 2025 challenge configuration, with reference-pinned
  estimators and kinematic features derived via central differences.
- **Steering bake-off harness** in `benchmarks/` scoring adversariality,
  realism, and feasibility per strategy/strength cell with calibrax
  reporting, plus Modal launchers (`deploy/modal_app.py`) for managed
  A100/H100 runs.
- **Batched scenario generation**: `MinerConfig.batch_size` drives vmapped
  sampling chunks with per-scenario keys.
- **Typed evaluation seam**: the `TrajectorySampler` protocol at the
  evaluation runner, a PEP 561 `py.typed` marker, and import-linter
  contracts covering every package layer.
- **CI workflow suite**: marker-tiered fast gate, full-suite coverage gate,
  quality/security/build/docs/publish lanes, a benchmark regression guard
  with a persistent baseline, and a Docker build lane.
- **Container image**: CUDA 12.6 runtime base with CPU fallback, frozen
  lockfile sync, and a non-root runtime user.

### Changed

- **BREAKING — the project is renamed from Simulacrax to DiffAV.** The
  distribution and import package are now `diffav` (`import diffav`,
  `pip install diffav`); the repository is
  [avitai/DiffAV](https://github.com/avitai/DiffAV) and the documentation
  moves to <https://diffav.readthedocs.io>. The `SIMULACRAX_*` environment
  variables are now `DIFFAV_*`, the generated backend file is `.diffav.env`,
  and the `Simulacrax`-prefixed classes (`SimulacraxPhysicsLoss`,
  `SimulacraxPhysicsConfig`, `SimulacraxCheckpointManager`,
  `SimulacraxScenarioMiner`) are `DiffAV`-prefixed. No release carried the
  old name, so no compatibility shim is provided.
- `adversarial_search` explores exactly its budget of candidates, filters by
  a severity threshold, accepts caller-supplied keys with a documented
  reproducibility contract, and attaches predictions regenerated on the
  perturbed context.
- The metrics dashboard renders the SDK's own report shape as an overall
  group and fails fast on unrenderable reports or unknown key prefixes.
- The road-boundary penalty is a hinge on the signed distance to oriented
  road edges; physics losses apply to model predictions with a traced epoch.
- DPO log-probability estimation shares (timestep, noise) draws between the
  policy and reference passes so sampling noise cancels in the log-ratio.
- Trainer stability handling wires real check/rollback/recovery semantics,
  honest NaN detection, opt-in FLOPs profiling, and truthful learning-rate
  reporting; both distributed paths compile their step exactly once.
- WOD parsing is validity-aware (padding rows never become agents) and
  submissions emit the full proto field set.
- Sensor simulation fixes: streak density semantics, LiDAR grid anchoring
  and step-scaled opacity, wired horizontal field of view, and
  camera-relative NeRF encoding.
- Occupancy continuity loss differentiates along the correct grid axes with
  interior-only stencils; configuration coupling is validated.

### Removed

- The constant-reward steering blend (zero gradient) and the synthetic
  preference pairs built from alternating identical samples.
- Dead abstractions: unused core config dataclasses, unconsumed protocols,
  orphaned domain types, unused data fields, and the bespoke CI workflows.
