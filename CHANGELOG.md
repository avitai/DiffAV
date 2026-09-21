# Changelog

All notable changes to DiffAV are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Security

- The lock moves anyio from 4.12.1 to 4.14.2 for CVE-2026-63374 and CVE-2026-64847; nothing else moves. 4.14.2 is the first fixed release; 4.15.1 needs typing-extensions 4.16.0, which a single-package upgrade does not allow to move.

### Changed

- Requires `substrax>=0.1.11`, `datarax>=0.1.14`, `avitai-artifex>=0.1.12`, `opifex>=0.2.8`
  and `calibrax>=0.1.9`, the latest releases. substrax 0.1.11 caps jax below 0.11.2, whose
  renamed `HiPrimitive` breaks `import flax.nnx` on flax 0.12.9.
- Optimizers come from `substrax.optim`: opifex 0.2.7 removed its `OptimizerConfig` and
  `create_optimizer`. `create_optimizer(model, config)` returns the `nnx.Optimizer`
  directly, gradient clipping is `gradient_clip_norm`, and a learning-rate schedule is an
  optax schedule passed as `learning_rate` (`schedule_type`, `peak_value` and the step
  counts are gone). The physics-informed tutorial's warmup-cosine is now
  `optax.warmup_cosine_decay_schedule` peaking at 2e-4. The trainer reports the learning
  rate read from the optimizer state inside the jitted step, and `nan` for an optimizer
  that carries none.
- Checkpoints are written in substrax's format 3: the model and optimizer are separate
  items beside a record holding the epoch, the metrics with `loss`, diffav as the producer
  and the architecture version. A root written by an earlier release still restores,
  through `DIFFAV_FORMAT2`, and `substrax.checkpoint.upgrade_checkpoints(src, dst,
  legacy_layout=DIFFAV_FORMAT2)` rewrites it in format 3. `CheckpointConfig.checkpoint_dir`
  is required: a shared default let two runs collide at the same steps.
- `scripts/train_wod.py` refuses a `--checkpoint-dir` that already holds checkpoints,
  before reading any data: a run trains from step 0 and the store refuses a step that
  exists or lies below the latest, so the first save would otherwise fail after
  `--checkpoint-every` steps. The Modal launcher's `--checkpoint` names the root on the
  Volume for every task, defaulting to the staged `wod-physics-tier0`; a training run
  passes an unused one.
- The five examples that save figures resolve their directory through
  `substrax.artifacts.resolve_output_dir("examples")`: with `AVITAI_OUTPUT_DIR` set the
  figures land in `examples` under it (`AVITAI_OUTPUT_DIR="$PWD/docs/assets/images"`
  regenerates the documentation figures in place), otherwise in a per-process temporary
  directory. `DIFFAV_EXAMPLES_OUTPUT_DIR` is gone. The example tests run each example in
  its own interpreter through `substrax.testing.run_example`, on the CPU backend, and skip
  only on the WOD-not-configured message; a run over `DIFFAV_EXAMPLE_TIMEOUT_SECONDS`
  fails, naming the budget. Requires `substrax>=0.1.7`, the locked release.

### Added

- The Modal launcher runs examples on a GPU (`--task examples`, an L4 by default) with
  deterministic kernels and WOD from the data Volume, writing each script's log and
  figures to the `diffav-example-outputs` volume and fetching them into `--out`;
  `--task fetch` downloads a run whose client detached.
- The Modal launcher runs the test-time guidance sweep (`--task guidance`) against the
  staged checkpoint, as it already ran the DPO ablation; both benchmarks write their CSV
  under the checkpoints Volume. The deploy guide stages only the training shards a run
  reads: `train_wod.py` streams shards in name order and stops at `--max-scenarios`, so
  the recorded `wod-physics-tier0` run needs the first 17 shards, not the full split.

### Changed

- Requires `datarax>=0.1.10`, whose operators take the record's PRNG key as the fourth
  argument of `apply` and draw from it; `generate_random_params` is gone. The rain, fog and
  glare augmentations draw each record's intensity from that key in stochastic mode,
  uniformly in `[0, intensity)`, and refuse to run without one rather than give every
  record the configured value; the deterministic operators name the argument `key` and
  ignore it. The lock moves datarax from 0.1.9 to 0.1.10 and substrax from 0.1.5 to 0.1.7.
- No stored checkpoint restores into a model built by this release, for two reasons.
  artifex 0.1.3 (2026-08-01) added `one_minus_alphas_cumprod` and
  `one_minus_alphas_cumprod_prev` to the noise schedule's state, and every checkpoint
  written before it (`wod-mini`, `wod-physics-tier0`, `wod-physics-showcase`) lacks the two
  leaves, so `DiffAVCheckpointManager`, which restores raw module state, has refused them
  since the lock took artifex 0.1.5. A checkpoint of a model that embeds a `SceneTokenizer`
  (`wod-physics-tier0`, `wod-physics-showcase`) additionally carries operator `rngs` and
  statistics leaves that datarax 0.1.10 no longer keeps. The examples that restore
  `wod-mini` fall back to an untrained model only when the directory is absent, as it is in
  CI; with the directory present they raise `CheckpointCorruptError`.

### Fixed

- A checkpoint is labelled with the updates it holds. `TrajectoryTrainer` saved under the
  index of the step it had just run, so the checkpoint at step 1000 held 1001 updates, and a
  resumed run restored step 1000, ran it again and saved step 1000 a second time, which
  format 3 refuses. It now saves after counting the step, and a resumed run continues at
  the next one. `scripts/train_wod.py` labels its periodic saves the same way and leaves a
  multiple of `--checkpoint-every` that equals `--steps` to its final save, which it used to
  collide with. A root written before this release holds one more update than its label.
- Training on Modal failed before the first step with `No module named 'IPython'`: the
  trainer reaches `fastprogress` through opifex, artifex and blackjax, and fastprogress
  1.1.5 imports IPython at module level without declaring it, which the image, installed
  without development tools, does not have. The lock takes fastprogress 1.1.6, where that
  import happens only when a notebook display is requested, and a test imports the trainer
  in a child interpreter with IPython blocked.
- `WODSource.element_spec` declares the dtypes `get_batch_at` emits. The batches are JAX
  arrays, so `int64` scenario fields such as `state/all/valid` arrive as `int32` while
  x64 is off, but the spec described the host arrays and declared `int64`. The spec is now
  datarax's `device_spec` of the first scenario, which needs `datarax>=0.1.9`: from that
  release `array_to_spec` describes a value exactly as given.

## [0.1.2] - 2026-09-09

### Changed

- `resolve_wrapped_indices` is imported from `datarax.sources`, its public home in
  datarax 0.1.7; the private `datarax.sources._source_base` path this package used is
  gone there, so the floor is `datarax>=0.1.7`.

## [0.1.1] - 2026-09-09

### Changed

- The Avitai siblings install from PyPI at `datarax>=0.1.6`, `avitai-artifex>=0.1.5`,
  `opifex>=0.2.2`, `calibrax>=0.1.5` and the new `substrax>=0.1.4`; the git-tag sources
  and the calibrax override are gone. The runtime floors are jax 0.11.1, flax 0.12.9,
  orbax-checkpoint 0.11.33, numpy 2.1 and Python 3.12.
- `DiffAVCheckpointManager` composes substrax's `OrbaxCheckpointStore` directly (the
  opifex path it imported is a re-export of the same class). Checkpoints written by earlier
  releases restore unchanged: substrax 0.1.5 places every array on the device of its target
  leaf, so a run saved on `cuda:0` restores in a CPU-only process. A step whose data cannot
  be read or does not match the model raises `CheckpointCorruptError` from the store's error
  instead of relying on the store to report it as missing.
- Per-step FLOPs are XLA's cost analysis of the lowered step (calibrax 0.1.3); the figures
  are larger than the parameter-count estimate they replace, and a step containing a custom
  call with no cost model raises `FlopsUnavailableError`.
- artifex 0.1.5 builds encoder dropout whenever `dropout_rate > 0`; the named `dropout`
  rng stream the tokenizer and `scripts/train_wod.py` added to activate it is gone, and the
  encoder tests assert the behaviour with plain rngs.

### Removed

- `diffav.core.distributed` (`DistributedConfig`, `create_device_mesh`, `shard_batch`,
  `get_data_parallel_sharding`). The mesh comes from substrax:
  `DeviceMeshManager.create_device_mesh({"data": jax.device_count()})`, the batch sharding
  from `substrax.spmd.create_data_parallel_sharding` and the placement from
  `substrax.spmd.place_batch_on_shards`. The `(data, model, pipeline)` mesh shape, the
  `"fsdp"`/`"mp"` placeholders that raised `NotImplementedError`, and the single-device
  collapse of any requested shape are gone with it; the trainers' `train_step_distributed`
  is unchanged and takes any `jax.sharding.Mesh`.

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
