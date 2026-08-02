# API Reference

Complete API documentation for all Simulacrax modules, auto-generated from
source code docstrings.

## Module Groups

| Section | Description |
|---------|-------------|
| [Core](#core) | Domain types, constants, and shared configuration validators |
| [Data](#data) | WOD ingestion, tokenization, parsing, and format conversion |
| [Models](#models) | Trajectory diffusion model, trainer, and checkpointing |
| [Physics](#physics) | Kinematic constraints and physics-informed loss functions |
| [Alignment](#alignment) | Reward functions, preference construction, DPO fine-tuning, and steering |
| [Evaluation](#evaluation) | Motion metrics, realism metametric, runner, and dashboard |
| [Occupancy](#occupancy) | FNO occupancy flow model, rasterizer, and continuity losses |
| [Sensor](#sensor) | Differentiable LiDAR, NeRF rendering, and weather augmentation |
| [SDK](#sdk) | `ScenarioMiner` high-level entry point and its configuration |

## How It Works

API pages use the `:::` directive to auto-generate documentation from Python
docstrings:

```markdown
::: simulacrax.data.tokenizer
```

This pulls class signatures, method documentation, type annotations, and
inheritance hierarchies directly from the source code. Google-style docstrings
are parsed automatically.

---

## Core

Domain types, constants, and shared configuration validators.

| Module | Description |
|--------|-------------|
| [types](core/types.md) | Immutable domain types: `AgentState`, `SceneContext`, `MapFeature`, and `StrEnum` vocabularies |
| [config](core/config.md) | Shared configuration field validators |
| [constants](core/constants.md) | WOD/WOSAC constants, scenario-dict keys, and estimator configs |
| [geometry](core/geometry.md) | Signed road-edge distance, `RoadEdges`, off-road penalties |
| [distributed](core/distributed.md) | Device mesh creation and data-parallel sharding |
| [training_utils](core/training_utils.md) | NaN-safe gradient filtering |

## Data

WOD data ingestion, differentiable scene tokenization, and format conversion.

| Module | Description |
|--------|-------------|
| [wod_source](data/wod_source.md) | `DataSourceModule` for Waymo Open Dataset TFRecords (eager and streaming modes) |
| [tokenizer](data/tokenizer.md) | `SceneTokenizer` — capability-negotiated multi-modal embedding pipeline |
| [encoders](data/encoders.md) | Modality encoders: agent, map, ego, LiDAR, camera, and cross-modal fusion |
| [operators](data/operators.md) | `AgentNormalizationOperator`, `MapCroppingOperator`, `TemporalStackingOperator` |
| [parsers](data/parsers.md) | Raw WOD dict to typed domain object parsers |
| [converters](data/converters.md) | JAX / WOD submission format converters |

## Models

Trajectory generation models, physics-informed training loop, and checkpointing.

| Module | Description |
|--------|-------------|
| [trajectory_diffusion](models/trajectory_diffusion.md) | `TrajectoryDiffusionModel` — DDPM with FactorizedSceneBackbone |
| [factorized_backbone](models/factorized_backbone.md) | `FactorizedSceneBackbone` — factorized temporal + social backbone for trajectory diffusion |
| [trainer](models/trainer.md) | `TrajectoryTrainer` — physics-informed training loop with adaptive weighting |
| [checkpointing](models/checkpointing.md) | `SimulacraxCheckpointManager` — Orbax-backed model state persistence |

## Physics

Kinematic vehicle constraints and composite physics-informed loss functions.

| Module | Description |
|--------|-------------|
| [kinematics](physics/kinematics.md) | `BicycleModelConstraint`, `AckermanSteeringConstraint` — differentiable vehicle dynamics |
| [losses](physics/losses.md) | `SimulacraxPhysicsLoss` — adaptive-weighted kinematic + collision + boundary penalties |

## Alignment

Safety reward functions, DPO preference construction, and trajectory alignment training.

| Module | Description |
|--------|-------------|
| [rewards](alignment/rewards.md) | `CollisionReward`, `KinematicReward`, `BoundaryReward`, `ComfortReward`, `SafetyReward` |
| [preferences](alignment/preferences.md) | `PreferencePairBuilder`, `PreferenceBatch` — ranked pair construction for DPO |
| [dpo_trainer](alignment/dpo_trainer.md) | `DPOAlignmentTrainer` — Diffusion-DPO with Monte Carlo log-probability estimation |
| [scenario_steering](alignment/scenario_steering.md) | `ScenarioSteeringTrainer` — ranked-DPO steering toward a target scenario type |
| [steering_spine](alignment/steering_spine.md) | `sample_and_score`, `build_steering_pairs` — shared candidate scoring and gating |
| [weight_soup](alignment/weight_soup.md) | `make_weight_soup` — parameter interpolation between base and steering expert |

## Evaluation

Motion metrics, WOSAC-style realism scoring, batch orchestration, and reporting.

| Module | Description |
|--------|-------------|
| [metrics](evaluation/metrics.md) | ADE/FDE/miss-rate functions, `MotionMetrics`, `SimAgentMetrics` |
| [estimators](evaluation/estimators.md) | WOSAC histogram/Bernoulli log-likelihood estimators |
| [wosac_metametric](evaluation/wosac_metametric.md) | `WosacMetametric` — per-feature realism scoring vs the 2025 config |
| [runner](evaluation/runner.md) | `EvaluationRunner` and the `TrajectorySampler` model seam |
| [dashboard](evaluation/dashboard.md) | `MetricsDashboard` — CSV/HTML table export via calibrax |

## Occupancy

Fourier Neural Operator occupancy flow prediction.

| Module | Description |
|--------|-------------|
| [flow_model](occupancy/flow_model.md) | `OccupancyFlowModel` — multi-scale FNO with validated config coupling |
| [rasterizer](occupancy/rasterizer.md) | `SceneRasterizer` — vectorized agent/map rasterization to grids |
| [losses](occupancy/losses.md) | `FlowConsistencyLoss` — continuity-equation residual on interior cells |

## Sensor

Differentiable sensor simulation for perception testing.

| Module | Description |
|--------|-------------|
| [lidar](sensor/lidar.md) | `LiDARRayCaster` — differentiable spherical ray marching |
| [nerf_renderer](sensor/nerf_renderer.md) | `NeRFRenderer` — camera-relative neural rendering |
| [weather](sensor/weather.md) | Rain/fog/snow augmentation operators (datarax `ModalityOperator`) |

## SDK

The high-level public entry point.

| Module | Description |
|--------|-------------|
| [scenario_miner](api/scenario_miner.md) | `ScenarioMiner` — generate, steer, evaluate, and adversarially search |
| [config](api/config.md) | `MinerConfig`, `Scenario`, `FailureCase` containers |
