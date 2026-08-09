# Architecture

DiffAV is structured as a modular evaluation pipeline for autonomous driving,
built on the JAX ecosystem with integration across four sister repositories.

## High-Level Architecture

```mermaid
graph TB
    subgraph "Data Ingestion"
        WOD[WOD TFRecords] --> WS[WODSource]
        WS --> P[Parsers]
        P --> SC[SceneContext]
    end

    subgraph "Tokenization"
        SC --> ST[SceneTokenizer]
        ST --> TS[scene embedding dict]
    end

    subgraph "Generation"
        TS --> TG[TrajectoryDiffusionModel]
        TG --> TP[TrajectoryPrediction]
    end

    subgraph "Validation & Evaluation"
        TP --> PV[DiffAVPhysicsLoss]
        PV --> VR[loss components]
        TP --> EV[EvaluationRunner]
        EV --> MR[MetricsReport]
    end

    subgraph "Output"
        TP --> CV[Converters]
        CV --> SUB[WOD Submission]
    end
```

### Trajectory model architecture

The `TrajectoryDiffusionModel` node above is backed by a factorized design.
The denoiser is a `FactorizedSceneBackbone` that interleaves separate temporal
and social attention blocks, configured by `num_blocks`, `num_temporal_layers`,
and `num_social_layers`. Map conditioning is applied through adaLN-gated
cross-attention to scene tokens, enabled by `use_map_cross_attention`, with
`MapConditionedTrajectoryModel` serving as the map-conditioned entrypoint.
Diffusion is performed in per-agent local frames: the denoiser predicts in each
agent's own local frame, with one context row per agent.

## Module Dependency Graph

```mermaid
graph LR
    subgraph "diffav"
        core[core.types<br/>core.constants<br/>core.config]
        data[data.wod_source<br/>data.parsers<br/>data.encoders]
        models[models]
        physics[physics]
        alignment[alignment]
        evaluation[evaluation]
        occupancy[occupancy]
        sensor[sensor]
        api[api]
    end

    subgraph "Sister Repos"
        datarax[datarax<br/>DataSourceModule<br/>operators]
        artifex[artifex<br/>Noise schedules]
        opifex[opifex<br/>Optimizers, physics]
        calibrax[calibrax<br/>Profiling, reporting]
    end

    data --> core
    data --> datarax
    data --> artifex
    models --> core
    models --> artifex
    models --> opifex
    models --> calibrax
    physics --> core
    physics --> opifex
    alignment --> core
    alignment --> models
    alignment --> physics
    alignment --> artifex
    evaluation --> core
    evaluation --> physics
    evaluation --> calibrax
    occupancy --> core
    occupancy --> opifex
    sensor --> core
    sensor --> datarax
    sensor --> opifex
    api --> alignment
    api --> models
    api --> opifex
```

## Design Principles

### Consumer-Side Decoupling

The one seam where the pipeline consumes an abstract model — the evaluation
runner — is typed by the `TrajectorySampler` protocol defined next to its
consumer in `evaluation.runner`. Any object with a matching `sample` method
can be evaluated, which enables:

- Swapping model architectures without changing evaluation code
- Testing with lightweight stub samplers
- No speculative interfaces: abstractions exist only where consumed

### Immutable Data Flow

All domain types are frozen dataclasses. Data flows through the pipeline as
immutable snapshots, enabling:

- Safe concurrent processing
- Reproducible pipeline runs
- Clear ownership semantics

### JAX/TF Coexistence

TensorFlow is used **only** for WOD TFRecord parsing on CPU. GPU visibility
is disabled before any TF operation to prevent memory conflicts with JAX:

```python
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")
```

### datarax Integration

`WODSource` extends `DataSourceModule` from datarax, following the
`TFDSEagerSource`/`TFDSStreamingSource` pattern:

- Standard `Element` iteration (data + state + metadata)
- Two loading modes: eager (all-at-init) and streaming (lazy prefetch)
- TFRecord parsing with Waymax-compatible temporal aggregation
- `StructuralConfig` validation pattern
- Compatibility with datarax batchers, operators, and the pipeline API
