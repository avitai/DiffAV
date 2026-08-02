# Simulacrax: Evaluation Engine for Autonomous Driving

Simulacrax is a physics-informed, RL-aligned evaluation engine for autonomous vehicle
motion planning. It generates realistic, adversarial, and physically plausible
counterfactual scenarios to evaluate the safety and performance of AV stacks.

Built on the JAX ecosystem with [datarax](https://github.com/avitai/datarax),
[artifex](https://github.com/avitai/artifex),
[opifex](https://github.com/avitai/opifex), and
[calibrax](https://github.com/avitai/calibrax).

## Key Features

- **Waymo Open Dataset integration** -- ingest WOD TFRecords with CPU-only TensorFlow
  parsing, avoiding GPU memory conflicts with JAX.
- **Immutable domain types** -- frozen dataclasses for `AgentState`, `SceneContext`,
  `MapFeature`, `TrajectoryPrediction`, and evaluation results.
- **Protocol-based architecture** -- structural subtyping protocols decouple tokenization,
  generation, physics validation, and evaluation stages.
- **datarax-native data pipeline** -- `WODSource` extends `DataSourceModule`, yielding
  standard `Element` objects compatible with the datarax operator ecosystem.
- **JAX-first design** -- all numerical computation uses JAX arrays with JIT compilation,
  automatic differentiation, and hardware acceleration.

## Quick Install

```bash
# Clone and set up the environment (auto-detects CUDA/Metal/CPU)
git clone https://github.com/avitai/simulacrax.git
cd simulacrax
./setup.sh
```

## Quick Navigation

<div class="grid cards" markdown>

-   :material-rocket-launch: **Getting Started**

    ---

    Installation, quick start, and core concepts.

    [:octicons-arrow-right-24: Get started](getting-started/installation.md)

-   :material-book-open-variant: **Examples**

    ---

    Hands-on examples from WOD loading to evaluation.

    [:octicons-arrow-right-24: Browse examples](examples/overview.md)

-   :material-api: **API Reference**

    ---

    Complete API documentation for all modules.

    [:octicons-arrow-right-24: API docs](api/index.md)

-   :material-hammer-wrench: **Development**

    ---

    Contributing guide and architecture overview.

    [:octicons-arrow-right-24: Contribute](development/contributing.md)

</div>
