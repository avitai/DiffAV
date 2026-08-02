# Simulacrax

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/JAX-0.6%2B-green)](https://github.com/google/jax)
[![Flax](https://img.shields.io/badge/Flax-NNX-orange)](https://github.com/google/flax)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## A physics-informed, RL-aligned evaluation engine for autonomous driving

*From Latin "simulacrum" — image, likeness, semblance*

---

> **Early Development — API Unstable**
>
> Simulacrax is in early development and undergoing rapid iteration.
> Breaking changes are expected. Pin to specific commits if stability is required.

---

## Overview

Simulacrax generates and evaluates adversarial driving scenarios to stress-test autonomous vehicle (AV) stacks. It combines generative trajectory prediction with physics constraints and reinforcement learning alignment to produce realistic yet safety-critical counterfactual scenarios, evaluated against the [Waymo Open Dataset Sim Agents Challenge (WOSAC)](https://waymo.com/open/challenges/sim-agents/) metrics.

### Why Simulacrax?

- **Physics-Informed**: Bicycle model kinematics constraints ensure generated trajectories are physically plausible — no teleporting vehicles or impossible accelerations
- **RL-Aligned**: Reinforcement learning fine-tuning steers generation toward safety-critical scenarios that expose AV stack weaknesses
- **JAX-Native**: Built entirely on JAX with Flax NNX modules for JIT compilation, automatic differentiation, and hardware acceleration
- **WOSAC-Compatible**: Evaluates against standard Waymo challenge metrics (ADE, FDE, collision rate, miss rate), computed as JAX-native Euclidean proxies rather than the official leaderboard implementation
- **Modular Architecture**: Clean protocol-based design with frozen dataclass configuration — easy to swap generative models, physics validators, and evaluation metrics

## Status & Results

Simulacrax is a working research scaffold: every pillar (diffusion world model, DPO/steering alignment, physics feasibility, occupancy flow, evaluation metrics) is built, and its core has been exercised on real Waymo Open Dataset data. It is not a leaderboard-tuned system — results are reported honestly.

- **Trajectory prediction.** A map-conditioned diffusion baseline trained on real WOD reaches **minADE₆ ≈ 5.6 m** on held-out validation (`tracks_to_predict`, WOMD 2 Hz). That is roughly 9× the ~0.6 m of full-scale WOSAC leaders: the model is **over-dispersed** (best-of-64 ≈ 2.5 m) and **under-fit**, trained on ~1.6% of WOMD. The map is load-bearing — zeroing the scene tokens degrades minADE ~8.6× (4.4 → 38 m).
- **Adversarial steering (the differentiator).** Test-time reward guidance on the frozen baseline steers a real scene's adversary from **7.30 m → 2.88 m** from its victim while *improving* off-road feasibility (0.21 → 0.08) and holding WOSAC realism (0.673 → 0.653). A map-conditioned Diffusion-DPO fine-tune reproduces this at training time.
- **Roadmap.** Close the trajectory-quality gap (scale up, fix over-dispersion), then a persistent-WOSAC leaderboard submission and GRPO/R1-style RL fine-tuning (now standard in the winning recipe).

The map-conditioned model (`MapConditionedTrajectoryModel`) and its trainer are driven end-to-end by [`scripts/train_wod.py`](scripts/train_wod.py) — warmup-cosine schedule, EMA, and held-out validation.

## Design

### Generative Trajectory Prediction

Simulacrax models multi-agent future trajectories conditioned on scene context (ego state, surrounding agents, HD map features). The generative backbone is a diffusion model:

- **Diffusion Models** — iterative denoising for high-quality multi-modal trajectory distributions

The protocol-based design leaves room for other families (normalizing flows, score-based models); those are not yet implemented.

All models predict in the WOSAC state space: 11 history steps (1.1s) conditioning 80 future steps (8.0s) at 10Hz, with state dimension (x, y, heading, velocity).

### Physics Constraints

Generated trajectories pass through a differentiable physics validation layer based on the bicycle kinematic model. Constraints include maximum acceleration, curvature bounds, and collision detection. Physics violations feed back as a loss term during training, ensuring the generator learns to produce plausible trajectories without post-hoc rejection sampling.

### Adversarial Alignment

An RL fine-tuning stage (DPO) optimizes the generator to produce scenarios that are simultaneously realistic (high WOSAC scores) and challenging (expose planning failures). This closes the loop between generation quality and safety-critical scenario discovery.

## Architecture

```
src/simulacrax/
  core/         # Domain types, protocols, and configuration
  data/         # WOD TFRecord loading and scene tokenization
  models/       # Trajectory prediction model wrappers (diffusion)
  physics/      # Bicycle model kinematics and constraint validation
  alignment/    # RL fine-tuning for adversarial scenario generation
  evaluation/   # WOSAC metrics (ADE, FDE, collision rate) and orchestration
  occupancy/    # Fourier Neural Operator occupancy flow prediction
  sensor/       # NeRF/LiDAR sensor simulation for perception testing
  api/          # Public SDK for scenario mining and evaluation
```

### Sister Repositories

Simulacrax builds on four companion libraries in the JAX ecosystem:

| Repository | Role | Key Components Used |
|---|---|---|
| [Datarax](https://github.com/avitai/datarax) | Data pipelines | Operators, cross-modal operators, TFDS sources |
| [Artifex](https://github.com/avitai/artifex) | Generative models | DiffusionModel (DiT), RL rewards/trainers, noise schedules |
| [Opifex](https://github.com/avitai/opifex) | Scientific ML / Physics | Optimizers, EMA & checkpointing, error recovery, adaptive physics-weight scheduling, and multi-scale Fourier neural operators |
| [Calibrax](https://github.com/avitai/calibrax) | Profiling / Benchmarking | Roofline analysis, memory profiling |

## Installation

```bash
# Clone and set up the environment (auto-detects CUDA/Metal/CPU)
git clone https://github.com/avitai/simulacrax.git
cd simulacrax
./setup.sh

# Or manually with uv (dev tooling is installed by default)
uv sync                 # CPU
uv sync --extra gpu     # Linux with NVIDIA GPU (CUDA 12)
```

### Requirements

- Python 3.11, 3.12, or 3.13
- JAX >= 0.6.1
- Flax >= 0.12.0
- TensorFlow >= 2.20.0 (CPU-only, for WOD proto parsing)
- Waymo Open Dataset access (requires [license agreement](https://waymo.com/open/))

## Quick Start

```python
from simulacrax.api import MinerConfig, create_scenario_miner

# Configure and create the scenario miner
config = MinerConfig(model_path="checkpoints/wod-mini")
miner = create_scenario_miner(config)

# Generate scenarios and evaluate a planner
scenarios = miner.generate("unprotected_left_turn", density="high", count=100)
report = miner.evaluate_planner(my_planner_fn, scenarios)
print(report.metric_values)  # {"ade": ..., "fde": ...}
```

## Development

Simulacrax uses `uv` as its package manager and enforces code quality via pre-commit hooks.

### Running Tests

```bash
# Core unit tests
uv run pytest tests/core/ -v

# Full test suite with coverage
uv run pytest -v --cov=src/simulacrax --cov-report=term-missing

# Skip tests requiring WOD data
uv run pytest -v -m "not wod"
```

### Code Quality

```bash
# All pre-commit hooks
uv run pre-commit run --all-files

# Individual checks
uv run ruff check src/          # Lint
uv run ruff format src/         # Format
uv run pyright src/             # Type check
```

## License

Simulacrax is licensed under the [MIT License](LICENSE).
