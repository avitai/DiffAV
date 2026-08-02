# Examples Overview

Simulacrax examples follow a three-tier system matching the
[example documentation design](../development/example_documentation_design.md).

<div class="grid cards" markdown>

-   :material-speedometer:{ .lg .middle } **Quick Start**

    ---

    Load WOD data and generate trajectories in under 10 minutes.

    [:octicons-arrow-right-24: WOD Loading Quick Reference](core/wod-loading-quickref.md)

-   :material-robot-outline:{ .lg .middle } **Physics-Informed Training**

    ---

    Train a trajectory diffusion model with bicycle model constraints and adaptive weighting.

    [:octicons-arrow-right-24: Physics-Informed Training Tutorial](models/physics-informed-training-tutorial.md)

-   :material-tune:{ .lg .middle } **DPO Alignment**

    ---

    Fine-tune a trajectory model toward safer behaviour using Direct Preference Optimization.

    [:octicons-arrow-right-24: DPO Fine-Tuning Quick Reference](alignment/dpo-finetuning-quickref.md)

</div>

---

## Documentation Tiers

| Tier | Name | Runtime | Audience | Code ratio |
|------|------|---------|----------|------------|
| **1** | Quick Reference | ~5–10 min (CPU) | All levels | 70% code / 30% explanation |
| **2** | Tutorial | ~15–30 min (CPU) | Beginner / Intermediate | 50% code / 50% explanation |
| **3** | Advanced Guide | ~30–60 min (CPU) | Advanced / Production | 40% code / 60% explanation |

---

## Available Examples

### Core

| Example | Tier | Runtime | Description |
|---------|------|---------|-------------|
| [WOD Loading Quick Reference](core/wod-loading-quickref.md) | 1 | ~5 min | Load WOD scenarios, iterate Elements, parse to `SceneContext` |
| [Scene Tokenization Tutorial](core/scene-tokenization-tutorial.md) | 2 | ~8 min | Multi-modal embedding pipeline: agents, map, ego, lidar, camera |
| [End-to-End Quick Reference](core/end-to-end-quickref.md) | 1 | ~5 min | Generate scenarios, evaluate a planner, run adversarial search |

### Models

| Example | Tier | Runtime | Description |
|---------|------|---------|-------------|
| [Trajectory Diffusion Quick Reference](models/trajectory-diffusion-quickref.md) | 1 | ~3 min | Build, train, and sample from the trajectory diffusion model |
| [Physics-Informed Training Tutorial](models/physics-informed-training-tutorial.md) | 2 | ~5 min | Training with bicycle model constraints, adaptive weighting, JIT |

### Physics

| Example | Tier | Runtime | Description |
|---------|------|---------|-------------|
| [Bicycle Model Quick Reference](physics/bicycle-model-quickref.md) | 1 | ~1 min | Kinematic constraints, trajectory validation, Ackerman steering |

### Alignment

| Example | Tier | Runtime | Description |
|---------|------|---------|-------------|
| [Preference Construction Quick Reference](alignment/preference-construction-quickref.md) | 1 | ~10 min | Score candidates with safety rewards, build chosen/rejected pairs |
| [DPO Fine-Tuning Quick Reference](alignment/dpo-finetuning-quickref.md) | 1 | ~10 min | Fine-tune trajectory diffusion with Diffusion-DPO alignment |
| [DPO Fine-Tuning Tutorial](alignment/dpo-finetuning-tutorial.md) | 2 | ~15 min | Full DPO pipeline plus ranked-pair scenario steering |

### Evaluation

| Example | Tier | Runtime | Description |
|---------|------|---------|-------------|
| [WOD Metrics Quick Reference](evaluation/wod-metrics-quickref.md) | 1 | ~5 min | ADE/FDE/miss-rate functions and the MotionMetrics aggregator |
| [Evaluation Pipeline Tutorial](evaluation/evaluation-pipeline-tutorial.md) | 2 | ~15 min | Runner orchestration, sim-agent realism, dashboard export |

### Advanced

| Example | Tier | Runtime | Description |
|---------|------|---------|-------------|
| [Occupancy Flow Tutorial](advanced/occupancy-flow-tutorial.md) | 3 | ~30 min | FNO occupancy prediction with continuity-loss regularisation |
| [ScenarioMiner Guide](advanced/scenario-miner-guide.md) | 3 | ~30 min | SDK deep dive: generation, evaluation, adversarial search |
| [Sensor Simulation Tutorial](advanced/sensor-simulation-tutorial.md) | 3 | ~30 min | Differentiable LiDAR, NeRF rendering, weather augmentation |

---

## Running Examples

All examples are Jupytext percent-format Python scripts with paired `.ipynb` notebooks.

```bash
# Run as a Python script
uv run python examples/core/01_wod_loading_quickref.py

# Open as a Jupyter notebook
jupyter lab examples/core/01_wod_loading_quickref.ipynb
```

To regenerate notebooks from source after editing the `.py` file:

```bash
uv run python scripts/jupytext_converter.py py-to-nb examples/core/01_wod_loading_quickref.py
```

---

## Example Format

Each example includes:

1. **Metadata table** — level, runtime, prerequisites, format
2. **Overview** — what the example demonstrates and why it matters
3. **Learning goals** — measurable objectives (action verbs: Create, Build, Verify)
4. **Files** — links to both the Python script and the Jupyter notebook
5. **Quick Start** — copy-paste bash commands to run immediately
6. **Key Concepts** — step-by-step code with explanations and terminal output
7. **Sister Repo Components** — datarax / artifex / opifex building blocks used
8. **Related** — next steps, API references, and linked examples

---

## For Contributors

<div class="grid cards" markdown>

-   :material-file-document-edit-outline:{ .lg .middle } **Documentation Design Guide**

    ---

    Complete standards for writing new examples, including tier guidelines,
    output capture, migration tables, and the quality checklist.

    [:octicons-arrow-right-24: Example Documentation Design](../development/example_documentation_design.md)

-   :material-language-python:{ .lg .middle } **Example Template**

    ---

    Jupytext percent-format template with all required sections pre-filled.

    [:octicons-arrow-right-24: examples/_templates/example_template.py](https://github.com/avitai/simulacrax/blob/main/examples/_templates/example_template.py)

</div>
