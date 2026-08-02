# Development Examples & Utilities

> **Looking for tutorials?** See the [Documentation Examples](https://simulacrax.readthedocs.io/examples/overview/) or browse `docs/examples/` directly.

This directory contains **development examples and utilities** for the Simulacrax
evaluation engine. Examples are written as Jupytext percent-format Python scripts
with paired `.ipynb` notebooks.

## Directory Structure

```
examples/
├── _templates/                                  # Jupytext percent-format template
├── core/
│   ├── 01_wod_loading_quickref.py               # WOD data loading quick reference
│   ├── 02_scene_tokenization_tutorial.py        # Multi-modal embedding pipeline
│   └── 03_end_to_end_quickref.py                # Generate → evaluate → adversarial search
├── models/
│   ├── 01_trajectory_diffusion_quickref.py      # Trajectory diffusion at WOD scale
│   └── 02_physics_informed_training_tutorial.py # Physics-constrained training
├── physics/
│   └── 01_bicycle_model_quickref.py             # Kinematic constraint validation
├── alignment/
│   ├── 01_preference_construction_quickref.py   # Reward-ranked preference pairs
│   ├── 02_dpo_finetuning_quickref.py            # Diffusion-DPO fine-tuning
│   └── 03_dpo_finetuning_tutorial.py            # DPO + ranked-pair steering
├── evaluation/
│   ├── 01_wod_metrics_quickref.py               # ADE/FDE/miss-rate + MotionMetrics
│   └── 02_evaluation_pipeline_tutorial.py       # Runner + realism + dashboard
├── advanced/
│   ├── 01_occupancy_flow_tutorial.py            # FNO occupancy flow
│   ├── 02_scenario_miner_guide.py               # ScenarioMiner SDK deep dive
│   └── 03_sensor_simulation_tutorial.py         # LiDAR / NeRF / weather
└── README.md                                    # This file
```

Every `.py` example has a paired `.ipynb` notebook kept in sync by the
pre-commit jupytext hook.

## Running Examples

```bash
# Run as a Python script (uv resolves the project environment)
uv run python examples/core/01_wod_loading_quickref.py

# Open as a Jupyter notebook
uv run jupyter notebook examples/core/01_wod_loading_quickref.ipynb

# Re-sync a .py/.ipynb pair
uv run python scripts/jupytext_converter.py sync examples/core/01_wod_loading_quickref.py
```

## Example Tiers

| Tier | Name | Runtime | Description |
|------|------|---------|-------------|
| 1 | Quick Reference | ~5 min | Minimal, focused code snippets |
| 2 | Tutorial | 15-30 min | Step-by-step with explanations |
| 3 | Advanced Guide | 30-60 min | End-to-end workflows |

## Creating New Examples

1. Copy the template: `cp examples/_templates/example_template.py examples/<category>/NN_name.py`
2. Follow the 7-part structure (metadata, overview, setup, implementation, results, next steps)
3. Generate the paired notebook: `python scripts/jupytext_converter.py py-to-nb examples/<category>/NN_name.py`
4. Validate: `python scripts/validate_examples.py --path examples/<category>/NN_name.py`
