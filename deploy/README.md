# Running DiffAV on Modal

Launch GPU training and the DPO ablation on a Modal-managed A100/H100 — the fix
for the 24 GB local-card OOM. The container runs the repository's existing
entrypoints unchanged; only the data path and checkpoint directory move to
persistent Modal Volumes.

## Why Modal

Python-native launch (no Dockerfile, no SSH), per-second billing with
scale-to-zero (an OOM-debug loop costs cents), and a monthly free credit. An
A100 80 GB (~$2.50/hr) or H100 (~$3.95/hr) directly clears the OOM. For the
cheapest sustained hours, the same container ports to RunPod (network volumes)
or vast.ai (interruptible, once runs are checkpoint-safe — which they are).

## One-time setup

```bash
uv pip install modal        # host-side launcher only; not a training dependency
modal setup                 # browser auth
```

The sister repositories (`avitai-artifex`, `opifex`, `datarax`, `calibrax`)
resolve from their public git tags during the image build, so no GitHub secret
is required.

## Stage the data and checkpoint (once)

A run-time function mount creates its Volume lazily, but the `modal volume put`
CLI needs the Volume to exist first — so create the two Volumes explicitly, then
upload the Waymo tfrecords and the reference checkpoint into them:

The Volume names keep the pre-rename `simulacrax-` prefix deliberately — they
identify already-populated remote stores, and renaming them would create new
empty Volumes rather than move the data.

```bash
modal volume create simulacrax-wod-data
modal volume create simulacrax-checkpoints

# WOD tfrecords — the DPO ablation reads ONLY the validation split (~8 GB, 16
# files). Upload just that; the container reads $DATA_MOUNT/tf_example/validation.
modal volume put simulacrax-wod-data \
  /mnt/ssd2/Data/waymo/motion_v1.2.1/tf_example/validation tf_example/validation

# Reference checkpoint (needed only for the ablation, not for fresh training).
modal volume put simulacrax-checkpoints \
  ./checkpoints/wod-physics-tier0 wod-physics-tier0
```

The volume persists across runs, so staging is a one-time cost.

**Only for fresh training** (`--task train`, not the benchmarks) do you also need
the `training/` split. `train_wod.py` streams the shards in name order and stops
at `--max-scenarios`, so a run needs only the leading shards that hold that many
records: the recorded `wod-physics-tier0` run (`--max-scenarios 8000`) reads
shards `00000` to `00016` (8,126 records, about 14.5 GB), not the full ~104 GB
split. Upload them one at a time — and if a large-shard upload stalls, just re-run
the `put` (`--force` replaces a partial file):

```bash
for i in $(seq -f "%05g" 0 16); do
  f="training_tfexample.tfrecord-${i}-of-01000"
  modal volume put simulacrax-wod-data \
    "/mnt/ssd2/Data/waymo/motion_v1.2.1/tf_example/training/$f" "tf_example/training/$f"
done
```

## Launch

```bash
# Train the map-conditioned diffusion model into a new root on the Volume.
modal run deploy/modal_app.py --task train --checkpoint wod-physics-run2 \
  --extra "--steps 20000 --batch-size 8"

# Bigger card:
modal run deploy/modal_app.py --task train --gpu H100 --checkpoint wod-physics-run3 \
  --extra "--batch-size 16"

# Frozen-vs-learnable DPO ablation against the staged checkpoint.
modal run deploy/modal_app.py --task ablation \
  --extra "--num-scenes 12 --num-dpo-steps 20"

# Test-time guidance sweep against the staged checkpoint (the results/ table).
modal run deploy/modal_app.py --task guidance \
  --extra "--num-scenes 6 --num-rollouts 6 --guidance-scales 0 2 5"

# Run examples on the GPU (every example when --paths is empty) and fetch their
# logs and figures into temp/modal_examples/<run>/.
modal run deploy/modal_app.py --task examples --gpu L4 \
  --paths "examples/models/02_physics_informed_training_tutorial.py"

# Download an examples run whose client detached or lost its connection.
modal run deploy/modal_app.py --task fetch --run <run name>
```

The examples default to an L4 (24 GB), the smallest card the WOD-scale ones are sized
for; the DPO tutorial, the largest, trains one pair per step. They read WOD from the data Volume, run with deterministic GPU kernels so a
quoted number reproduces, and write each script's log plus everything under
`AVITAI_OUTPUT_DIR` to the `diffav-example-outputs` volume, committed after every script.
The image excludes `checkpoints/`, so an example that restores `checkpoints/wod-mini`
runs its untrained fallback there.

`--checkpoint` names the checkpoint root on the Volume and defaults to the staged
`wod-physics-tier0`. The benchmarks read it; a training run writes it and needs an
unused one, since `train_wod.py` refuses a root that already holds checkpoints rather
than fail at its first save. Pass `--checkpoint <root>` to the benchmarks to evaluate a
new run.

`--extra` forwards its contents verbatim to the entrypoint's argparse, so every
flag `scripts/train_wod.py`, `benchmarks/steer_wod_dpo_ablation.py` and
`benchmarks/steer_wod_guidance.py` accept is available (steps, batch size, beta,
learning rate, scene counts, guidance scales, …).

## Retrieve results

```bash
# A trained checkpoint root (the --checkpoint a run wrote) and the benchmark CSVs.
modal volume get simulacrax-checkpoints wod-physics-run2 ./checkpoints/
modal volume get simulacrax-checkpoints ablation ./temp/steering/
modal volume get simulacrax-checkpoints guidance ./temp/steering/
```

## Notes

- **Data path**: the container sets `WOD_MOTION_TFRECORD_PATH=/data/tf_example`.
  The build deliberately excludes every local `.env*` file, because
  `train_wod.py` calls `load_dotenv(".env.data")` and the local file would
  otherwise point the container back at the `/mnt/ssd2` path.
- **CUDA**: JAX's `[cuda12]` wheels bundle their own CUDA/cuDNN, so the plain
  `debian_slim` image plus Modal's driver is enough — no CUDA base image needed.
  `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` is set to match the local recipe.
- **Iterating on the image**: edit `deploy/modal_app.py`; Modal rebuilds only the
  changed layers. The `uv sync --locked` step re-runs when `pyproject.toml` or
  `uv.lock` change.
- **Cost control**: per-second billing means killing a run (`Ctrl-C`, or the
  Modal dashboard) stops the meter immediately. Prefer a short `--steps` smoke
  run before a long one.
