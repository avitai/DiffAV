# Running Simulacrax on Modal

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

**Only for fresh training** (`--task train`, not the ablation) do you also need
the ~104 GB `training/` split. Upload it separately — and if a large-shard upload
stalls, just re-run the `put` (Modal skips files already present, resuming):

```bash
modal volume put simulacrax-wod-data \
  /mnt/ssd2/Data/waymo/motion_v1.2.1/tf_example/training tf_example/training
```

## Launch

```bash
# Train the map-conditioned diffusion model (checkpoints land on the Volume).
modal run deploy/modal_app.py --task train --extra "--steps 20000 --batch-size 8"

# Bigger card:
modal run deploy/modal_app.py --task train --gpu H100 --extra "--batch-size 16"

# Frozen-vs-learnable DPO ablation against the staged checkpoint.
modal run deploy/modal_app.py --task ablation \
  --extra "--num-scenes 12 --num-dpo-steps 20"
```

`--extra` forwards its contents verbatim to the entrypoint's argparse, so every
flag `scripts/train_wod.py` and `benchmarks/steer_wod_dpo_ablation.py` accept is
available (steps, batch size, beta, learning rate, scene counts, …).

## Retrieve results

```bash
# Trained checkpoint / ablation CSV written under the checkpoints Volume.
modal volume get simulacrax-checkpoints wod-physics-tier0 ./checkpoints/
modal volume get simulacrax-checkpoints ablation ./temp/steering/
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
