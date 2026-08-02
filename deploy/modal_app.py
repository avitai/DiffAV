"""Modal launcher for Simulacrax GPU training and the DPO ablation.

Runs the repository's existing entrypoints (``scripts/train_wod.py`` and
``benchmarks/steer_wod_dpo_ablation.py``) unchanged on a Modal-managed A100/H100,
fixing the 24 GB local-card OOM. The container is built once from this repo with
``uv sync`` (the sister repos resolve from their public git tags), the Waymo
tfrecords and checkpoints live on persistent Modal Volumes, and each task shells
out through ``uv run`` exactly as the local workflow does.

Prerequisites and one-time data staging are documented in ``deploy/README.md``.

Usage::

    modal run deploy/modal_app.py --task train --extra "--steps 20000 --batch-size 8"
    modal run deploy/modal_app.py --task ablation --gpu A100-80GB
    modal run deploy/modal_app.py --task train --gpu H100
"""

from __future__ import annotations

import os
import subprocess

import modal


APP_NAME = "simulacrax"
REPO_PATH = "/root/simulacrax"
DATA_MOUNT = "/data"
CHECKPOINT_MOUNT = "/checkpoints"
DEFAULT_GPU = "A100-80GB"
CHECKPOINT_NAME = "wod-physics-tier0"

# Excludes for the build-time repo copy: local venv/caches, heavy artifacts, and
# — critically — every ``.env*`` file. ``scripts/train_wod.py`` calls
# ``load_dotenv(".env.data")``; shipping the local file would point the container
# at the on-disk ``/mnt/ssd2`` data path instead of the mounted Volume.
_IMAGE_IGNORE = [
    "**/.git",
    "**/.venv",
    "**/.env*",
    "**/checkpoints",
    "**/temp",
    "**/__pycache__",
    "**/*.pyc",
    "**/.pytest_cache",
    "**/.ruff_cache",
    "**/site",
]

app = modal.App(APP_NAME)

# Pin uv to the version that generated uv.lock, for build stability.
_UV_VERSION = "0.11.25"

# `--frozen` installs the exact pinned versions from uv.lock WITHOUT re-resolving:
# it gives full lock reproducibility while skipping the `--locked` consistency
# re-check, which spuriously fails on Modal because its build environment (uv's
# managed Python 3.12.13 vs local 3.12.6, plus the calibrax override) re-resolves
# the git-sourced graph slightly differently even though the lock is correct.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(f"uv=={_UV_VERSION}")
    .add_local_dir(".", REPO_PATH, copy=True, ignore=_IMAGE_IGNORE)
    .run_commands(f"cd {REPO_PATH} && uv sync --extra gpu --frozen --no-dev")
    .workdir(REPO_PATH)
)

# Persistent stores: stage the tfrecords and the reference checkpoint once (see
# deploy/README.md), then every run reuses them.
data_volume = modal.Volume.from_name("simulacrax-wod-data", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("simulacrax-checkpoints", create_if_missing=True)

# WOD_MOTION_TFRECORD_PATH wins over the (excluded) .env.data because python-dotenv
# does not override already-set variables. XLA memory fraction mirrors the local recipe.
_RUN_ENV = {
    "WOD_MOTION_TFRECORD_PATH": f"{DATA_MOUNT}/tf_example",
    "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.9",
}


def _run_entrypoint(argv: list[str]) -> None:
    """Run a repo entrypoint through ``uv run`` with the staged environment."""
    os.environ.update(_RUN_ENV)
    subprocess.run(
        ["uv", "run", "--no-sync", "python", "-u", *argv],
        cwd=REPO_PATH,
        check=True,
    )


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    timeout=24 * 60 * 60,
    volumes={DATA_MOUNT: data_volume, CHECKPOINT_MOUNT: checkpoint_volume},
)
def train(extra_args: list[str]) -> None:
    """Train the map-conditioned diffusion model, checkpointing to the Volume."""
    _run_entrypoint(
        [
            "scripts/train_wod.py",
            "--checkpoint-dir",
            f"{CHECKPOINT_MOUNT}/{CHECKPOINT_NAME}",
            *extra_args,
        ]
    )
    checkpoint_volume.commit()


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    timeout=6 * 60 * 60,
    volumes={DATA_MOUNT: data_volume, CHECKPOINT_MOUNT: checkpoint_volume},
)
def ablation(extra_args: list[str]) -> None:
    """Run the frozen-vs-learnable DPO ablation against the staged checkpoint."""
    _run_entrypoint(
        [
            "benchmarks/steer_wod_dpo_ablation.py",
            "--checkpoint-dir",
            f"{CHECKPOINT_MOUNT}/{CHECKPOINT_NAME}",
            "--output-dir",
            f"{CHECKPOINT_MOUNT}/ablation",
            *extra_args,
        ]
    )
    checkpoint_volume.commit()


@app.local_entrypoint()
def main(task: str = "train", gpu: str = DEFAULT_GPU, extra: str = "") -> None:
    """Dispatch a task to Modal.

    Args:
        task: ``"train"`` or ``"ablation"``.
        gpu: Modal GPU spec, e.g. ``"A100-80GB"`` or ``"H100"``.
        extra: Extra CLI arguments forwarded verbatim to the entrypoint, as one
            space-separated string.
    """
    extra_args = extra.split() if extra else []
    tasks = {"train": train, "ablation": ablation}
    if task not in tasks:
        msg = f"Unknown task {task!r}; expected one of {sorted(tasks)}"
        raise ValueError(msg)
    tasks[task].with_options(gpu=gpu).remote(extra_args)
