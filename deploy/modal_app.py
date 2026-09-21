"""Modal launcher for DiffAV GPU training, the steering benchmarks and the examples.

Runs the repository's existing entrypoints (``scripts/train_wod.py``,
``benchmarks/steer_wod_dpo_ablation.py`` and ``benchmarks/steer_wod_guidance.py``)
unchanged on a Modal-managed A100/H100,
fixing the 24 GB local-card OOM. The container is built once from this repo with
``uv sync`` (the sister repos resolve from their public git tags), the Waymo
tfrecords and checkpoints live on persistent Modal Volumes, and each task shells
out through ``uv run`` exactly as the local workflow does.

Prerequisites and one-time data staging are documented in ``deploy/README.md``.

Usage::

    modal run deploy/modal_app.py --task train --checkpoint wod-physics-run2 \
        --extra "--steps 20000 --batch-size 8"
    modal run deploy/modal_app.py --task ablation --gpu A100-80GB
    modal run deploy/modal_app.py --task guidance --extra "--num-scenes 6 --guidance-scales 0 2 5"
    modal run deploy/modal_app.py --task train --gpu H100 --checkpoint wod-physics-run3

    modal run deploy/modal_app.py --task examples --gpu L4 \
        --paths "examples/models/02_physics_informed_training_tutorial.py"
    modal run deploy/modal_app.py --task fetch --run <run name>

``--checkpoint`` names the checkpoint root on the Volume (default ``wod-physics-tier0``,
the staged reference). A training run needs a root of its own: ``train_wod.py`` refuses
one that already holds checkpoints.

``--task examples`` runs example scripts on the GPU, since a docs page quotes a GPU run.
Each script's log and everything it wrote under ``AVITAI_OUTPUT_DIR`` go to the
``diffav-example-outputs`` volume under the run's name, committed after every script, and
are then fetched into ``--out``; ``--task fetch`` downloads a run whose client detached or
lost its connection.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import modal


LOGGER = logging.getLogger("diffav.deploy")


APP_NAME = "diffav"
REPO_PATH = "/root/diffav"
DATA_MOUNT = "/data"
CHECKPOINT_MOUNT = "/checkpoints"
DEFAULT_GPU = "A100-80GB"
CHECKPOINT_NAME = "wod-physics-tier0"
OUTPUT_VOLUME = "diffav-example-outputs"
OUTPUT_MOUNT = "/outputs"
# The examples default to a 24 GB card; the largest, the DPO tutorial, trains one pair per step.
EXAMPLES_GPU = "L4"

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
    "**/memory-bank",
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
# The volume names keep the pre-rename "simulacrax-" prefix on purpose: they are
# the identity of already-populated remote stores. Renaming them would silently
# create new empty volumes (create_if_missing=True) and orphan the staged WOD
# tfrecords and trained checkpoints.
data_volume = modal.Volume.from_name("simulacrax-wod-data", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("simulacrax-checkpoints", create_if_missing=True)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME, create_if_missing=True)

# WOD_MOTION_TFRECORD_PATH wins over the (excluded) .env.data because python-dotenv
# does not override already-set variables. XLA memory fraction mirrors the local recipe.
_RUN_ENV = {
    "WOD_MOTION_TFRECORD_PATH": f"{DATA_MOUNT}/tf_example",
    "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.9",
}


# A published example number must reproduce from run to run, and a short example must not
# take the whole card.
_EXAMPLE_ENV = {
    "WOD_MOTION_TFRECORD_PATH": f"{DATA_MOUNT}/tf_example",
    "XLA_FLAGS": "--xla_gpu_deterministic_ops=true",
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
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
def train(extra_args: list[str], checkpoint: str) -> None:
    """Train the map-conditioned diffusion model into the Volume root ``checkpoint``."""
    _run_entrypoint(
        [
            "scripts/train_wod.py",
            "--checkpoint-dir",
            f"{CHECKPOINT_MOUNT}/{checkpoint}",
            *extra_args,
        ]
    )
    checkpoint_volume.commit()


def _run_benchmark(script: str, output_name: str, extra_args: list[str], checkpoint: str) -> None:
    """Run a steering benchmark against the Volume root ``checkpoint``, writing to the Volume."""
    _run_entrypoint(
        [
            script,
            "--checkpoint-dir",
            f"{CHECKPOINT_MOUNT}/{checkpoint}",
            "--output-dir",
            f"{CHECKPOINT_MOUNT}/{output_name}",
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
def ablation(extra_args: list[str], checkpoint: str) -> None:
    """Run the frozen-vs-learnable DPO ablation against the Volume root ``checkpoint``."""
    _run_benchmark("benchmarks/steer_wod_dpo_ablation.py", "ablation", extra_args, checkpoint)


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    timeout=6 * 60 * 60,
    volumes={DATA_MOUNT: data_volume, CHECKPOINT_MOUNT: checkpoint_volume},
)
def guidance(extra_args: list[str], checkpoint: str) -> None:
    """Run the test-time guidance sweep against the Volume root ``checkpoint``."""
    _run_benchmark("benchmarks/steer_wod_guidance.py", "guidance", extra_args, checkpoint)


@app.function(
    image=image,
    gpu=EXAMPLES_GPU,
    timeout=6 * 60 * 60,
    volumes={DATA_MOUNT: data_volume, OUTPUT_MOUNT: output_volume},
)
def examples(run: str, paths: list[str]) -> str:
    """Run example scripts on the GPU, writing their logs and outputs to the output volume.

    Args:
        run: Name of this run; everything lands under ``<run>/`` in the output volume.
        paths: Example scripts, relative to the repository root, run in order.

    Returns:
        One tab-separated ``<exit code> <script>`` line per script. The volume holds
        ``<run>/logs/<script stem>.log`` per script, ``<run>/logs/summary.txt``, and
        whatever the scripts wrote under ``AVITAI_OUTPUT_DIR`` (``<run>/examples/...``).
    """
    output = Path(OUTPUT_MOUNT) / run
    logs = output / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    lines = []
    for path in paths:
        completed = subprocess.run(
            ["uv", "run", "--no-sync", "python", "-u", path],
            cwd=REPO_PATH,
            env={**os.environ, **_EXAMPLE_ENV, "AVITAI_OUTPUT_DIR": str(output)},
            check=False,
            capture_output=True,
            text=True,
        )
        (logs / f"{Path(path).stem}.log").write_text(completed.stdout + completed.stderr)
        sys.stdout.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        lines.append(f"{completed.returncode}\t{path}")
        (logs / "summary.txt").write_text("\n".join(lines) + "\n")
        output_volume.commit()
    return "\n".join(lines) + "\n"


def fetch(run: str, destination: Path) -> int:
    """Download everything a run wrote to the output volume into ``destination``."""
    count = 0
    for entry in output_volume.listdir(run, recursive=True):
        if entry.type != modal.volume.FileEntryType.FILE:
            continue
        target = destination / entry.path
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            for chunk in output_volume.read_file(entry.path):
                handle.write(chunk)
        count += 1
    return count


def _run_examples(gpu: str, paths: str, run: str, out: str) -> None:
    """Run the named example scripts, or every example when none is named, then fetch."""
    name = run or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    scripts = paths.split() if paths else sorted(str(p) for p in Path("examples").rglob("*.py"))
    LOGGER.info("run %s: %d scripts", name, len(scripts))
    LOGGER.info("%s", examples.with_options(gpu=gpu).remote(name, scripts))
    count = fetch(name, Path(out))
    LOGGER.info("%d files of run %s written under %s", count, name, Path(out) / name)


@app.local_entrypoint()
def main(
    task: str = "train",
    gpu: str = "",
    checkpoint: str = CHECKPOINT_NAME,
    extra: str = "",
    paths: str = "",
    run: str = "",
    out: str = "temp/modal_examples",
) -> None:
    """Dispatch a task to Modal.

    Args:
        task: ``"train"``, ``"ablation"``, ``"guidance"``, ``"examples"`` (run, then fetch)
            or ``"fetch"`` (download an examples run that detached or lost its connection).
        gpu: Modal GPU spec, e.g. ``"L4"``, ``"A100-80GB"`` or ``"H100"``; empty takes
            ``EXAMPLES_GPU`` for the examples and ``DEFAULT_GPU`` otherwise.
        checkpoint: Checkpoint root on the Volume that the task trains into or reads.
            A training run needs an unused one.
        extra: Extra CLI arguments forwarded verbatim to the entrypoint, as one
            space-separated string.
        paths: Example scripts to run, space-separated; every example when empty.
        run: The examples run's name in the output volume; a UTC timestamp when empty.
        out: Local directory that receives ``<run>/``; ``temp/`` is gitignored.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if task == "examples":
        _run_examples(gpu or EXAMPLES_GPU, paths, run, out)
        return
    if task == "fetch":
        if not run:
            raise ValueError("--task fetch needs the --run to download")
        LOGGER.info("%d files of run %s fetched", fetch(run, Path(out)), run)
        return
    extra_args = extra.split() if extra else []
    tasks = {"train": train, "ablation": ablation, "guidance": guidance}
    if task not in tasks:
        msg = f"Unknown task {task!r}; expected one of {sorted([*tasks, 'examples', 'fetch'])}"
        raise ValueError(msg)
    tasks[task].with_options(gpu=gpu or DEFAULT_GPU).remote(extra_args, checkpoint)
