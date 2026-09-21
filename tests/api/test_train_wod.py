"""The training script refuses a checkpoint root that already holds checkpoints.

A run trains from step 0 and saves every ``--checkpoint-every`` steps, and the checkpoint
store refuses a step that exists or lies below the latest. Into an occupied root the first
save would fail only after that much training, so the script refuses before loading data.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from flax import nnx
from scripts import train_wod

from diffav.models.checkpointing import CheckpointConfig, DiffAVCheckpointManager


class _Tiny(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.linear = nnx.Linear(2, 2, rngs=rngs)


def test_an_occupied_checkpoint_root_is_refused_before_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "run"
    with DiffAVCheckpointManager(CheckpointConfig(checkpoint_dir=str(root))) as manager:
        manager.save(_Tiny(rngs=nnx.Rngs(0)), step=4_000, loss=0.5)
    before = sorted(path.name for path in root.iterdir())

    def _no_data(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("the refusal must come before any data is read")

    monkeypatch.setattr(train_wod, "WODSource", _no_data)

    with pytest.raises(FileExistsError, match=r"already holds checkpoints up to step 4000"):
        train_wod.main(["--checkpoint-dir", str(root)])

    assert sorted(path.name for path in root.iterdir()) == before


@pytest.mark.parametrize(
    ("total_steps", "every", "saved"),
    [(10, 4, [4, 8, 10]), (8, 4, [4, 8]), (3, 4, [3]), (1, 1, [1])],
)
def test_checkpoints_are_labelled_with_the_updates_they_hold(
    total_steps: int, every: int, saved: list[int]
) -> None:
    """Periodic saves fall on multiples of the interval, counted in completed updates, and
    the final save at the total is never a second save of a periodic one."""
    periodic = [
        completed
        for completed in range(1, total_steps + 1)
        if train_wod.is_periodic_checkpoint(completed, every=every, total_steps=total_steps)
    ]

    assert [*periodic, total_steps] == saved
