"""Checkpoint management for trajectory model training.

Composes opifex's :class:`OrbaxCheckpointStore` with diffav-specific
training state: model parameters, optimizer state, step, epoch, and metrics
are persisted together so training can resume exactly where it stopped.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast, Self

from flax import nnx
from opifex.core.training.components.checkpoint_store import OrbaxCheckpointStore

from diffav.core.config import validate_positive


logger = logging.getLogger(__name__)


class CheckpointCorruptError(RuntimeError):
    """Raised when a checkpoint exists on disk but cannot be restored.

    Distinguishes unreadable/mismatched checkpoints from the benign
    no-checkpoint-yet case (which restore reports as ``None``).
    """


@dataclass(frozen=True, slots=True, kw_only=True)
class CheckpointConfig:
    """Configuration for checkpoint management.

    Attributes:
        checkpoint_dir: Directory path for storing checkpoints.
        save_interval_steps: Save a checkpoint every N training steps.
        max_to_keep: Maximum number of recent checkpoints to retain.
    """

    checkpoint_dir: str = "checkpoints"
    save_interval_steps: int = 1000
    max_to_keep: int = 5

    def __post_init__(self) -> None:
        """Validate field constraints."""
        validate_positive("save_interval_steps", self.save_interval_steps)
        validate_positive("max_to_keep", self.max_to_keep)


@dataclass(frozen=True, slots=True, kw_only=True)
class TrainingState:
    """Immutable snapshot of training state restored from a checkpoint.

    Attributes:
        step: Training step the checkpoint was saved at.
        epoch: Training epoch the checkpoint was saved at.
        model_state: PyTree of restored model parameters.
        optimizer_state: PyTree of restored optimizer state, or ``None``
            when the checkpoint was saved without an optimizer.
        best_loss: Loss value recorded at save time.
        metrics: Dictionary of metric name to value recorded at save time.
        architecture_version: Backbone architecture revision the checkpoint
            was saved under, or ``None`` for checkpoints written before
            versioning existed.
    """

    step: int
    epoch: int
    model_state: Any
    optimizer_state: Any
    best_loss: float
    metrics: Mapping[str, float]
    architecture_version: int | None = None


class DiffAVCheckpointManager:
    """Checkpoint manager for diffav trajectory model training.

    Wraps opifex's :class:`OrbaxCheckpointStore` with periodic-save gating
    and full training-state persistence (model + optimizer + step/epoch).
    Doubles as a context manager so backend resources are released
    deterministically::

        with DiffAVCheckpointManager(config) as manager:
            manager.save(model, step, loss, optimizer=optimizer, epoch=epoch)
    """

    def __init__(self, config: CheckpointConfig) -> None:
        """Initialize the checkpoint manager.

        Creates the checkpoint directory if it does not exist and
        instantiates the underlying OrbaxCheckpointStore.

        Args:
            config: Checkpoint configuration.
        """
        self.config = config
        self._store = OrbaxCheckpointStore(
            config.checkpoint_dir,
            max_to_keep=config.max_to_keep,
        )

    def __enter__(self) -> Self:
        """Enter the context manager."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close the underlying store on context exit."""
        self.close()

    @staticmethod
    def _build_payload(model: nnx.Module, optimizer: nnx.Optimizer | None) -> dict[str, Any]:
        """Build the on-disk payload structure for save and restore."""
        payload: dict[str, Any] = {"model": nnx.state(model)}
        if optimizer is not None:
            payload["optimizer"] = nnx.state(optimizer)
        return payload

    def save(
        self,
        model: nnx.Module,
        step: int,
        loss: float,
        metrics: dict[str, float] | None = None,
        *,
        optimizer: nnx.Optimizer | None = None,
        epoch: int = 0,
        architecture_version: int | None = None,
    ) -> str:
        """Save a checkpoint at the given step.

        Args:
            model: Flax NNX model to checkpoint.
            step: Current training step number.
            loss: Current loss value.
            metrics: Optional dictionary of additional metrics.
            optimizer: Optional NNX optimizer whose state is persisted
                alongside the model.
            epoch: Current training epoch, recorded in metadata.
            architecture_version: Backbone architecture revision to stamp
                into the checkpoint metadata. Pass the running code's
                version so a later restore can reject incompatible weights;
                ``None`` writes an unversioned checkpoint.

        Returns:
            Path string to the saved checkpoint.
        """
        payload = self._build_payload(model, optimizer)
        checkpoint_path = self._store.save(
            payload,
            step,
            loss,
            additional_metadata={
                "epoch": epoch,
                "metrics": dict(metrics or {}),
                "architecture_version": architecture_version,
            },
        )
        logger.info("Saved checkpoint at step %d: %s", step, checkpoint_path)
        return checkpoint_path

    def save_if_due(
        self,
        model: nnx.Module,
        step: int,
        loss: float,
        metrics: dict[str, float] | None = None,
        *,
        optimizer: nnx.Optimizer | None = None,
        epoch: int = 0,
        architecture_version: int | None = None,
    ) -> str | None:
        """Save a checkpoint only if the step aligns with the save interval.

        A checkpoint is saved when ``step > 0`` and ``step`` is a multiple
        of ``config.save_interval_steps``.

        Args:
            model: Flax NNX model to checkpoint.
            step: Current training step number.
            loss: Current loss value.
            metrics: Optional dictionary of additional metrics.
            optimizer: Optional NNX optimizer whose state is persisted
                alongside the model.
            epoch: Current training epoch, recorded in metadata.
            architecture_version: Backbone architecture revision stamped
                into the checkpoint metadata (see :meth:`save`).

        Returns:
            Path string if a checkpoint was saved, None otherwise.
        """
        if step > 0 and step % self.config.save_interval_steps == 0:
            return self.save(
                model,
                step,
                loss,
                metrics,
                optimizer=optimizer,
                epoch=epoch,
                architecture_version=architecture_version,
            )
        return None

    def restore_latest(
        self,
        model: nnx.Module,
        *,
        optimizer: nnx.Optimizer | None = None,
        expected_architecture_version: int | None = None,
    ) -> TrainingState | None:
        """Restore model (and optionally optimizer) from the latest checkpoint.

        The model and optimizer are updated **in place**; the returned
        :class:`TrainingState` carries the bookkeeping needed to resume
        training (step, epoch, loss, metrics).

        Args:
            model: Flax NNX model to restore into.
            optimizer: Optional NNX optimizer to restore into. Must be
                provided when the checkpoint was saved with an optimizer
                and vice versa — the payload structures have to match.
            expected_architecture_version: When set, the checkpoint's stored
                architecture version must equal it or the restore is rejected
                **before** any weights are loaded. Guards against silently
                loading stale weights into an incompatible backbone that
                happens to share leaf names. ``None`` skips the check.

        Returns:
            The restored :class:`TrainingState`, or ``None`` when no
            checkpoint exists yet.

        Raises:
            CheckpointCorruptError: If the latest checkpoint exists on disk
                but cannot be restored (unreadable data, a payload whose
                structure does not match ``model``/``optimizer``, or an
                architecture version that does not match
                ``expected_architecture_version``).
        """
        latest = self._store.latest_step()
        if latest is None:
            return None

        abstract = self._build_payload(model, optimizer)
        restored, metadata = self._store.restore(
            abstract,
            step=latest,
            return_original_on_missing=False,
        )
        if restored is None or not metadata:
            raise CheckpointCorruptError(
                f"Checkpoint at step {latest} in {self.config.checkpoint_dir!r} exists "
                "but could not be restored (corrupt data or mismatched payload structure)"
            )

        stored_version = metadata.get("architecture_version")
        if (
            expected_architecture_version is not None
            and stored_version != expected_architecture_version
        ):
            raise CheckpointCorruptError(
                f"Checkpoint at step {latest} in {self.config.checkpoint_dir!r} was saved "
                f"under backbone architecture version {stored_version!r}, but the running "
                f"code expects {expected_architecture_version!r}. Retrain the checkpoint "
                "for the current architecture."
            )

        restored_payload = cast(dict[str, Any], restored)
        nnx.update(model, restored_payload["model"])
        optimizer_state = None
        if optimizer is not None and "optimizer" in restored_payload:
            nnx.update(optimizer, restored_payload["optimizer"])
            optimizer_state = restored_payload["optimizer"]

        return TrainingState(
            step=int(metadata.get("step", latest)),
            epoch=int(metadata.get("epoch", 0)),
            model_state=restored_payload["model"],
            optimizer_state=optimizer_state,
            best_loss=float(metadata.get("loss", float("inf"))),
            metrics=metadata.get("metrics", {}),
            architecture_version=stored_version,
        )

    def latest_step(self) -> int | None:
        """Return the step number of the latest checkpoint.

        Returns:
            Latest checkpoint step, or None if no checkpoints exist.
        """
        return self._store.latest_step()

    def list_steps(self) -> list[int]:
        """Return all retained checkpoint steps in ascending order."""
        return sorted(self._store.list_steps())

    def close(self) -> None:
        """Close the underlying checkpoint store and release resources."""
        self._store.close()
