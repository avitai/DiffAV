"""Tests for checkpoint management."""

from __future__ import annotations

import shutil
from collections.abc import Generator
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx
from substrax.checkpoint import OrbaxCheckpointStore
from substrax.checkpoint.metadata import JsonValue

from diffav.models.checkpointing import (
    CheckpointConfig,
    CheckpointCorruptError,
    DiffAVCheckpointManager,
    TrainingState,
)


class _SimpleModel(nnx.Module):
    """Minimal model for checkpoint round-trip tests."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.linear = nnx.Linear(4, 4, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.linear(x)


def _make_model(seed: int) -> _SimpleModel:
    """Create a _SimpleModel initialized from the given seed."""
    return _SimpleModel(rngs=nnx.Rngs(params=jax.random.key(seed)))


def _make_optimizer(model: _SimpleModel) -> nnx.Optimizer:
    """Create an SGD-with-momentum optimizer (stateful) for the model."""
    return nnx.Optimizer(model, optax.sgd(1e-2, momentum=0.9), wrt=nnx.Param)


def _train_one_step(model: _SimpleModel, optimizer: nnx.Optimizer) -> None:
    """Apply one gradient step so both weights and optimizer state are nonzero."""

    def loss_fn(m: _SimpleModel) -> jax.Array:
        return jnp.sum(m(jnp.ones((2, 4))) ** 2)

    grads = nnx.grad(loss_fn)(model)
    optimizer.update(model, grads)


def _params_equal(a: nnx.Module, b: nnx.Module) -> bool:
    """Return True when all parameter arrays of two modules are identical."""
    leaves_a = jax.tree.leaves(nnx.state(a, nnx.Param))
    leaves_b = jax.tree.leaves(nnx.state(b, nnx.Param))
    return all(bool(jnp.array_equal(x, y)) for x, y in zip(leaves_a, leaves_b, strict=True))


def _states_equal(a: object, b: object) -> bool:
    """Return True when all array leaves of two NNX states are identical."""
    leaves_a = jax.tree.leaves(a)
    leaves_b = jax.tree.leaves(b)
    return all(bool(jnp.array_equal(x, y)) for x, y in zip(leaves_a, leaves_b, strict=True))


# ---------------------------------------------------------------------------
# CheckpointConfig
# ---------------------------------------------------------------------------


class TestCheckpointConfig:
    """Tests for CheckpointConfig validation and defaults."""

    def test_defaults(self, tmp_path: Path) -> None:
        """The cadence and retention have defaults; the directory is the caller's."""
        cfg = CheckpointConfig(checkpoint_dir=str(tmp_path))
        assert cfg.checkpoint_dir == str(tmp_path)
        assert cfg.save_interval_steps == 1000
        assert cfg.max_to_keep == 5

    def test_the_directory_is_required(self) -> None:
        """No default under the working directory, where two runs would collide."""
        with pytest.raises(TypeError, match="checkpoint_dir"):
            CheckpointConfig()  # type: ignore[call-arg]

    def test_invalid_save_interval(self, tmp_path: Path) -> None:
        """save_interval_steps=0 raises ValueError."""
        with pytest.raises(ValueError, match="save_interval_steps"):
            CheckpointConfig(checkpoint_dir=str(tmp_path), save_interval_steps=0)

    def test_invalid_max_to_keep(self, tmp_path: Path) -> None:
        """max_to_keep=0 raises ValueError."""
        with pytest.raises(ValueError, match="max_to_keep"):
            CheckpointConfig(checkpoint_dir=str(tmp_path), max_to_keep=0)

    def test_custom_values(self, tmp_path: Path) -> None:
        """Custom values are stored correctly."""
        cfg = CheckpointConfig(
            checkpoint_dir=str(tmp_path / "my_ckpts"),
            save_interval_steps=500,
            max_to_keep=3,
        )
        assert cfg.checkpoint_dir == str(tmp_path / "my_ckpts")
        assert cfg.save_interval_steps == 500
        assert cfg.max_to_keep == 3


# ---------------------------------------------------------------------------
# TrainingState
# ---------------------------------------------------------------------------


class TestTrainingState:
    """Tests for TrainingState frozen dataclass."""

    def test_creation(self) -> None:
        """TrainingState can be created with required fields."""
        state = TrainingState(
            step=10,
            epoch=1,
            model_state={"params": jnp.zeros(5)},
            optimizer_state={"momentum": jnp.ones(5)},
            best_loss=0.5,
            metrics={"ade": 1.2, "fde": 2.3},
        )
        assert state.step == 10
        assert state.epoch == 1
        assert state.best_loss == 0.5
        assert state.metrics["ade"] == 1.2

    def test_frozen(self) -> None:
        """TrainingState fields cannot be reassigned."""
        state = TrainingState(
            step=0,
            epoch=0,
            model_state=None,
            optimizer_state=None,
            best_loss=float("inf"),
            metrics={},
        )
        with pytest.raises(AttributeError):
            state.step = 1  # type: ignore[misc]


# ---------------------------------------------------------------------------
# DiffAVCheckpointManager
# ---------------------------------------------------------------------------


class TestDiffAVCheckpointManager:
    """Tests for DiffAVCheckpointManager with real Orbax I/O."""

    @pytest.fixture()
    def model(self) -> _SimpleModel:
        """Create a fresh _SimpleModel for each test."""
        return _make_model(42)

    @pytest.fixture()
    def config(self, tmp_path: Path) -> CheckpointConfig:
        """Create a CheckpointConfig pointing at a temp directory."""
        return CheckpointConfig(
            checkpoint_dir=str(tmp_path / "ckpts"),
            save_interval_steps=5,
            max_to_keep=5,
        )

    @pytest.fixture()
    def manager(self, config: CheckpointConfig) -> Generator[DiffAVCheckpointManager]:
        """Create a DiffAVCheckpointManager from the config fixture."""
        with DiffAVCheckpointManager(config) as mgr:
            yield mgr

    def test_save_writes_the_record_substrax_reads(
        self,
        manager: DiffAVCheckpointManager,
        config: CheckpointConfig,
        model: _SimpleModel,
    ) -> None:
        """A save is a format-3 step: the model item, the loss and metrics, the epoch, the
        producer and the architecture version, readable without DiffAV."""
        optimizer = _make_optimizer(model)
        path = manager.save(
            model,
            step=10,
            loss=0.5,
            metrics={"ade": 1.5},
            optimizer=optimizer,
            epoch=2,
            architecture_version=3,
        )

        assert isinstance(path, Path)
        assert path.is_dir()
        with OrbaxCheckpointStore(config.checkpoint_dir) as store:
            record = store.read_metadata(10)
        assert record.items == ("model", "optimizer")
        assert record.epoch == 2
        assert record.metrics == {"ade": 1.5, "loss": 0.5}
        assert record.extra == {"architecture_version": 3}
        assert record.producer is not None
        assert record.producer.name == "diffav"

    def test_a_save_without_optimizer_or_version_writes_the_model_alone(
        self,
        manager: DiffAVCheckpointManager,
        config: CheckpointConfig,
        model: _SimpleModel,
    ) -> None:
        manager.save(model, step=4, loss=0.25)

        with OrbaxCheckpointStore(config.checkpoint_dir) as store:
            record = store.read_metadata(4)
        assert record.items == ("model",)
        assert record.epoch == 0
        assert "architecture_version" not in record.extra

    def test_save_and_restore_round_trip(
        self,
        manager: DiffAVCheckpointManager,
        model: _SimpleModel,
    ) -> None:
        """Restored weights are bit-identical to the saved ones."""
        manager.save(model, step=10, loss=0.5, metrics={"ade": 1.5})

        fresh_model = _make_model(99)
        assert not _params_equal(model, fresh_model)

        state = manager.restore_latest(fresh_model)

        assert isinstance(state, TrainingState)
        assert state.step == 10
        assert state.best_loss == 0.5
        assert state.metrics["ade"] == 1.5
        assert _params_equal(model, fresh_model)

    def test_restore_with_optimizer_round_trip(
        self,
        manager: DiffAVCheckpointManager,
        model: _SimpleModel,
    ) -> None:
        """Optimizer state, step, and epoch survive the round trip."""
        optimizer = _make_optimizer(model)
        _train_one_step(model, optimizer)

        manager.save(model, step=7, loss=0.25, optimizer=optimizer, epoch=3)

        fresh_model = _make_model(99)
        fresh_optimizer = _make_optimizer(fresh_model)
        assert not _states_equal(nnx.state(optimizer), nnx.state(fresh_optimizer))

        state = manager.restore_latest(fresh_model, optimizer=fresh_optimizer)

        assert isinstance(state, TrainingState)
        assert state.step == 7
        assert state.epoch == 3
        assert _params_equal(model, fresh_model)
        assert _states_equal(nnx.state(optimizer), nnx.state(fresh_optimizer))

    def test_restore_latest_none_when_empty(
        self,
        manager: DiffAVCheckpointManager,
        model: _SimpleModel,
    ) -> None:
        """restore_latest returns None (model untouched) with no checkpoints."""
        before = nnx.state(model, nnx.Param)
        state = manager.restore_latest(model)
        assert state is None
        assert _states_equal(before, nnx.state(model, nnx.Param))

    def test_architecture_version_round_trip(
        self,
        manager: DiffAVCheckpointManager,
        model: _SimpleModel,
    ) -> None:
        """A matching architecture version restores and is carried through."""
        manager.save(model, step=10, loss=0.5, architecture_version=1)

        fresh_model = _make_model(99)
        state = manager.restore_latest(fresh_model, expected_architecture_version=1)

        assert isinstance(state, TrainingState)
        assert state.architecture_version == 1
        assert _params_equal(model, fresh_model)

    def test_restore_rejects_version_mismatch_before_loading(
        self,
        manager: DiffAVCheckpointManager,
        model: _SimpleModel,
    ) -> None:
        """A version mismatch raises and leaves the target model untouched."""
        manager.save(model, step=10, loss=0.5, architecture_version=1)

        fresh_model = _make_model(99)
        before = nnx.state(fresh_model, nnx.Param)
        with pytest.raises(CheckpointCorruptError, match="architecture version"):
            manager.restore_latest(fresh_model, expected_architecture_version=2)
        # The gate fires before nnx.update, so no weights leaked in.
        assert _states_equal(before, nnx.state(fresh_model, nnx.Param))

    def test_restore_rejects_unversioned_when_version_expected(
        self,
        manager: DiffAVCheckpointManager,
        model: _SimpleModel,
    ) -> None:
        """An unversioned checkpoint is rejected when a version is required."""
        manager.save(model, step=10, loss=0.5)  # architecture_version defaults to None

        fresh_model = _make_model(99)
        with pytest.raises(CheckpointCorruptError, match="architecture version"):
            manager.restore_latest(fresh_model, expected_architecture_version=1)

    @pytest.mark.parametrize("stored", ["1", 1.5, True, [1]])
    def test_restore_rejects_a_version_that_is_not_an_integer(
        self,
        config: CheckpointConfig,
        model: _SimpleModel,
        stored: JsonValue,
    ) -> None:
        """A record whose architecture version is not an integer is corrupt, and nothing
        loads, whether or not a version is expected."""
        with OrbaxCheckpointStore(config.checkpoint_dir) as store:
            store.save(
                10,
                {"model": nnx.state(model)},
                metrics={"loss": 0.5},
                extra={"architecture_version": stored},
            )

        fresh_model = _make_model(99)
        before = nnx.state(fresh_model, nnx.Param)
        with (
            DiffAVCheckpointManager(config) as manager,
            pytest.raises(CheckpointCorruptError, match="architecture version"),
        ):
            manager.restore_latest(fresh_model)
        assert _states_equal(before, nnx.state(fresh_model, nnx.Param))

    def test_restore_skips_version_check_by_default(
        self,
        manager: DiffAVCheckpointManager,
        model: _SimpleModel,
    ) -> None:
        """With no expected version, a versioned checkpoint restores unchecked."""
        manager.save(model, step=10, loss=0.5, architecture_version=1)

        fresh_model = _make_model(99)
        state = manager.restore_latest(fresh_model)

        assert isinstance(state, TrainingState)
        assert state.architecture_version == 1
        assert _params_equal(model, fresh_model)

    def test_restore_latest_raises_on_corrupt(
        self,
        config: CheckpointConfig,
        model: _SimpleModel,
    ) -> None:
        """A checkpoint that exists but cannot be read raises, not silently skips."""
        with DiffAVCheckpointManager(config) as manager:
            manager.save(model, step=5, loss=0.1)

        # Destroy the checkpoint payload while keeping the step directory listed.
        step_dir = Path(config.checkpoint_dir) / "5"
        for child in step_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()

        with (
            DiffAVCheckpointManager(config) as manager,
            pytest.raises(CheckpointCorruptError, match="step 5"),
        ):
            manager.restore_latest(model)

    def test_restore_latest_raises_on_mismatched_payload(
        self,
        config: CheckpointConfig,
        model: _SimpleModel,
    ) -> None:
        """A checkpoint whose arrays do not fit the model raises, naming the step."""
        with DiffAVCheckpointManager(config) as manager:
            manager.save(model, step=5, loss=0.1)

        class _WiderModel(nnx.Module):
            def __init__(self, *, rngs: nnx.Rngs) -> None:
                self.linear = nnx.Linear(8, 8, rngs=rngs)

        with (
            DiffAVCheckpointManager(config) as manager,
            pytest.raises(CheckpointCorruptError, match="step 5"),
        ):
            manager.restore_latest(_WiderModel(rngs=nnx.Rngs(0)))

    def test_context_manager_closes(self, config: CheckpointConfig, model: _SimpleModel) -> None:
        """The manager works as a context manager and persists on exit."""
        with DiffAVCheckpointManager(config) as manager:
            manager.save(model, step=1, loss=0.9)
        assert (Path(config.checkpoint_dir) / "1").exists()

    def test_save_if_due_at_interval(
        self,
        manager: DiffAVCheckpointManager,
        model: _SimpleModel,
    ) -> None:
        """When step is a multiple of save_interval_steps, save occurs."""
        result = manager.save_if_due(model, step=5, loss=0.3)
        assert result is not None

    def test_save_if_due_not_at_interval(
        self,
        manager: DiffAVCheckpointManager,
        model: _SimpleModel,
    ) -> None:
        """When step is not a multiple of save_interval_steps, no save occurs."""
        result = manager.save_if_due(model, step=3, loss=0.3)
        assert result is None

    def test_save_if_due_at_zero(
        self,
        manager: DiffAVCheckpointManager,
        model: _SimpleModel,
    ) -> None:
        """Step 0 never triggers a save even though 0 % N == 0."""
        result = manager.save_if_due(model, step=0, loss=0.3)
        assert result is None

    def test_latest_step_after_saves(
        self,
        manager: DiffAVCheckpointManager,
        model: _SimpleModel,
    ) -> None:
        """After saving at steps 1, 2, 3, latest_step returns 3."""
        for step in (1, 2, 3):
            manager.save(model, step=step, loss=0.1 * step)

        assert manager.latest_step() == 3

    def test_latest_step_empty(
        self,
        manager: DiffAVCheckpointManager,
    ) -> None:
        """With no checkpoints, latest_step returns None."""
        assert manager.latest_step() is None

    def test_max_to_keep_prunes(
        self,
        tmp_path: Path,
        model: _SimpleModel,
    ) -> None:
        """Only the most recent max_to_keep checkpoints are retained."""
        config = CheckpointConfig(
            checkpoint_dir=str(tmp_path / "pruned_ckpts"),
            save_interval_steps=1,
            max_to_keep=2,
        )
        with DiffAVCheckpointManager(config) as mgr:
            for step in (1, 2, 3):
                mgr.save(model, step=step, loss=0.1 * step)

            kept_steps = mgr.list_steps()
            assert kept_steps == [2, 3]
