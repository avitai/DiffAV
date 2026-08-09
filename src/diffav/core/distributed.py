"""JAX mesh sharding utilities for distributed training.

Provides :class:`DistributedConfig` for mesh configuration, and the
helper functions :func:`create_device_mesh`, :func:`shard_batch`, and
:func:`get_data_parallel_sharding` for data-parallel training across
multiple JAX devices.

On single-device environments (CPU, single GPU) all sharding calls
degrade to trivial no-ops — the training code runs unchanged without
any multi-device hardware.

Example::

    from diffav.core.distributed import (
        DistributedConfig, create_device_mesh, shard_batch,
    )

    config = DistributedConfig()
    mesh = create_device_mesh(config)
    sharded = shard_batch(batch, mesh)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec


@dataclass(frozen=True, slots=True, kw_only=True)
class DistributedConfig:
    """Configuration for a JAX device mesh.

    Specifies the shape of the mesh (how devices are arranged across
    the data, model, and pipeline dimensions) and the sharding
    strategy to apply during training.

    Attributes:
        mesh_shape: 3-tuple of positive ints ``(data, model, pipeline)``
            describing the device mesh dimensions.  Product must equal
            ``jax.device_count()``.  Defaults to ``(1, 1, 1)`` for
            single-device operation.
        sharding_strategy: Sharding mode.  Only ``"ddp"`` (data-parallel
            replication) is implemented in this release.  ``"fsdp"`` and
            ``"mp"`` are reserved for future sprints and will raise
            :exc:`NotImplementedError`.
        axis_names: Labels for the three mesh axes, in the same order
            as ``mesh_shape``.  Defaults to
            ``("data", "model", "pipeline")``.
    """

    mesh_shape: tuple[int, int, int] = (1, 1, 1)
    sharding_strategy: Literal["fsdp", "ddp", "mp"] = "ddp"
    axis_names: tuple[str, str, str] = ("data", "model", "pipeline")


def create_device_mesh(config: DistributedConfig) -> Mesh:
    """Create a JAX device mesh from a :class:`DistributedConfig`.

    On single-device environments ``jax.device_count() == 1``, the
    function returns a trivial 1-element ``(1, 1, 1)`` mesh regardless
    of ``config.mesh_shape``, so all distributed code runs unchanged
    without multi-GPU hardware.

    Only the ``"ddp"`` sharding strategy is supported in this release.
    Requesting ``"fsdp"`` or ``"mp"`` raises :exc:`NotImplementedError`.

    Args:
        config: Distributed configuration specifying mesh shape and
            sharding strategy.

    Returns:
        A :class:`jax.sharding.Mesh` ready for use with
        :func:`shard_batch`.

    Raises:
        NotImplementedError: When ``config.sharding_strategy`` is
            ``"fsdp"`` or ``"mp"`` (reserved for future sprints).
    """
    if config.sharding_strategy != "ddp":
        msg = (
            f"sharding_strategy={config.sharding_strategy!r} is reserved for a "
            "future sprint. Only 'ddp' is implemented."
        )
        raise NotImplementedError(msg)

    devices = jax.devices()
    if len(devices) == 1:
        # Trivial single-device mesh — sharding is a no-op
        return Mesh(np.array(devices).reshape((1, 1, 1)), config.axis_names)

    return Mesh(np.array(devices).reshape(config.mesh_shape), config.axis_names)


def get_data_parallel_sharding(mesh: Mesh) -> NamedSharding:
    """Return a :class:`NamedSharding` for data-parallel (DDP) replication.

    The returned sharding shards the leading (batch) axis across the
    ``"data"`` mesh dimension and replicates all remaining axes.

    Args:
        mesh: Device mesh created by :func:`create_device_mesh`.

    Returns:
        A :class:`jax.sharding.NamedSharding` with
        ``PartitionSpec("data")`` that shards on the ``"data"`` axis.
    """
    return NamedSharding(mesh, PartitionSpec("data"))


def shard_batch(batch: dict[str, Any], mesh: Mesh) -> dict[str, Any]:
    """Apply data-parallel sharding to every array in a batch PyTree.

    Each array value in ``batch`` is annotated with the
    data-parallel :class:`NamedSharding` so that JAX XLA can
    shard the leading (batch) dimension across the ``"data"`` mesh
    axis.  On a single-device mesh this is a trivial no-op — values
    are returned unchanged.

    Args:
        batch: Dict mapping string keys to :class:`jax.Array` values.
            All arrays must have at least one dimension (the batch
            axis at index 0).
        mesh: Device mesh from :func:`create_device_mesh`.

    Returns:
        A new dict with the same keys and structure as ``batch``,
        where every array has been annotated with the data-parallel
        sharding.
    """
    sharding = get_data_parallel_sharding(mesh)
    return {key: jax.device_put(value, sharding) for key, value in batch.items()}
