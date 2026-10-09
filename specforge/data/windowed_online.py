"""Deterministic parent prompts for windowed online DSpark training."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np


class Int32RingReader:
    """Read fixed-size samples from a little-endian int32 token ring."""

    def __init__(self, path: str | os.PathLike[str], sample_length: int) -> None:
        self.path = os.fspath(path)
        self.sample_length = int(sample_length)
        if self.sample_length < 1:
            raise ValueError("sample_length must be positive")
        size = os.path.getsize(self.path)
        itemsize = np.dtype("<i4").itemsize
        if size == 0:
            raise ValueError(f"token bin {self.path!r} is empty")
        if size % itemsize:
            raise ValueError(
                f"token bin {self.path!r} size {size} is not divisible by {itemsize}"
            )
        self.total_tokens = size // itemsize
        self.parents_per_epoch = (
            self.total_tokens + self.sample_length - 1
        ) // self.sample_length
        self._tokens = np.memmap(
            self.path, dtype="<i4", mode="r", shape=(self.total_tokens,)
        )

    def read(self, parent_index: int) -> np.ndarray:
        if isinstance(parent_index, bool) or int(parent_index) < 0:
            raise IndexError("parent_index must be a non-negative integer")
        offset = (int(parent_index) * self.sample_length) % self.total_tokens
        end = offset + self.sample_length
        if end <= self.total_tokens:
            return np.asarray(self._tokens[offset:end]).copy()
        parts = []
        remaining = self.sample_length
        cursor = offset
        while remaining:
            take = min(self.total_tokens - cursor, remaining)
            parts.append(np.asarray(self._tokens[cursor : cursor + take]))
            remaining -= take
            cursor = 0
        return np.concatenate(parts).astype("<i4", copy=False)

    def close(self) -> None:
        tokens = getattr(self, "_tokens", None)
        if tokens is not None:
            mmap = getattr(tokens, "_mmap", None)
            if mmap is not None:
                mmap.close()
            del self._tokens


@dataclass(frozen=True)
class WindowGeometry:
    child_length: int
    overlap: int
    children_per_parent: int

    @property
    def stride(self) -> int:
        return self.child_length - self.overlap

    @property
    def parent_length(self) -> int:
        return self.child_length + (self.children_per_parent - 1) * self.stride


class WindowedParentPlan:
    """Lazy, epoch-addressable parent prompt plan.

    Only one parent token list is materialized by :meth:`iter_epoch` at a time.
    """

    is_windowed_parent_plan = True

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        geometry: WindowGeometry,
        seed: int,
        shuffle: bool,
        max_parents: Optional[int] = None,
    ) -> None:
        self.path = str(Path(path).expanduser().resolve())
        self.geometry = geometry
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.reader = Int32RingReader(self.path, geometry.parent_length)
        available = self.reader.parents_per_epoch
        self.parents_per_epoch = (
            min(available, int(max_parents))
            if max_parents not in (None, 0)
            else available
        )
        stat = os.stat(self.path)
        identity = (
            f"{self.path}:{stat.st_size}:{stat.st_mtime_ns}:"
            f"{geometry.child_length}:{geometry.overlap}:"
            f"{geometry.children_per_parent}"
        )
        self.fingerprint = hashlib.sha256(identity.encode()).hexdigest()[:16]

    def __len__(self) -> int:
        return self.parents_per_epoch

    def _order(self, epoch: int) -> np.ndarray:
        order = np.arange(self.reader.parents_per_epoch, dtype=np.int64)
        if self.shuffle:
            # PCG64 is named explicitly so the checkpoint contract can reject a
            # future shuffle implementation change instead of silently drifting.
            rng = np.random.Generator(np.random.PCG64(self.seed + int(epoch)))
            rng.shuffle(order)
        return order[: self.parents_per_epoch]

    def iter_epoch(
        self, epoch: int, *, start_parent: int = 0
    ) -> Iterator[dict]:
        if start_parent < 0 or start_parent > self.parents_per_epoch:
            raise ValueError(
                f"start_parent {start_parent} outside [0, {self.parents_per_epoch}]"
            )
        order = self._order(epoch)
        for logical_position in range(start_parent, self.parents_per_epoch):
            physical_index = int(order[logical_position])
            tokens = self.reader.read(physical_index).astype(np.int64).tolist()
            parent_id = (
                f"e{int(epoch):06d}-p{logical_position:012d}-"
                f"src{physical_index:012d}"
            )
            yield {
                "task_id": parent_id,
                "source_id": self.path,
                "max_length": self.geometry.parent_length,
                "payload": {
                    "input_ids": tokens,
                    # The parent mask is transport-only. Child masks are rebuilt
                    # after capture according to the window geometry.
                    "loss_mask": [1] * self.geometry.parent_length,
                },
                "metadata": {
                    "windowed_parent": True,
                    "epoch": int(epoch),
                    "logical_parent_index": logical_position,
                    "physical_parent_index": physical_index,
                    "dataset_fingerprint": self.fingerprint,
                    "child_length": self.geometry.child_length,
                    "overlap": self.geometry.overlap,
                    "stride": self.geometry.stride,
                    "children_per_parent": self.geometry.children_per_parent,
                    "parent_length": self.geometry.parent_length,
                    "shuffle_algorithm": "numpy-pcg64-v1",
                    "num_tokens": self.geometry.parent_length,
                },
            }


__all__ = [
    "Int32RingReader",
    "WindowGeometry",
    "WindowedParentPlan",
]
