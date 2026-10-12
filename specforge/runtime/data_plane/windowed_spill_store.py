# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Windowed-online A' data plane: Mooncake parents, per-child spill files.

The windowed online topology (one capture server, producer + N trainer ranks on
one node) previously round-tripped every *child* sample through Mooncake: the
producer fetched a ~43k-token parent, sliced 48 children locally, wrote all 48
back into the remote store, and every trainer rank fetched them back across the
wire. The remote writes dominated each ``run_once`` and put the child traffic
(one ``get_into`` per rank per child) on the critical path.

``WindowedFileSpillStore`` keeps the *parent* on Mooncake (only the producer
ever reads it, once) and materializes children as plain files under a shared
``spill_root``:

* ``put()`` ``torch.save``s the child tensors to
  ``{spill_root}/{run_id}/{sample_id}.ckpt`` (staged ``.tmp`` +
  ``os.replace`` atomic publish) and returns a ``file://``-addressed
  :class:`SampleRef` — the exact ref flavour the offline ``file://`` reader
  already serves, so ``FeatureDataLoader``/collate need no changes. The
  sanitized sample id carries the parent identity in the filename, so the
  run directory stays flat: per-parent subdirectories would only
  accumulate as empty husks once the optimizer-boundary abort deletes the
  child files.
* ``get()`` on a ``file://`` ref loads the named keys via
  ``load_feature_file`` (``torch.load(mmap=True)``): same-node ranks share the
  OS page cache, so the physical read happens once per node per child.
* ``abort()`` deletes the child file (idempotent; a missing file is already
  freed). ``dp_ack``'s optimizer-boundary abort therefore costs one
  ``os.remove`` instead of a lease-deferred Mooncake RPC — there is no -706
  retry window for children at all.

Every parent-lifecycle operation (``adopt``/``track_external_attempt``/
``discard_external_attempts``/``get`` of a parent ref/...) forwards to the
wrapped Mooncake store, so server capture, producer fan-out, and terminal
parent cleanup behave exactly as before.

The store is *not* a :class:`FeatureStore` subclass; it implements the
consumer-side ``retain_on_release`` contract (release is a bookkeeping no-op —
children are freed only by an explicit ``abort`` after the durable ack).

``spill_max_bytes`` bounds the directory (default
``runtime.feature_store_max_resident_bytes`` semantics): ``put()`` raises
:class:`MemoryError` when exceeded, which drives the producer's existing
backpressure / hard-cap paths.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from specforge.runtime.contracts import SCHEMA_VERSION, FeatureHandle, SampleRef
from specforge.runtime.data_plane.feature_store import (
    load_feature_file,
    spec_from_tensor,
)

_FILE_URI_PREFIX = "file://"

logger = logging.getLogger(__name__)

# sample ids look like ``{run_id}:{parent_task_id}:{window}`` (e.g.
# ``glm53-online:e000000-p...-src...:w0008``); ":" is not filesystem-safe.
# "__" is a safe reversible stand-in (task ids never contain "__"; the parent
# task id format is ``e{epoch}-p{pos}-src{src}``, separator "-").
_COLON = ":"
_COLON_ESC = "__"


def _sanitize_component(raw: str) -> str:
    return raw.replace(_COLON, _COLON_ESC)


def _spill_dir_bytes(root: str) -> int:
    """Recursive byte size of a directory tree (files + directories)."""
    total = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        try:
            total += os.lstat(dirpath).st_size
        except OSError:  # pragma: no cover - concurrent rmtree
            continue
        for name in filenames:
            try:
                total += os.lstat(os.path.join(dirpath, name)).st_size
            except OSError:  # pragma: no cover - deleted mid-walk
                continue
    return total


class WindowedFileSpillStore:
    """Mooncake parents + per-child spill files, behind one store object.

    ``mooncake`` must be a :class:`MooncakeFeatureStore`-like backend owning the
    parent lifecycle (``adopt``/``track_external_attempt``/
    ``discard_external_attempts``/``get``/``release``/``abort``/``gc``/
    ``drain_pending_removals``/``health``). Child samples written through this
    wrapper never touch Mooncake.
    """

    def __init__(
        self,
        mooncake,
        *,
        spill_root: str,
        run_id: str,
        spill_max_bytes: Optional[int] = None,
    ) -> None:
        self._mooncake = mooncake
        self.spill_root = os.path.abspath(os.path.expanduser(spill_root))
        self.run_id = str(run_id)
        self.spill_max_bytes = spill_max_bytes
        self.store_id = getattr(mooncake, "store_id", None) or self.run_id
        # Online consumer contract (launch.py consumer preflight): children are
        # freed only by an explicit abort at the optimizer-boundary durable ack.
        self.retain_on_release = True
        # In-process lease bookkeeping for file:// refs (release is a no-op on
        # the physical file; the lease table exists so get()/release() pair
        # correctly and health() can report borrow counts).
        self._active_leases: Dict[str, FeatureHandle] = {}
        self._counter = 0
        self._lock = threading.RLock()

    # -- layout ------------------------------------------------------------
    def _child_path(self, sample_id: str) -> str:
        """``{spill_root}/{run_id}/{sample_id}.ckpt`` for one sample id.

        The sanitized child id is ``{run_id}__{parent_task_id}__w{index}``,
        so the flat filename already carries the parent identity. Per-parent
        subdirectories were tried first and accumulate as empty husks: only
        the child *files* are removed at the optimizer-boundary abort, nobody
        removes the directory. A raw non-windowed id (no ``:w`` suffix, e.g. a
        plain producer-put sample) maps to a flat file the same way.
        """
        run_dir = _sanitize_component(self.run_id)
        return os.path.join(
            self.spill_root, run_dir, f"{_sanitize_component(sample_id)}.ckpt"
        )

    def _ensure_dir(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def _check_spill_budget(self, incoming: int) -> None:
        if self.spill_max_bytes is None:
            return
        run_dir = os.path.join(self.spill_root, _sanitize_component(self.run_id))
        used = _spill_dir_bytes(run_dir) if os.path.isdir(run_dir) else 0
        if used + incoming > self.spill_max_bytes:
            raise MemoryError(
                f"windowed spill directory {run_dir} over budget: "
                f"{used + incoming} > {self.spill_max_bytes} bytes; "
                "consumer is behind"
            )

    def over_budget(self, extra_bytes: int = 0) -> bool:
        """True when the spill directory already leaves no headroom.

        Producer-side backpressure probe: lets the drive loop pause BEFORE
        leasing a parent, because discovering the overflow inside ``put()``
        costs a full server capture (~43k-token generate) plus a 2.8 GiB
        parent fetch on every retry.
        """
        if self.spill_max_bytes is None:
            return False
        run_dir = os.path.join(self.spill_root, _sanitize_component(self.run_id))
        used = _spill_dir_bytes(run_dir) if os.path.isdir(run_dir) else 0
        return used + int(extra_bytes) > self.spill_max_bytes

    # -- child write/read/free (files) --------------------------------------
    def put(
        self,
        tensors: Dict[str, torch.Tensor],
        *,
        sample_id: str,
        metadata: Dict[str, Any],
    ) -> SampleRef:
        """Write one child sample to a spill file; return a ``file://`` ref."""
        if not tensors:
            raise ValueError("put requires at least one tensor")
        # Guard against views into a larger storage (e.g. a slice of a parent
        # hidden state): torch.save serializes the WHOLE underlying storage of
        # every tensor, so a small slice would silently write the full parent
        # (~GBs) into the spill file. Materialize a minimal copy first.
        staged = {}
        for name, value in tensors.items():
            t = value.detach().cpu()
            own_bytes = t.numel() * t.element_size()
            storage_bytes = t.untyped_storage().nbytes()
            if storage_bytes > own_bytes:
                logger.warning(
                    "put(%s): tensor %r is a view into a %d-byte storage but "
                    "only owns %d bytes; cloning to avoid serializing the "
                    "shared storage",
                    sample_id,
                    name,
                    storage_bytes,
                    own_bytes,
                )
                t = t.clone()
            staged[name] = t.contiguous()
        specs = {k: spec_from_tensor(k, v) for k, v in staged.items()}
        nbytes = sum(t.numel() * t.element_size() for t in staged.values())
        self._check_spill_budget(nbytes)
        path = self._child_path(sample_id)
        self._ensure_dir(path)
        tmp = path + ".tmp"
        # Atomic publish: an interrupted torch.save leaves at most a .tmp file
        # (never a half-written .ckpt the consumer could mmap).
        torch.save(staged, tmp)
        os.replace(tmp, path)
        # Sanity check: the file on disk must match the claimed tensor bytes.
        # A large mismatch means a shared/oversized storage leaked into the
        # file and the spill budget is being consumed far faster than
        # `estimated_bytes` suggests.
        file_bytes = os.path.getsize(path)
        if file_bytes > nbytes + (4 << 20):
            logger.warning(
                "spill file %s is %d bytes but its tensors only claim %d "
                "bytes; oversized storage was serialized",
                path,
                file_bytes,
                nbytes,
            )
        return SampleRef(
            sample_id=sample_id,
            run_id=str(metadata.get("run_id", self.run_id)),
            source_task_id=metadata.get("source_task_id"),
            feature_store_uri=f"{_FILE_URI_PREFIX}{path}",
            feature_keys={k: k for k in staged},
            feature_specs=specs,
            strategy=metadata.get("strategy", "eagle3"),
            schema_version=int(metadata.get("schema_version", SCHEMA_VERSION)),
            target_model_version=str(metadata.get("target_model_version", "unknown")),
            draft_weight_version=metadata.get("draft_weight_version"),
            tokenizer_version=str(metadata.get("tokenizer_version", "unknown")),
            num_tokens=int(metadata.get("num_tokens", 0)),
            estimated_bytes=nbytes,
            metadata={k: v for k, v in metadata.items() if k != "num_tokens"},
        )

    def get(
        self,
        sample_ref: SampleRef,
        *,
        device: "torch.device | str" = "cpu",
        names: Optional[List[str]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], FeatureHandle]:
        uri = sample_ref.feature_store_uri
        if uri.startswith(_FILE_URI_PREFIX):
            wanted = names or list(sample_ref.feature_keys.keys())
            raw = load_feature_file(uri[len(_FILE_URI_PREFIX) :])
            out: Dict[str, torch.Tensor] = {}
            for n in wanted:
                raw_key = sample_ref.feature_keys.get(n, n)
                if raw_key not in raw:
                    raise KeyError(
                        f"{uri} missing key {raw_key!r} for feature {n!r}"
                    )
                out[n] = raw[raw_key]
            handle = self._register_file_lease(sample_ref)
            try:
                if str(device) != "cpu":
                    out = {k: v.to(device) for k, v in out.items()}
            except Exception:
                self.release(handle, reason="materialization_failed")
                raise
            return out, handle
        return self._mooncake.get(sample_ref, device=device, names=names)

    def release(self, handle: FeatureHandle, *, reason: str = "consumed") -> None:
        if handle.lease_token.startswith("file:"):
            # Retain-on-release contract: the child file stays until the
            # optimizer-boundary durable ack aborts it. Only drop the lease.
            with self._lock:
                self._active_leases.pop(handle.lease_token, None)
            return
        self._mooncake.release(handle, reason=reason)

    def abort(self, sample_id: str, *, reason: str = "aborted") -> None:
        """Free a sample. Child -> delete spill file; parent -> Mooncake abort.

        ``sample_id`` alone is ambiguous between a child (``...:w0008``) and
        its parent; route by the window suffix, which only children carry.
        """
        sid = str(sample_id)
        head, sep, tail = sid.rpartition(":")
        if sep and head and tail.startswith("w") and tail[1:].isdigit():
            path = self._child_path(sid)
            try:
                os.remove(path)
            except FileNotFoundError:
                pass  # already freed; abort is idempotent
            except OSError:
                # keep parity with Mooncake semantics: a failed free must stay
                # visible so the terminal drain cannot silently drop it
                raise
            return
        self._mooncake.abort(sid, reason=reason)

    def _register_file_lease(self, ref: SampleRef) -> FeatureHandle:
        with self._lock:
            self._counter += 1
            handle = FeatureHandle(
                sample_id=ref.sample_id,
                generation=0,
                lease_token=f"file:{ref.sample_id}:{self._counter}",
            )
            self._active_leases[handle.lease_token] = handle
            return handle

    # -- parent lifecycle (forwarded to Mooncake) ----------------------------
    def adopt(self, sample_ref: SampleRef) -> None:
        """Register an externally-produced sample for lifecycle management.

        Child refs (``file://``) carry no Mooncake generation — their physical
        storage is a spill file this class frees directly in :meth:`abort`, so
        there is nothing to register. Restart reconciliation may still call
        ``adopt`` on them before aborting; accept that as a no-op.
        """
        if sample_ref.feature_store_uri.startswith(_FILE_URI_PREFIX):
            return
        self._mooncake.adopt(sample_ref)

    def track_external_attempt(
        self, sample_id: str, *, generation: int, feature_names: List[str]
    ) -> None:
        self._mooncake.track_external_attempt(
            sample_id, generation=generation, feature_names=feature_names
        )

    def discard_external_attempts(self, *, reason: str = "unadopted-external-capture") -> int:
        return self._mooncake.discard_external_attempts(reason=reason)

    def gc(self, *, now: Optional[float] = None) -> Dict[str, int]:
        return self._mooncake.gc(now=now)

    def drain_pending_removals(
        self,
        *,
        max_attempts: int = 3,
        retry_interval_s: float = 12.0,
        sleep: Callable[[float], None] = None,
    ) -> Dict[str, int]:
        # Only parent objects can be lease-deferred; child frees are file
        # deletes. Default window (3 x 12s) outlasts Mooncake's 10s read lease.
        if sleep is None:
            import time as _time

            sleep = _time.sleep
        return self._mooncake.drain_pending_removals(
            max_attempts=max_attempts,
            retry_interval_s=retry_interval_s,
            sleep=sleep,
        )

    def health(self) -> Dict[str, Any]:
        base = dict(self._mooncake.health())
        run_dir = os.path.join(self.spill_root, _sanitize_component(self.run_id))
        with self._lock:
            file_leases = len(self._active_leases)
        base.update(
            {
                "backend": "windowed-file-spill",
                "spill_root": self.spill_root,
                "spill_bytes": (
                    _spill_dir_bytes(run_dir) if os.path.isdir(run_dir) else 0
                ),
                "spill_max_bytes": self.spill_max_bytes,
                "file_active_leases": file_leases,
            }
        )
        return base

    def estimate_bytes(self, specs) -> int:
        return self._mooncake.estimate_bytes(specs)


__all__ = ["WindowedFileSpillStore"]
