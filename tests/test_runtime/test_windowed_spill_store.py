"""WindowedFileSpillStore: child files + forwarded Mooncake parent lifecycle."""

import os
import tempfile
import unittest

import torch

from specforge.runtime.contracts import SampleRef
from specforge.runtime.data_plane.mooncake_store import MooncakeFeatureStore
from specforge.runtime.data_plane.windowed_spill_store import (
    WindowedFileSpillStore,
)
from tests.test_runtime.test_server_capture import _FakeMooncakeStore


RUN = "glm53-online"
PARENT_TASK = "e000000-p000000000000-src000000000042"
PARENT_ID = f"{RUN}:{PARENT_TASK}"
CHILD_ID = f"{PARENT_ID}:w0008"


def _store(root, **kw):
    mooncake = MooncakeFeatureStore(store=_FakeMooncakeStore(), store_id=RUN)
    return WindowedFileSpillStore(mooncake, spill_root=root, run_id=RUN, **kw)


def _child_tensors():
    return {
        "input_ids": torch.arange(8, dtype=torch.long).unsqueeze(0),
        "hidden_states": torch.zeros((1, 8, 4), dtype=torch.bfloat16),
    }


class WindowedFileSpillStoreTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = self._tmp.name

    def test_retain_on_release_contract(self):
        store = _store(self.root)
        self.assertTrue(store.retain_on_release)

    def test_child_put_publishes_file_ref_and_layout_is_reversible(self):
        store = _store(self.root)
        ref = store.put(
            _child_tensors(),
            sample_id=CHILD_ID,
            metadata={"run_id": RUN, "strategy": "dspark"},
        )
        self.assertTrue(ref.feature_store_uri.startswith("file://"))
        self.assertEqual(ref.feature_keys, {"input_ids": "input_ids"})
        self.assertEqual(ref.estimated_bytes, 8 * 8 + 8 * 4 * 2)
        # layout: {root}/{run}/{parent}/w0008.ckpt
        rel = os.path.relpath(
            ref.feature_store_uri[len("file://") :], self.root
        )
        self.assertEqual(
            rel,
            os.path.join(RUN.replace(":", "__"), PARENT_ID.replace(":", "__"), "w0008.ckpt"),
        )

    def test_child_roundtrip_and_abort_deletes_file(self):
        store = _store(self.root)
        ref = store.put(
            _child_tensors(),
            sample_id=CHILD_ID,
            metadata={"run_id": RUN},
        )
        tensors, handle = store.get(ref)
        self.assertEqual(tensors["input_ids"].shape, (1, 8))
        self.assertTrue(handle.lease_token.startswith("file:"))
        store.release(handle, reason="loaded")
        path = ref.feature_store_uri[len("file://") :]
        self.assertTrue(os.path.exists(path), "release must retain the file")
        store.abort(CHILD_ID, reason="optimizer-boundary-durable-ack")
        self.assertFalse(os.path.exists(path))
        # idempotent: reconcile / dp_ack retries must not raise
        store.abort(CHILD_ID, reason="optimizer-boundary-durable-ack")

    def test_spill_max_bytes_raises_memory_error(self):
        store = _store(self.root, spill_max_bytes=16)
        with self.assertRaises(MemoryError):
            store.put(
                _child_tensors(),
                sample_id=CHILD_ID,
                metadata={"run_id": RUN},
            )
        self.assertEqual(os.listdir(os.path.join(self.root)), [])

    def test_get_of_missing_child_key_is_loud(self):
        store = _store(self.root)
        ref = store.put(
            _child_tensors(),
            sample_id=CHILD_ID,
            metadata={"run_id": RUN},
        )
        bad = SampleRef(
            **{
                **{
                    field: getattr(ref, field)
                    for field in (
                        "sample_id",
                        "run_id",
                        "source_task_id",
                        "feature_store_uri",
                        "feature_keys",
                        "feature_specs",
                        "strategy",
                    )
                },
                "feature_keys": {"input_ids": "input_ids", "nope": "nope"},
            }
        )
        with self.assertRaises(KeyError):
            store.get(bad)

    def test_parent_operations_forward_to_mooncake(self):
        store = _store(self.root)
        mooncake = store._mooncake
        parent_ref = mooncake.put(
            {"hidden_states": torch.zeros((1, 4, 4), dtype=torch.bfloat16)},
            sample_id=PARENT_ID,
            metadata={"run_id": RUN, "num_tokens": 4},
        )
        # adopt: parent refs forward; file:// child refs are a no-op
        store.adopt(parent_ref)
        child_ref = store.put(
            _child_tensors(),
            sample_id=CHILD_ID,
            metadata={"run_id": RUN},
        )
        store.adopt(child_ref)  # must not raise (no generation on file refs)

        # A pending external capture attempt (registered, never adopted) is
        # reclaimed through the forwarded discard path.
        pending_parent = f"{RUN}:e000000-p000000000001-src000000000001"
        store.track_external_attempt(
            pending_parent, generation=7, feature_names=["hidden_states"]
        )
        self.assertEqual(mooncake.health()["provisional_external"], 1)
        self.assertEqual(store.discard_external_attempts(), 1)
        self.assertEqual(mooncake.health()["provisional_external"], 0)

        # parent get/release/abort ride Mooncake
        tensors, handle = store.get(parent_ref)
        self.assertEqual(tensors["hidden_states"].shape, (1, 4, 4))
        store.release(handle)
        with self.assertRaises(KeyError):
            mooncake.get(parent_ref)
        # parent abort must not touch the child's spill file
        child_path = child_ref.feature_store_uri[len("file://") :]
        store.abort(PARENT_ID, reason="window-parent-fanned-out")
        self.assertTrue(os.path.exists(child_path))

    def test_health_reports_spill_usage(self):
        store = _store(self.root, spill_max_bytes=1 << 20)
        store.put(
            _child_tensors(),
            sample_id=CHILD_ID,
            metadata={"run_id": RUN},
        )
        health = store.health()
        self.assertEqual(health["backend"], "windowed-file-spill")
        self.assertEqual(health["spill_max_bytes"], 1 << 20)
        self.assertGreater(health["spill_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
