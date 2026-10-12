import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from specforge.inference.adapters.windowed_capture import (
    WindowedServerCaptureAdapter,
)
from specforge.inference.capture import CaptureConfig
from specforge.runtime.contracts import PromptTask, SampleRefGroup
from specforge.runtime.data_plane.mooncake_store import MooncakeFeatureStore
from specforge.runtime.data_plane.windowed_spill_store import (
    WindowedFileSpillStore,
)
from tests.test_runtime.test_server_capture import _FakeMooncakeStore


HIDDEN = 4
LAYERS = (1, 2, 3)


def _capture():
    return CaptureConfig.from_strategy(
        required_features={
            "input_ids",
            "loss_mask",
            "hidden_states",
            "target_last_hidden_states",
        },
        aux_hidden_state_layer_ids=LAYERS,
        target_repr="hidden_state",
        target_hidden_size=HIDDEN,
    )


def _spill_store(root):
    """A' topology: fake Mooncake backend for parents + tmpdir spill for children."""
    mooncake = MooncakeFeatureStore(store=_FakeMooncakeStore(), store_id="run")
    return WindowedFileSpillStore(
        mooncake,
        spill_root=root,
        run_id="run",
        spill_max_bytes=None,
    )


class _ParentSource:
    """Stands in for SGLangServerCaptureAdapter: writes parents to Mooncake."""

    def __init__(self, mooncake):
        # The wrapped mooncake store: parents must land there, not in spill.
        self.store = mooncake
        self.schema = SimpleNamespace(
            aux_feature="hidden_states",
            last_hidden_feature="target_last_hidden_states",
        )
        self.parents = []

    def produce_refs(self, tasks, *, capture):
        refs = []
        for task in tasks:
            length = len(task.payload["input_ids"])
            aux = torch.arange(length * len(LAYERS) * HIDDEN).reshape(
                1, length, len(LAYERS) * HIDDEN
            )
            target = torch.arange(length * HIDDEN).reshape(1, length, HIDDEN)
            ref = self.store.put(
                {
                    "hidden_states": aux.to(torch.bfloat16),
                    "target_last_hidden_states": target.to(torch.bfloat16),
                },
                sample_id=f"run:{task.task_id}",
                metadata={
                    "run_id": "run",
                    "strategy": "dspark",
                    "target_model_version": "target",
                    "num_tokens": length,
                },
            )
            self.parents.append(ref)
            refs.append(ref)
        return refs


class _TrackingSpillStore(WindowedFileSpillStore):
    def __init__(self, root):
        super().__init__(
            _mooncake=MooncakeFeatureStore(
                store=_FakeMooncakeStore(), store_id="run"
            ),
            spill_root=root,
            run_id="run",
        )
        self.aborted = []

    def abort(self, sample_id, *, reason="aborted"):
        self.aborted.append((sample_id, reason))
        return super().abort(sample_id, reason=reason)


class WindowedCaptureTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def test_parent_is_split_into_independent_children(self):
        store = _spill_store(self._tmp.name)
        parent_source = _ParentSource(store._mooncake)
        adapter = WindowedServerCaptureAdapter(
            parent_source,
            store,
            child_length=4,
            overlap=1,
            children_per_parent=3,
        )
        task = PromptTask(
            task_id="parent-0",
            run_id="run",
            source_id="tokens",
            payload={"input_ids": list(range(10)), "loss_mask": [1] * 10},
            max_length=10,
        )

        results = adapter.produce_refs([task], capture=_capture())

        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], SampleRefGroup)
        refs = results[0].refs
        self.assertEqual(len(refs), 3)
        self.assertEqual(len({ref.sample_id for ref in refs}), 3)
        # Children are published as file:// refs into the spill directory...
        for ref in refs:
            self.assertTrue(ref.feature_store_uri.startswith("file://"))
            self.assertTrue(ref.feature_store_uri.endswith(".ckpt"))
        # ...while the parent lives on (and is freed through) Mooncake.
        with self.assertRaises(KeyError):
            store._mooncake.get(parent_source.parents[0])

        expected_tokens = ([0, 1, 2, 3], [3, 4, 5, 6], [6, 7, 8, 9])
        global_coverage = [0] * 10
        for index, ref in enumerate(refs):
            tensors, handle = store.get(ref)
            self.assertEqual(tensors["input_ids"][0].tolist(), expected_tokens[index])
            self.assertEqual(
                tensors["loss_mask"][0].tolist(),
                [1, 1, 1, 1] if index == 0 else [0, 1, 1, 1],
            )
            self.assertEqual(tensors["hidden_states"].shape, (1, 4, 12))
            self.assertEqual(
                tensors["target_last_hidden_states"].shape, (1, 4, 4)
            )
            self.assertEqual(
                ref.feature_specs["target_last_hidden_states"].target_repr,
                "hidden_state",
            )
            start = index * 3
            for local, enabled in enumerate(tensors["loss_mask"][0].tolist()):
                global_coverage[start + local] += enabled
            # retain-on-release: releasing the read lease keeps the file
            store.release(handle)
            self.assertTrue(
                os.path.exists(ref.feature_store_uri[len("file://") :])
            )
        self.assertEqual(global_coverage, [1] * 10)
        # Freeing each child through abort deletes its spill file.
        for ref in refs:
            path = ref.feature_store_uri[len("file://") :]
            self.assertTrue(os.path.exists(path))
            store.abort(ref.sample_id, reason="consumed")
            self.assertFalse(os.path.exists(path))
            # abort is idempotent (dp_ack reconcile may repeat it)
            store.abort(ref.sample_id, reason="consumed")

    def test_post_put_validation_failure_rolls_back_child_and_parent(self):
        store = _TrackingSpillStore(self._tmp.name)
        parent_source = _ParentSource(store._mooncake)
        adapter = WindowedServerCaptureAdapter(
            parent_source,
            store,
            child_length=4,
            overlap=1,
            children_per_parent=3,
        )
        task = PromptTask(
            task_id="parent-failure",
            run_id="run",
            source_id="tokens",
            payload={"input_ids": list(range(10)), "loss_mask": [1] * 10},
            max_length=10,
        )

        with mock.patch(
            "specforge.inference.adapters.windowed_capture.verify_capture_specs",
            side_effect=ValueError("bad child specs"),
        ):
            with self.assertRaisesRegex(ValueError, "bad child specs"):
                adapter.produce_refs([task], capture=_capture())

        parent_id = parent_source.parents[0].sample_id
        self.assertIn(
            (f"{parent_id}:w0000", "window-child-validation-failed"),
            store.aborted,
        )
        self.assertIn((parent_id, "window-fanout-failed"), store.aborted)
        # Parent bookkeeping is on the wrapped Mooncake store.
        self.assertEqual(store._mooncake.health()["resident_samples"], 0)


if __name__ == "__main__":
    unittest.main()
