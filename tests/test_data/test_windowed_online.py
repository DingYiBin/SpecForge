import os
import tempfile
import unittest

import numpy as np

from specforge.data.windowed_online import (
    Int32RingReader,
    WindowGeometry,
    WindowedParentPlan,
)


class WindowedOnlineDataTest(unittest.TestCase):
    def test_ring_reader_wraps_and_repeats(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "tokens.bin")
            np.arange(10, dtype="<i4").tofile(path)
            reader = Int32RingReader(path, 6)
            self.assertEqual(reader.parents_per_epoch, 2)
            self.assertEqual(reader.read(0).tolist(), [0, 1, 2, 3, 4, 5])
            self.assertEqual(reader.read(1).tolist(), [6, 7, 8, 9, 0, 1])
            self.assertEqual(reader.read(2).tolist(), [2, 3, 4, 5, 6, 7])
            reader.close()

    def test_parent_plan_is_lazy_and_shuffle_is_reproducible(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "tokens.bin")
            np.arange(40, dtype="<i4").tofile(path)
            geometry = WindowGeometry(
                child_length=4, overlap=1, children_per_parent=3
            )
            first = WindowedParentPlan(
                path, geometry=geometry, seed=7, shuffle=True
            )
            second = WindowedParentPlan(
                path, geometry=geometry, seed=7, shuffle=True
            )
            first_epoch = list(first.iter_epoch(0))
            second_epoch = list(second.iter_epoch(0))
            self.assertEqual(
                [row["task_id"] for row in first_epoch],
                [row["task_id"] for row in second_epoch],
            )
            self.assertEqual(geometry.parent_length, 10)
            self.assertEqual(len(first_epoch[0]["payload"]["input_ids"]), 10)
            self.assertEqual(first_epoch[0]["payload"]["loss_mask"], [1] * 10)
            self.assertNotEqual(
                [row["task_id"] for row in first_epoch],
                [row["task_id"] for row in first.iter_epoch(1)],
            )


if __name__ == "__main__":
    unittest.main()
