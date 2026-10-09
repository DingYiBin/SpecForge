import copy
import unittest

from specforge.config import Config
from specforge.training.disaggregated import (
    _windowed_data_contract,
    _windowed_resume_parent,
)


class _Plan:
    fingerprint = "dataset-fingerprint"

    def __len__(self):
        return 10


def _config():
    return Config.model_validate(
        {
            "model": {
                "target_model_path": "target",
                "draft_model_config": "draft.json",
            },
            "data": {
                "parent_bin_path": "/tokens.bin",
                "max_length": 1024,
                "windowed_online": {
                    "child_length": 1024,
                    "overlap": 128,
                    "children_per_parent": 48,
                },
            },
            "runtime": {"producer_lease": 1},
            "training": {
                "strategy": "dspark",
                "batch_size": 1,
                "accumulation_steps": 1,
                "max_steps": 12,
                "save_interval": 6,
            },
            "deployment": {
                "mode": "disaggregated",
                "trainer": {"nproc_per_node": 8},
                "disaggregated": {
                    "control_dir": "/control",
                    "backend": "mooncake",
                    "server_urls": ["http://capture"],
                    "producer_segment_size": 6 << 30,
                },
            },
        }
    )


class WindowedResumeTest(unittest.TestCase):
    def test_global_step_maps_to_whole_parent(self):
        cfg = _config()
        plan = _Plan()
        state = {
            "global_step": 6,
            "windowed_online_data": _windowed_data_contract(cfg, plan),
        }
        self.assertEqual(_windowed_resume_parent(cfg, plan, state), 1)

    def test_dataset_contract_drift_is_rejected(self):
        cfg = _config()
        plan = _Plan()
        contract = _windowed_data_contract(cfg, plan)
        changed = copy.deepcopy(contract)
        changed["dataset_fingerprint"] = "other"
        with self.assertRaisesRegex(ValueError, "does not match"):
            _windowed_resume_parent(
                cfg,
                plan,
                {"global_step": 6, "windowed_online_data": changed},
            )


if __name__ == "__main__":
    unittest.main()
