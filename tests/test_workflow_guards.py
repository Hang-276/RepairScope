"""Offline workflow guard checks, not live environment acceptance."""
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from b2_appworld.cli import doctor
from b2_appworld.config import load_config
from b2_appworld.runner import run_experiment, verify_test_protocol


class WorkflowGuards(unittest.TestCase):
    def test_final_flag_alone_never_opens_test_environment(self):
        config = load_config()
        with tempfile.TemporaryDirectory() as folder:
            config["output_dir"] = str(Path(folder) / "runs")
            with self.assertRaisesRegex(ValueError, "frozen protocol"):
                run_experiment(config, "test_normal", "baseline", "test", final=True)

    def test_test_config_and_checkpoint_must_belong_to_freeze(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as folder:
            try:
                os.chdir(folder)
                Path("scripts").mkdir()
                Path("scripts/freeze_protocol.py").write_text("# synthetic fixture")
                config = load_config()
                Path("runs/configs").mkdir(parents=True)
                Path("runs/configs/appworld.json").write_text(json.dumps(config))
                Path("runs/frozen_protocol.json").write_text(json.dumps({"files_sha256": {}}))
                with patch("b2_appworld.runner.subprocess.run") as verifier:
                    verify_test_protocol(config)
                    verifier.assert_called_once()
                    with self.assertRaisesRegex(ValueError, "seed differs"):
                        verify_test_protocol({**config, "seed": 101})
                    with self.assertRaisesRegex(ValueError, "checkpoint was not included"):
                        verify_test_protocol(config, "unfrozen_checkpoint")
            finally:
                os.chdir(previous)

    def test_doctor_does_not_mark_unpacked_code_ready_from_find_spec(self):
        fake_torch = types.ModuleType("torch")
        fake_torch.__version__ = "synthetic"
        fake_torch.cuda = types.SimpleNamespace(is_available=lambda: True, get_device_name=lambda i: "fixture GPU",
                                               mem_get_info=lambda i: (100, 200))
        config = load_config()
        with tempfile.TemporaryDirectory() as folder:
            config["root"] = folder
            with patch("b2_appworld.cli.importlib.util.find_spec", return_value=object()), \
                    patch.dict(sys.modules, {"torch": fake_torch}), \
                    patch("b2_appworld.runner.verify_upstream", return_value="synthetic source fixture"), \
                    patch("b2_appworld.datasets.dataset_manifest", return_value={"splits": {}}):
                report = doctor(config)
        self.assertFalse(report["live_ready"])
        self.assertFalse(report["appworld_unpacked"])
        self.assertIn("Static prerequisites", report["live_ready_scope"])


if __name__ == "__main__":
    unittest.main()
