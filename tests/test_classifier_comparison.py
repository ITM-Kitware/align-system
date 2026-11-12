import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/summarize_classifier_comparison.py"


class ComparisonCliTest(unittest.TestCase):
    def test_pairs_changed_input_ids_and_rejects_foreign_predictions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            specs = []
            for name, identities, predicted in (
                ("old", ("a1", "a2"), ("B-line-C", "Meander-C")),
                ("new", ("b1", "b2"), ("B-line-C", "Meander-A")),
            ):
                requests = []
                predictions = []
                output = root / name
                output.mkdir()
                for index, (identity, label, prediction) in enumerate(zip(
                    identities, ("B-line-C", "Meander-A"), predicted
                )):
                    requests.append({"request_id": identity, "original_request_id": f"a{index+1}",
                                     "true_label": label, "trajectory_id": f"episode-{index}"})
                    predictions.append({"request_id": identity, "true_label": label,
                                        "predicted_label": prediction, "inference_seconds": 1})
                request_path = root / f"{name}.jsonl"
                request_path.write_text("".join(json.dumps(row) + "\n" for row in requests))
                (output / "predictions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in predictions))
                (output / "run_manifest.json").write_text(json.dumps({
                    "requests_sha256": hashlib.sha256(request_path.read_bytes()).hexdigest(),
                    "model_snapshot": "/models/test-model/snapshot",
                }))
                specs.append({"name": name, "requests": request_path.name, "output": name,
                              "model_directory": "test-model", "model": "test",
                              "input_view": name, "blue_policy": "Monitor"})
            config = root / "config.json"
            config.write_text(json.dumps({"shared_root": str(root), "experiments": specs,
                                          "comparisons": [["old", "new"]], "interpretation": []}))
            command = [sys.executable, str(SCRIPT), "--config", str(config), "--output-dir", str(root / "report")]
            completed = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report_path = root / "report/comparison.json"
            preserved = report_path.read_bytes()
            report = json.loads(preserved)
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["runs"]["old"]["accuracy"], 0.5)
            self.assertEqual(report["runs"]["new"]["accuracy"], 1.0)
            pair = report["paired_comparisons"][0]
            self.assertEqual(pair["paired_windows"], 2)
            self.assertEqual(pair["accuracy_difference_right_minus_left"], 0.5)
            with (root / "new/predictions.jsonl").open("a") as stream:
                stream.write(json.dumps({"request_id": "foreign", "true_label": "B-line-C",
                                         "predicted_label": "B-line-C", "inference_seconds": 1}) + "\n")
            failed = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertEqual(report_path.read_bytes(), preserved)


if __name__ == "__main__":
    unittest.main()
