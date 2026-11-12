import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_classifier_replay.py"
SPEC = importlib.util.spec_from_file_location("classifier_replay", SCRIPT)
REPLAY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPLAY)


def request(identity="one"):
    return {
        "request_id": identity,
        "true_label": "B-line-C",
        "messages": [{"role": "user", "content": "Classify these observed actions."}],
        "response_format": {"json_schema": {"schema": {
            "properties": {"action_choice": {"enum": list(REPLAY.LABELS)}}}}},
        "generation": {"temperature": 0, "top_p": 1, "max_tokens": 32},
    }


class ReplaySafety(unittest.TestCase):
    def test_validation_cli_needs_no_inference_dependencies(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "requests.jsonl"
            path.write_text(json.dumps(request()) + "\n")
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--requests", str(path), "--validate-only"],
                text=True, capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["n_requests"], 1)

    def test_validation_rejects_duplicate_requests(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "requests.jsonl"
            path.write_text((json.dumps(request()) + "\n") * 2)
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--requests", str(path), "--validate-only"],
                text=True, capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Duplicate request_id", result.stderr)

    def test_validation_rejects_saved_answers_as_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "requests.jsonl"
            item = request()
            item["messages"].append({"role": "assistant", "content": "B-line-C"})
            path.write_text(json.dumps(item) + "\n")
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--requests", str(path), "--validate-only"],
                text=True, capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("system/user input", result.stderr)

    def test_resume_preserves_completed_predictions_and_interrupted_tail(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "predictions.jsonl"
            complete = json.dumps({"request_id": "one", "true_label": "B-line-C",
                                   "predicted_label": "B-line-I"}) + "\n"
            partial = b'{"request_id":"two"'
            path.write_bytes(complete.encode() + partial)
            rows = REPLAY.read_results(path, [request("one"), request("two")])
            self.assertEqual(list(rows), ["one"])
            self.assertEqual(path.read_text(), complete)
            tails = list(Path(temporary).glob("incomplete-tail-*.bin"))
            self.assertEqual(len(tails), 1)
            self.assertEqual(tails[0].read_bytes(), partial)

    def test_resume_rejects_mismatched_labels_without_changing_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "predictions.jsonl"
            original = json.dumps({"request_id": "one", "true_label": "Meander-A",
                                   "predicted_label": "B-line-C"}) + "\n"
            path.write_text(original)
            with self.assertRaisesRegex(ValueError, "true label differs"):
                REPLAY.read_results(path, [request()])
            self.assertEqual(path.read_text(), original)

    def test_partial_report_cannot_claim_complete_accuracy(self):
        with tempfile.TemporaryDirectory() as temporary:
            rows = {"one": {"true_label": "B-line-C", "predicted_label": "B-line-C",
                             "inference_seconds": 1.25}}
            result = REPLAY.report(Path(temporary), [request("one"), request("two")], rows)
            self.assertEqual(result["status"], "partial")
            self.assertEqual(result["n_samples"], 1)
            self.assertEqual(result["expected_samples"], 2)
            self.assertEqual(result["per_class"]["Meander-A"]["accuracy"], None)

    def test_report_identifies_defender_view_without_claiming_prompt_adaptation(self):
        with tempfile.TemporaryDirectory() as temporary:
            item = {**request(), "input_view": "defender_table_and_blue_action",
                    "condition": "active_hybrid_ppo"}
            REPLAY.report(Path(temporary), [item], {})
            result = json.loads((Path(temporary) / 'classification_report.json').read_text())
            self.assertEqual(result['input_protocol'], {
                'input_views': ['defender_table_and_blue_action'],
                'conditions': ['active_hybrid_ppo'],
                'prompt_protocol': 'historical_instructions_and_examples_held_fixed',
            })

    def test_report_keeps_peak_memory_across_resume_and_completed_recheck(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            REPLAY.report(output, [request()], {}, gpu_memory_bytes=1000)
            resumed = REPLAY.report(output, [request()], {}, gpu_memory_bytes=700)
            self.assertEqual(resumed["peak_gpu_allocated_bytes"], 1000)
            checked = REPLAY.report(output, [request()], {})
            self.assertEqual(checked["peak_gpu_allocated_bytes"], 1000)


if __name__ == "__main__":
    unittest.main()
