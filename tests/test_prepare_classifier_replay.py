"""Exercise replay preparation through its command-line interface."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_classifier_replay.py"
LABELS = ["B-line-C", "B-line-I", "B-line-A", "Meander-C", "Meander-I", "Meander-A"]
DIRECTORIES = ["bline_c", "bline_i", "bline_a", "meander_c", "meander_i", "meander_a"]


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def make_run(root):
    root.mkdir()
    for directory, label in zip(DIRECTORIES, LABELS):
        folder = root / directory
        folder.mkdir()
        (folder / ".hydra").mkdir()
        (folder / ".hydra/config.yaml").write_text("model: saved-model\n")
        write_json(folder / "run_manifest.json", {"replay": {"window_size": 2}})
        predictions, history, lines = [], [], []
        for i, start in enumerate([10, 20]):
            sid = f"{directory}/E{i}.json#steps={start}:{start + 2}"
            chosen = label if i == 0 else LABELS[(LABELS.index(label) + 1) % len(LABELS)]
            messages = [
                {"role": "system", "content": "Use these historical instructions exactly."},
                {"role": "user", "content": f"# OBSERVATIONS\nStep {start}:\n  Blue: Monitor\nStep {start + 1}:\n  Blue: Monitor"},
            ]
            lines.extend([f"[bold]*Scenario ID*[/bold]: {sid}", json.dumps(messages)])
            if i == 0:
                lines.append(json.dumps(messages))
            predictions.append({
                "scenario_id": sid, "trajectory_id": f"{directory}/E{i}.json",
                "window_start": start, "window_end": start + 2, "window_size": 2,
                "true_label": label, "predicted_label": chosen, "inference_request_indices": [i],
            })
            history.append({
                "requested_model": "saved-model", "request_model": "saved-model:provider",
                "provider": "provider", "model": "served-model-Turbo",
                "temperature": 0.0, "top_p": 1.0, "seed": 0, "max_tokens": 32 if i == 0 else 256,
                "status": "completed", "response_text": json.dumps({"action_choice": chosen}),
                "request_sha256": str(i) * 64,
                "response_format": {"type": "json_schema", "json_schema": {
                    "name": "response", "strict": True, "schema": {"type": "object",
                    "properties": {"action_choice": {"type": "string", "enum": LABELS}},
                    "required": ["action_choice"], "additionalProperties": False}}},
            })
        write_json(folder / "classification_results.json", {
            "complete": True, "evaluation_unit": "window", "labels": LABELS,
            "n_labeled": 2, "n_correct": 1, "accuracy": 0.5, "predictions": predictions,
        })
        write_json(folder / "request_history.json", history)
        (folder / "raw_align_system.log").write_text("\n".join(lines) + "\n")
    write_json(root / "classification_report.json", {"n_samples": 12, "n_correct": 6, "overall_accuracy": 0.5})


class PrepareReplayCLI(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run = self.root / "run"
        make_run(self.run)

    def invoke(self, output_name="prepared", *options):
        output = self.root / output_name
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--run-dir", str(self.run), "--output-dir", str(output), *options],
            text=True, capture_output=True,
        )
        return result, output

    def test_complete_dataset_preserves_inputs_settings_and_scoring(self):
        result, output = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = [json.loads(line) for line in (output / "requests.jsonl").read_text().splitlines()]
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(len(rows), 12)
        self.assertEqual(len({row["request_id"] for row in rows}), 12)
        self.assertEqual(manifest["source_accuracy"], 0.5)
        self.assertEqual(manifest["source_class_counts"], dict.fromkeys(LABELS, 2))
        self.assertEqual(manifest["source_final_request_token_caps"], {"32": 6, "256": 6})
        self.assertEqual(manifest["identical_resumed_dialogues_collapsed"], 6)
        self.assertEqual(manifest["source_models"][0]["served_model"], "served-model-Turbo")
        self.assertEqual(manifest["requests_sha256"], hashlib.sha256((output / "requests.jsonl").read_bytes()).hexdigest())
        for row in rows:
            self.assertEqual(row["messages"][0]["content"], "Use these historical instructions exactly.")
            self.assertNotIn(row["true_label"], row["messages"][1]["content"])
            self.assertEqual(set(row["messages"][1]), {"role", "content"})
            self.assertEqual(row["response_format"]["json_schema"]["schema"]["properties"]["action_choice"]["enum"], LABELS)

    def test_balanced_subset_is_reproducible_and_keeps_full_source_statistics(self):
        first, first_dir = self.invoke("first", "--per-class", "1")
        second, second_dir = self.invoke("second", "--per-class", "1")
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual((first_dir / "requests.jsonl").read_bytes(), (second_dir / "requests.jsonl").read_bytes())
        manifest = json.loads((first_dir / "manifest.json").read_text())
        self.assertEqual(manifest["selected_class_counts"], dict.fromkeys(LABELS, 1))
        self.assertEqual(manifest["source_count"], 12)
        self.assertEqual(manifest["request_count"], 6)

    def test_existing_output_is_preserved(self):
        result, output = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        original = (output / "requests.jsonl").read_bytes()
        result, _ = self.invoke()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("already exists", result.stderr)
        self.assertEqual((output / "requests.jsonl").read_bytes(), original)

    def test_missing_evidence_fails_before_creating_output(self):
        (self.run / "bline_c/request_history.json").unlink()
        result, output = self.invoke()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Missing evidence", result.stderr)
        self.assertFalse(output.exists())

    def test_conflicting_resumed_prompt_is_rejected(self):
        path = self.run / "bline_c/raw_align_system.log"
        lines = path.read_text().splitlines()
        dialogue = json.loads(lines[2])
        dialogue[0]["content"] = "Different instructions for the same prediction."
        lines[2] = json.dumps(dialogue)
        path.write_text("\n".join(lines))
        result, output = self.invoke()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("conflicting dialogues", result.stderr)
        self.assertFalse(output.exists())

    def test_prediction_response_mismatch_is_rejected(self):
        path = self.run / "bline_c/request_history.json"
        history = json.loads(path.read_text())
        history[0]["response_text"] = json.dumps({"action_choice": "Meander-A"})
        write_json(path, history)
        result, output = self.invoke()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("final response disagrees", result.stderr)
        self.assertFalse(output.exists())

    def test_duplicate_prediction_ids_are_rejected(self):
        path = self.run / "bline_c/classification_results.json"
        results = json.loads(path.read_text())
        results["predictions"][1] = results["predictions"][0]
        write_json(path, results)
        result, output = self.invoke()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("duplicate prediction IDs", result.stderr)
        self.assertFalse(output.exists())

    def test_window_mismatch_is_rejected(self):
        path = self.run / "bline_c/raw_align_system.log"
        path.write_text(path.read_text().replace("Step 10:", "Step 11:"))
        result, output = self.invoke()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("dialogue steps do not match", result.stderr)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
