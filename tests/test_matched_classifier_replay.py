import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from test_classifier_replay import request, REPLAY


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/prepare_matched_classifier_replay.py'


class MatchedReplayTest(unittest.TestCase):
    def test_cli_retains_both_views_and_balances_early_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original, paired = [], []
            for label in REPLAY.LABELS:
                for episode in range(2):
                    for start in range(20):
                        row = request(f'{label}:{episode}:{start}')
                        row.update(true_label=label, trajectory_id=f'E{episode}',
                                   provenance={'class_directory': label, 'window_start': start, 'window_end': start+2})
                        original.append(row)
                        paired.append({**row, 'request_id': 'blue-' + row['request_id'],
                                       'original_request_id': row['request_id'],
                                       'messages': [{'role': 'user', 'content': f'Blue view {episode}:{start}'}]})
            for name, rows in (('original', original), ('paired', paired)):
                (root / f'{name}.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
            command = [sys.executable, str(SCRIPT), '--requests', str(root/'original.jsonl'),
                       '--paired-requests', str(root/'paired.jsonl'), '--output-dir', str(root/'output')]
            result = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            selected = [json.loads(line) for line in (root/'output/privileged-requests.jsonl').read_text().splitlines()]
            blue = [json.loads(line) for line in (root/'output/defender-requests.jsonl').read_text().splitlines()]
            self.assertEqual(len(selected), 60)
            self.assertEqual(len({(r['true_label'], r['trajectory_id']) for r in selected}), 12)
            for offset in range(0, len(selected), 6):
                self.assertEqual([r['true_label'] for r in selected[offset:offset+6]], list(REPLAY.LABELS))
            source = {r['request_id']:r for r in original}
            counterpart = {r['original_request_id']:r for r in paired}
            for row, observation in zip(selected, blue):
                self.assertEqual(row, source[row['request_id']])
                self.assertEqual(observation, counterpart[row['request_id']])
            starts = [r['provenance']['window_start'] for r in selected
                      if r['true_label']==REPLAY.LABELS[0] and r['trajectory_id']=='E0']
            self.assertEqual(starts, [0, 4, 9, 14, 19])
            paired[0]['true_label'] = REPLAY.LABELS[1]
            (root/'paired.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in paired))
            command[-1] = str(root/'bad-output')
            failed = subprocess.run(command, text=True, capture_output=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn('Paired true_label differs', failed.stderr)
            self.assertFalse((root/'bad-output').exists())


if __name__ == '__main__':
    unittest.main()
