import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from align_system.interfaces.cia_cage2_service import CAGEActionBasedServiceInterface


ROOT = Path(__file__).resolve().parents[1]
LABELS = ['B-line-C', 'B-line-I', 'B-line-A', 'Meander-C', 'Meander-I', 'Meander-A']


def save(path, value):
    path.write_text(json.dumps(value))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(root, active=False):
    captures = root / 'captures'
    captures.mkdir()
    rows = [{
        'step': i, 'blue_action': {'Action': 'Monitor', 'Status': 'TRUE'},
        'red_action': {'Action': 'PRIVATE_RED_ACTION', 'Host': 'PRIVATE_TARGET', 'Status': 'TRUE'},
        'defender_observation': f'VISIBLE_BLUE_TABLE_{i}',
        'defender_raw_observation': {'private_unused_raw': 'RAW_SENTINEL'},
        'metric_scores': {'private_metric': 'METRIC_SENTINEL'},
    } for i in range(3)]
    record = {
        'steps': rows, 'red_agent_name': 'BlineJumping_C',
        'observation_timing': 'post-action', 'cage_python_source_sha256': 'wrong-source',
        'repeat_capture_identical': True, 'monitor_control_reproduced': True,
        'initial_network_matches_monitor': True, 'training_seed': 0,
        'checkpoint_sha256': 'checkpoint', 'policy_deterministic': False,
    }
    sha = save(captures / 'episode.json', record)
    episode = {
        'class_directory': 'bline_c', 'trajectory_id': 'E0.json',
        'capture_path': 'episode.json', 'capture_sha256': sha,
        'fresh_process_repeat_identical': True, 'historical_initial_network_matches': True,
        'historical_metric_scores_match': True, 'historical_action_trace_matches': True,
    }
    summary = {'status': 'complete', 'episodes': [episode]}
    if active:
        summary['condition'] = 'active_hybrid_ppo'
    save(captures / 'summary.json', summary)
    observations = '\n'.join(
        f'Step {i}:\n  Blue: Monitor (success: TRUE)\n  Red: PRIVATE_RED_ACTION on PRIVATE_TARGET (success: TRUE)'
        for i in (1, 2)
    )
    item = {
        'request_id': 'original', 'scenario_id': 'E0.json#steps=1:3', 'trajectory_id': 'E0.json',
        'true_label': 'B-line-C', 'historical_prediction': 'Meander-A',
        'messages': [
            {'role': 'system', 'content': 'Fixed historical examples.'},
            {'role': 'user', 'content': '# OBSERVATIONS\n' + observations + '\n\n# AVAILABLE LABELS\n' + '\n'.join(LABELS)},
        ],
        'response_format': {'json_schema': {'schema': {'properties': {'action_choice': {'enum': LABELS}}}}},
        'generation': {'temperature': 0, 'top_p': 1, 'max_tokens': 32},
        'provenance': {'class_directory': 'bline_c', 'window_start': 1, 'window_end': 3,
                       'messages_sha256': 'historical', 'scoring_note': 'PRIVATE_LABEL_METADATA'},
    }
    requests = root / 'requests.jsonl'
    requests.write_text(json.dumps(item) + '\n')
    return captures, requests


def run_prepare(root, captures, requests, active=False):
    return subprocess.run([
        sys.executable, str(ROOT / 'scripts' / ('prepare_active_replay.py' if active else 'prepare_defender_replay.py')),
        '--requests', str(requests), '--capture-dir', str(captures), '--output-dir', str(root / 'prepared'),
    ], text=True, capture_output=True)


class DefenderClassificationTest(unittest.TestCase):
    def test_preparation_and_native_interface_show_identical_blue_only_windows(self):
        for active in (False, True):
            with self.subTest(active=active), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                captures, requests = fixture(root, active)
                if active:
                    # Active preparation must not need or serialize attacker actions.
                    path = captures / 'episode.json'
                    record = json.loads(path.read_text())
                    for row in record['steps']:
                        del row['red_action']
                    summary = json.loads((captures / 'summary.json').read_text())
                    summary['episodes'][0]['capture_sha256'] = save(path, record)
                    save(captures / 'summary.json', summary)
                process = run_prepare(root, captures, requests, active)
                self.assertEqual(process.returncode, 0, process.stderr)
                prepared = json.loads((root / 'prepared/defender-requests.jsonl').read_text())
                original = json.loads(requests.read_text())
                self.assertEqual(prepared['messages'][0], original['messages'][0])
                self.assertEqual(prepared['generation'], original['generation'])
                self.assertEqual(prepared['original_request_id'], original['request_id'])
                if active:
                    self.assertIsNone(prepared['historical_prediction'])
                else:
                    self.assertEqual((root / 'prepared/privileged-requests.jsonl').read_bytes(), requests.read_bytes())
                interface = CAGEActionBasedServiceInterface(
                    n_rollouts=1, episode_length=3, trajectory_dir=str(captures),
                    start_step=1, window_size=2, red_agent_labels=LABELS,
                )
                scenario = interface.start_scenario()
                expected = ('Step 1:\n  Blue: Monitor (success: TRUE)\n  Blue observation:\nVISIBLE_BLUE_TABLE_1\n'
                            'Step 2:\n  Blue: Monitor (success: TRUE)\n  Blue observation:\nVISIBLE_BLUE_TABLE_2')
                self.assertEqual(scenario.get_state().unstructured, expected)
                self.assertEqual(scenario.get_red_agent_label(), 'B-line-C')
                self.assertIn('# OBSERVATIONS\n' + expected + '\n\n# AVAILABLE LABELS\n',
                              prepared['messages'][-1]['content'])
                serialized = json.dumps(prepared['messages'])
                for private in ('PRIVATE_RED_ACTION', 'PRIVATE_TARGET', 'METRIC_SENTINEL',
                                'RAW_SENTINEL', 'PRIVATE_LABEL_METADATA', 'BlineJumping_C'):
                    self.assertNotIn(private, serialized)
                self.assertIsNone(interface.start_scenario())

    def test_failed_historical_audit_blocks_preparation_and_active_capture(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            captures, requests = fixture(root)
            summary = json.loads((captures / 'summary.json').read_text())
            summary['episodes'][0]['historical_action_trace_matches'] = False
            save(captures / 'summary.json', summary)
            result = run_prepare(root, captures, requests)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Historical episode verification failed', result.stderr)
            result = subprocess.run([
                sys.executable, str(ROOT / 'scripts/capture_active_defender.py'),
                '--monitor-capture-dir', str(captures), '--cage-root', str(root / 'no-simulator'),
                '--checkpoint-template', str(root / 'seed{seed}.zip'), '--output-dir', str(root / 'active'),
            ], text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Historical episode verification failed', result.stderr)
            self.assertFalse((root / 'active').exists())
            self.assertFalse((root / 'prepared').exists())

    def test_active_capture_rejects_different_simulator_before_loading_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            captures, _ = fixture(root)
            package = root / 'simulator/CybORG'
            package.mkdir(parents=True)
            (package / '__init__.py').write_text('')
            result = subprocess.run([
                sys.executable, str(ROOT / 'scripts/capture_active_defender.py'),
                '--monitor-capture-dir', str(captures), '--cage-root', str(package.parent),
                '--checkpoint-template', str(root / 'seed{seed}.zip'), '--output-dir', str(root / 'active'),
            ], text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Simulator source differs', result.stderr)
            self.assertFalse((root / 'active').exists())

    def test_changed_evidence_and_incomplete_windows_are_rejected(self):
        for failure in ('hash', 'window', 'action'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                captures, requests = fixture(root)
                if failure == 'hash':
                    with (captures / 'episode.json').open('a') as stream:
                        stream.write(' ')
                else:
                    item = json.loads(requests.read_text())
                    if failure == 'window':
                        item['provenance']['window_end'] = 4
                    else:
                        item['messages'][-1]['content'] = item['messages'][-1]['content'].replace('PRIVATE_TARGET', 'WRONG_TARGET')
                    requests.write_text(json.dumps(item) + '\n')
                result = run_prepare(root, captures, requests)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((root / 'prepared').exists())

    def test_historical_actions_require_explicit_field_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            action = {'Action': 'Monitor', 'Status': 'TRUE'}
            save(root / 'old.json', {'blue_actions': [action], 'red_actions': [action],
                                    'red_agent_name': 'BlineJumping_C', 'network_topology': {}})
            interface = CAGEActionBasedServiceInterface(n_rollouts=1, episode_length=1, trajectory_dir=str(root))
            with self.assertRaisesRegex(ValueError, 'Missing recorded observations'):
                interface.start_scenario()
            explicit = CAGEActionBasedServiceInterface(
                n_rollouts=1, episode_length=1, trajectory_dir=str(root),
                observation_fields=('blue_actions', 'red_actions'),
            )
            self.assertEqual(explicit.start_scenario().get_state().unstructured,
                             'Step 0:\n  Blue: Monitor (success: TRUE)\n  Red: Monitor (success: TRUE)')


if __name__ == '__main__':
    unittest.main()
