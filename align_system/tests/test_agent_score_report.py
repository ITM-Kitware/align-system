"""Score reports must distinguish missing scores from sessions that never score."""
import json

import pytest

from align_system.cli import report_agent_scores as cli


def write_status(tmp_path, records):
    (tmp_path / 'scoring_status.json').write_text(json.dumps(records))
    return tmp_path


def record(scenario_id, status, final_state=None, api=None, **fields):
    return {'scenario_id': scenario_id, 'alignment_target_id': 'T1', 'status': status,
            'completed': status in ('available', 'unavailable'),
            'final_state': final_state, 'api': api or {'status': 'skipped', 'training_session': None},
            **fields}


SKIPPED = {'status': 'skipped', 'reason': 'TA3 score API is only called for full training sessions.',
           'training_session': None}
AVAILABLE = record('S1', 'available', {
    'status': 'available', 'score': 0.75, 'raw_value': '0.75',
    'text': 'Scenario S1 complete for target T1. Session alignment score = 0.75'}, SKIPPED)
TEST_SESSION = record('S2', 'unavailable', {
    'status': 'missing', 'score': None, 'text': 'Test scenario S2 complete.'}, SKIPPED)
TA1_ERROR = record('S3', 'unavailable', {
    'status': 'unavailable', 'score': None, 'raw_value': 'Error',
    'text': 'Scenario S3 complete for target T1. Session alignment score = Error'}, SKIPPED)
PENDING = record('S4', 'pending')
API_SCORE = record('S5', 'available', {'status': 'missing', 'score': None, 'text': 'Scenario S5 complete.'},
                   {'status': 'available', 'result': {'score': 0.4, 'kdma_values': []}})


def test_report_lists_scores_and_reasons(tmp_path, capsys):
    run = write_status(tmp_path, [AVAILABLE, TEST_SESSION, TA1_ERROR, PENDING, API_SCORE])
    assert cli.main([str(run)]) == 0
    out = capsys.readouterr().out
    assert 'S1 target=T1: score=0.75 (final_state)' in out
    assert 'S5 target=T1: score=0.4 (api)' in out
    assert 'S2' in out and 'test session' in out
    assert 'S3' in out and 'TA1' in out
    assert 'S4' in out and 'never finished' in out
    assert '2 of 5 scenarios scored' in out


@pytest.mark.parametrize('records,expected', [
    ([AVAILABLE, API_SCORE], 0),
    ([AVAILABLE, TEST_SESSION], 1),
    ([], 1),
])
def test_require_fails_when_any_scenario_lacks_a_score(tmp_path, capsys, records, expected):
    assert cli.main([str(write_status(tmp_path, records)), '--require']) == expected
    assert ('scores are missing' in capsys.readouterr().err) == bool(expected)


def test_accepts_status_file_path_and_rejects_missing_file(tmp_path):
    run = write_status(tmp_path, [AVAILABLE])
    assert cli.main([str(run / 'scoring_status.json')]) == 0
    with pytest.raises(SystemExit):
        cli.main([str(tmp_path / 'absent')])
