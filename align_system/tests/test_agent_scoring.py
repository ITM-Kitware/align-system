"""Offline scoring persistence, availability, and failure contracts."""
from copy import deepcopy
import json
from unittest.mock import Mock

from omegaconf import OmegaConf
import pytest

from align_system.drivers.itm_open_world_langchain import ITMOpenWorldLangChainDriver
from align_system.exceptions import ScenarioActionLimitError
from align_system.interfaces.ta3_caci_action_based_service import (
    TA3CACIActionBasedServiceInterface)
from align_system.tests.test_agent_assessments import Model, Scenario, action
from align_system.tests.test_agent_controls_and_events import Actor
from align_system.utils.agent_events import AgentEventLog
from align_system.utils.agent_scoring import AgentScoreRecorder, final_state_score


def read_json(path, name):
    return json.loads((path / name).read_text())


def read_events(path):
    return [json.loads(line) for line in
            (path / 'agent_events.jsonl').read_text().splitlines()]


def make_scenario(text='Scenario complete.', complete=True, target=True):
    scenario = Scenario([action('a')])
    scenario.get_alignment_target = lambda: (
        Model(id='target', kdma_values=[{'kdma': 'affiliation', 'value': 0.8}])
        if target else None)

    def take_action(selected):
        state = deepcopy(scenario.state)
        state.scenario_complete = complete
        state.unstructured = text
        return state

    scenario.take_action = take_action
    return scenario


def drive(tmp_path, monkeypatch, interface, driver=None, enabled=True):
    cfg = OmegaConf.create(dict(
        interface=interface, align_to_target=True, save_input_output=True,
        save_scoring_output=enabled, save_alignment_targets=False, save_timing=False),
        flags={'allow_objects': True})
    monkeypatch.setattr(
        'hydra.core.hydra_config.HydraConfig.get',
        lambda: Model(runtime=Model(output_dir=str(tmp_path))))
    driver = driver if driver is not None else ITMOpenWorldLangChainDriver(chat_model=Actor())
    driver.drive(cfg)


@pytest.mark.parametrize('response', [
    {'scenario': {'accuracy': {'value': 0.75, 'description': 'Replay accuracy'}}},
    Model(alignment_target_id='target', score=0.75, kdma_values=[]),
])
def test_api_shape_and_scores_survive_next_scenario_failure(tmp_path, monkeypatch, response):
    expected = response if isinstance(response, dict) else response.to_dict()
    scenario = make_scenario()

    def start_scenario():
        if interface.start_scenario.call_count == 1:
            return scenario
        # Files must already exist when the next scenario is requested.
        assert read_json(tmp_path, 'scores.json') == [expected]
        status, = read_json(tmp_path, 'scoring_status.json')
        assert status['status'] == 'available' and status['completed']
        assert status['alignment_target_id'] == 'target'
        raise RuntimeError('next scenario failed')

    interface = Model(start_scenario=Mock(side_effect=start_scenario),
                      get_session_alignment=Mock(return_value=response))
    with pytest.raises(RuntimeError, match='next scenario failed'):
        drive(tmp_path, monkeypatch, interface)
    assert read_json(tmp_path, 'scores.json') == [expected]
    interface.get_session_alignment.assert_called_once()
    events = read_events(tmp_path)
    scoring, = [event for event in events if event['event'] == 'alignment_scoring']
    assert scoring['api']['result'] == expected
    record, = read_json(tmp_path, 'scoring_status.json')
    action_record, = read_json(tmp_path, 'input_output.json')
    for key in ('run_id', 'episode_id'):
        assert record[key] == scoring[key] == action_record['choice_info']['event_log'][key]
    assert events[-1]['event'] == 'run_aborted'


@pytest.mark.parametrize('text, status, score', [
    ('Complete.\nSession alignment score = 0\nDone.', 'available', 0.0),
    ('Session alignment score = 0.625', 'available', 0.625),
    ('Session alignment score = N/A', 'unavailable', None),
    ('Session alignment score = None', 'unavailable', None),
    ('Session alignment score = NaN', 'unavailable', None),
    ('Session alignment score = inf', 'unavailable', None),
    ('Session alignment score = unknown value', 'unavailable', None),
    ('Scenario complete.', 'missing', None),
    (None, 'missing', None),
])
def test_final_state_score_preserves_missing_and_unavailable(text, status, score):
    result = final_state_score(text)
    assert result['status'] == status
    assert result['score'] == score
    assert result['text'] == text


@pytest.mark.parametrize('text, status', [
    ('Session alignment score = 0', 'available'),
    ('Scenario complete.', 'unavailable'),
])
def test_live_eval_skips_api_and_keeps_text_scores_separate(
        tmp_path, monkeypatch, text, status):
    # Bypass the constructor: even starting a session must stay offline.
    interface = TA3CACIActionBasedServiceInterface.__new__(TA3CACIActionBasedServiceInterface)
    interface.training_session = None
    interface.session_id = 'test-session'
    interface.connection = Mock()
    interface.start_scenario = Mock(side_effect=[make_scenario(text), None])
    drive(tmp_path, monkeypatch, interface)
    interface.connection.get_session_alignment.assert_not_called()
    assert read_json(tmp_path, 'scores.json') == []
    record, = read_json(tmp_path, 'scoring_status.json')
    assert record['status'] == status and record['completed']
    assert record['api']['status'] == 'skipped'
    assert record['api']['training_session'] is None
    assert record['session_id'] == 'test-session'
    if status == 'available':
        assert record['final_state']['score'] == 0


def test_score_api_failure_is_logged_without_aborting_run(tmp_path, monkeypatch, caplog):
    class ScoreUnavailable(RuntimeError):
        status = 503

    interface = Model(start_scenario=Mock(side_effect=[make_scenario(), None]),
                      get_session_alignment=Mock(side_effect=ScoreUnavailable('score service down')))
    drive(tmp_path, monkeypatch, interface)
    record, = read_json(tmp_path, 'scoring_status.json')
    assert record['status'] == 'unavailable'
    assert record['api']['status'] == 'error'
    assert record['api']['error_type'] == 'ScoreUnavailable'
    assert record['api']['error'] == 'score service down'
    assert record['api']['http_status'] == 503
    assert 'score service down' in caplog.text
    assert any(entry.exc_info for entry in caplog.records)
    assert read_json(tmp_path, 'scores.json') == []
    assert read_events(tmp_path)[-1]['event'] == 'run_finished'


@pytest.mark.parametrize('target, status', [(True, 'unavailable'), (False, 'skipped')])
def test_missing_api_response_or_target_is_explicit(tmp_path, monkeypatch, target, status):
    interface = Model(start_scenario=Mock(side_effect=[make_scenario(target=target), None]),
                      get_session_alignment=Mock(return_value=None))
    drive(tmp_path, monkeypatch, interface)
    record, = read_json(tmp_path, 'scoring_status.json')
    assert record['status'] == 'unavailable'
    assert record['api']['status'] == status
    assert read_json(tmp_path, 'scores.json') == []
    assert interface.get_session_alignment.call_count == int(target)


@pytest.mark.parametrize('abort', [False, True])
def test_unfinished_or_aborted_episode_does_not_request_scores(tmp_path, monkeypatch, abort):
    interface = Model(start_scenario=Mock(side_effect=[make_scenario(complete=False), None]),
                      get_session_alignment=Mock())
    actor = Actor([RuntimeError('actor down')] * 3) if abort else Actor()
    driver = ITMOpenWorldLangChainDriver(chat_model=actor, max_actions_per_scenario=1)
    if abort:
        with pytest.raises(RuntimeError, match='3 consecutive'):
            drive(tmp_path, monkeypatch, interface, driver)
    else:
        with pytest.raises(ScenarioActionLimitError):
            drive(tmp_path, monkeypatch, interface, driver)
        interface.start_scenario.assert_called_once()
    record, = read_json(tmp_path, 'scoring_status.json')
    assert record['status'] == ('aborted' if abort else 'incomplete')
    assert not record['completed']
    assert read_json(tmp_path, 'scores.json') == []
    interface.get_session_alignment.assert_not_called()
    if abort:
        assert record['error_type'] == 'RuntimeError'
        assert '3 consecutive' in record['error']


def test_pending_status_is_durable_before_episode_finishes(tmp_path):
    interface = Model(get_session_alignment=Mock())
    with AgentEventLog(tmp_path / 'agent_events.jsonl') as events:
        recorder = AgentScoreRecorder(tmp_path, interface, events)
        with events.episode('scenario'), recorder.episode('scenario') as record:
            assert read_json(tmp_path, 'scores.json') == []
            assert read_json(tmp_path, 'scoring_status.json') == [record]
            assert record['status'] == 'pending'
            assert record['episode_id'] == events.episode_id
    interface.get_session_alignment.assert_not_called()


def test_model_setup_failure_still_creates_empty_scoring_artifacts(tmp_path, monkeypatch):
    interface = Model(start_scenario=Mock())
    with pytest.raises(ValueError):
        drive(tmp_path, monkeypatch, interface, ITMOpenWorldLangChainDriver())
    assert read_json(tmp_path, 'scores.json') == []
    assert read_json(tmp_path, 'scoring_status.json') == []
    interface.start_scenario.assert_not_called()
    assert read_events(tmp_path)[-1]['event'] == 'run_aborted'


def test_disabled_saving_keeps_scoring_diagnostics_in_journal(tmp_path, monkeypatch):
    interface = Model(start_scenario=Mock(side_effect=[make_scenario(), None]),
                      get_session_alignment=Mock(return_value={'score': 0.5}))
    drive(tmp_path, monkeypatch, interface, enabled=False)
    assert not (tmp_path / 'scores.json').exists()
    assert not (tmp_path / 'scoring_status.json').exists()
    scoring, = [event for event in read_events(tmp_path)
                if event['event'] == 'alignment_scoring']
    assert scoring['status'] == 'available'
    assert scoring['api']['result'] == {'score': 0.5}
