"""Replay the April stall offline and check recovery never fabricates a score."""
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage
from swagger_client.models import ActionTypeEnum

from align_system.drivers.itm_open_world_langchain import (
    ITMOpenWorldLangChainDriver, ScenarioSession)
from align_system.exceptions import ScenarioStalledError
from align_system.tests.test_agent_assessments import Model, Scenario, action
from align_system.tests.test_agent_controls_and_events import Actor
from align_system.tests.test_agent_scoring import drive
from align_system.utils.agent_events import AgentEventLog
from align_system.utils.agent_scoring import AgentScoreRecorder


def model(value):
    if isinstance(value, dict):
        return Model(**{key: model(item) for key, item in value.items()})
    if isinstance(value, list):
        return [model(item) for item in value]
    return value


def session_for(scenario):
    return ScenarioSession(scenario, None, True, False, lambda record: None)


def replay_session():
    fixture = json.loads((Path(__file__).parent / 'fixtures' /
                          'agent_stall_3949.json').read_text())
    choices = [action(item['action_id'], **{
        key: value for key, value in item.items() if key != 'action_id'})
        for item in fixture['choices']]
    scenario = Scenario(choices)
    scenario.state = model(fixture['state'])
    for patient in scenario.state.characters:
        # The generated API model supplies absent optional attributes as None.
        patient.tag = getattr(patient, 'tag', None)
    session = session_for(scenario)
    for name, values in fixture['bookkeeping'].items():
        getattr(session, name).update(values)
    return session


def test_recorded_3949_stall_moves_to_unseen_patient_with_valid_id():
    session = replay_session()
    selected = ITMOpenWorldLangChainDriver()._take_fallback_action(session)
    assert selected.action_type == ActionTypeEnum.MOVE_TO
    assert selected.character_id in {'Patient 11', 'Patient 12'}
    assert session.scenario.executed[0].character_id == selected.character_id
    assert selected.parameters == {}


def test_unseen_navigation_does_not_expose_hidden_clinical_fields():
    session = replay_session()
    hidden = next(c for c in session.current_state.characters if c.unseen)
    hidden.injuries = [Model(name='Hidden injury', severity='substantial')]
    hidden.vitals = Model(heart_rate=171)
    hidden.unstructured = 'HIDDEN CLINICAL DESCRIPTION'
    observation = json.loads(ITMOpenWorldLangChainDriver()._observation_text(
        session.current_state))
    assert hidden.id not in {c['id'] for c in observation['casualties']}
    navigation = next(c for c in observation['navigation_targets']
                      if c['id'] == hidden.id)
    assert navigation == {'id': hidden.id, 'name': hidden.name,
                          'unseen': True, 'nearby': False}
    assert 'HIDDEN CLINICAL DESCRIPTION' not in json.dumps(observation)
    assert 'Hidden injury' not in json.dumps(observation)
    assert '171' not in json.dumps(observation)


def test_distant_clinical_target_requires_navigation_before_fallback_treatment():
    scenario = Scenario([
        action('treat', ActionTypeEnum.TREAT_PATIENT,
               parameters={'treatment': 'Tourniquet', 'location': 'left calf'}),
        action('move', ActionTypeEnum.MOVE_TO),
    ])
    scenario.state.characters[0].nearby = False
    scenario.state.supplies = [Model(type='Tourniquet', quantity=1)]
    selected = ITMOpenWorldLangChainDriver()._take_fallback_action(session_for(scenario))
    assert selected.action_type == ActionTypeEnum.MOVE_TO
    assert selected.character_id == 'p1'
    assert all(a.action_type != ActionTypeEnum.TREAT_PATIENT for a in scenario.executed)


def test_rejected_navigation_tries_another_concrete_patient():
    session = replay_session()
    attempted = []

    class Rejected(Exception):
        status = 400
        body = 'Patient location is inaccessible.'

    def take_action(selected):
        attempted.append(deepcopy(selected))
        if len(attempted) == 1:
            raise Rejected()
        return deepcopy(session.current_state)

    session.scenario.take_action = take_action
    selected = ITMOpenWorldLangChainDriver()._take_fallback_action(session)
    assert len(attempted) == 2
    assert all(a.action_type == ActionTypeEnum.MOVE_TO for a in attempted)
    assert {a.character_id for a in attempted} == {'Patient 11', 'Patient 12'}
    assert selected.character_id == attempted[-1].character_id


def test_successful_treatment_allows_vitals_reassessment():
    scenario = Scenario([action('check', ActionTypeEnum.CHECK_VITALS),
                         action('treat', ActionTypeEnum.TREAT_PATIENT)])
    scenario.state.characters[0].nearby = True
    session = session_for(scenario)
    session.execute(action('check', ActionTypeEnum.CHECK_VITALS, character_id='p1'))
    assert 'p1' in session.vitals_checked
    session.execute(action('treat', ActionTypeEnum.TREAT_PATIENT, character_id='p1',
                           parameters={'treatment': 'Tourniquet',
                                       'location': 'left calf'}))
    assert 'p1' not in session.vitals_checked
    assert any(a.action_type == ActionTypeEnum.CHECK_VITALS
               for a in session.refresh_actions())


@pytest.mark.parametrize('change, should_reassess', [
    ('unchanged', False), ('proximity', False), ('new_injury', True),
    ('vitals', True),
])
def test_rechecks_follow_observed_clinical_changes(change, should_reassess):
    scenario = Scenario([action('check', ActionTypeEnum.CHECK_VITALS), action('search')])
    patient = scenario.state.characters[0]
    patient.nearby = True
    patient.injuries = [Model(name='Laceration', location='left calf', status='visible')]
    patient.vitals = Model(heart_rate=90)
    session = session_for(scenario)
    session.execute(action('check', ActionTypeEnum.CHECK_VITALS, character_id='p1'))
    updated = deepcopy(session.current_state)
    if change == 'proximity':
        updated.characters[0].nearby = False
        updated.characters[0].vitals = Model()
        updated.characters[0].injuries = None
    elif change == 'new_injury':
        updated.characters[0].injuries.append(
            Model(name='Puncture', location='right chest', status='visible'))
    elif change == 'vitals':
        updated.characters[0].vitals.heart_rate = 140
    scenario.take_action = lambda selected: deepcopy(updated)
    session.execute(action('search'))
    assert ('p1' not in session.vitals_checked) == should_reassess


def test_exhausted_recovery_preserves_stall_diagnostics_without_score(tmp_path):
    scenario = Scenario([action('search')])

    class Rejected(Exception):
        status = 400
        body = 'Search is unavailable.'

    scenario.take_action = Mock(side_effect=Rejected())
    interface = Model(get_session_alignment=Mock())
    session = session_for(scenario)
    with AgentEventLog(tmp_path / 'agent_events.jsonl') as events:
        recorder = AgentScoreRecorder(tmp_path, interface, events)
        with pytest.raises(ScenarioStalledError) as caught:
            with events.episode('scenario'), recorder.episode('scenario'):
                ITMOpenWorldLangChainDriver()._take_fallback_action(session)
    assert caught.value.diagnostics
    assert scenario.take_action.call_count == 1
    status, = json.loads((tmp_path / 'scoring_status.json').read_text())
    assert status['status'] == 'stalled'
    assert status['completed'] is False
    assert status['error_type'] == 'ScenarioStalledError'
    assert status['stall'] == caught.value.diagnostics
    assert 'final_state' not in status and 'api' not in status
    assert json.loads((tmp_path / 'scores.json').read_text()) == []
    interface.get_session_alignment.assert_not_called()


def test_stall_error_and_scoring_diagnostics_preserve_original_reason(tmp_path):
    interface = Model(get_session_alignment=Mock())
    diagnostics = {'scene_id': 'building_explosion', 'rejections': [
        {'action_type': 'MOVE_TO', 'character_id': 'Patient 11',
         'reason': 'Location unavailable'}]}
    error = ScenarioStalledError('Every safe recovery action was rejected', diagnostics)
    with AgentEventLog(tmp_path / 'agent_events.jsonl') as events:
        recorder = AgentScoreRecorder(tmp_path, interface, events)
        with pytest.raises(ScenarioStalledError) as caught:
            with events.episode('scenario'), recorder.episode('scenario'):
                raise error
    assert caught.value is error
    status, = json.loads((tmp_path / 'scoring_status.json').read_text())
    assert status['stall'] == diagnostics
    assert status['status'] == 'stalled'
    assert status['error'] == str(error)
    assert json.loads((tmp_path / 'scores.json').read_text()) == []
    interface.get_session_alignment.assert_not_called()


def test_unrecoverable_live_session_is_not_silently_advanced(tmp_path, monkeypatch):
    scenario = Scenario([action('search')])
    scenario.get_alignment_target = lambda: None

    class Rejected(Exception):
        status = 400
        body = 'Search is unavailable.'

    scenario.take_action = Mock(side_effect=Rejected())
    interface = Model(start_scenario=Mock(side_effect=[scenario, None]),
                      get_session_alignment=Mock())
    driver = ITMOpenWorldLangChainDriver(
        chat_model=Actor([AIMessage(content='No viable action')]),
        max_llm_calls_between_actions=1)
    with pytest.raises(ScenarioStalledError):
        drive(tmp_path, monkeypatch, interface, driver)
    interface.start_scenario.assert_called_once()
    interface.get_session_alignment.assert_not_called()
    status, = json.loads((tmp_path / 'scoring_status.json').read_text())
    assert status['status'] == 'stalled' and not status['completed']
    saved = json.loads((tmp_path / f"{status['episode_id']}.stall.json").read_text())
    assert saved == status['stall']
    assert saved['full_state']['meta_info']['scene_id'] == 'scene'
    assert json.loads((tmp_path / 'scores.json').read_text()) == []
