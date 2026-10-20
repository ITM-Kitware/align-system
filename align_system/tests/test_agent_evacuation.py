"""Replay the evacuation stall without changing any live server session."""
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import Mock

from langchain_core.messages import AIMessage
import pytest
from swagger_client.models import ActionTypeEnum

from align_system.agent_tools.concrete_actions import concrete_actions
from align_system.drivers.itm_open_world_langchain import (
    ITMOpenWorldLangChainDriver, ScenarioSession)
from align_system.exceptions import ScenarioStalledError
from align_system.tests.test_agent_assessments import Model, Scenario, action, action_call
from align_system.tests.test_agent_controls_and_events import Actor
from align_system.tests.test_agent_scoring import drive
from align_system.tests.test_agent_stall_recovery import model


def stalled_session(filtering=True):
    fixture = json.loads((Path(__file__).parent / 'fixtures' /
                          'agent_evac_stall_3964.json').read_text())
    scenario = Scenario([action(**a) for a in fixture['choices']])
    scenario.state = model(fixture['state'])
    for patient in scenario.state.characters:
        patient.tag = getattr(patient, 'tag', None)
    session = ScenarioSession(scenario, None, filtering, False, lambda record: None)
    session.evac_patients.update(fixture['bookkeeping']['evac_patients'])
    session.n_actions = fixture['decision_index']
    return session


@pytest.mark.parametrize('tool_name', ['move_to_evac', 'execute_available_action'])
@pytest.mark.parametrize('filtering', [True, False])
def test_recorded_unseen_patient_reaches_server_through_both_tools(tool_name, filtering):
    session = stalled_session(filtering)
    driver = ITMOpenWorldLangChainDriver(apply_action_filtering=filtering)
    assert all(c.unseen and c.nearby is False for c in session.current_state.characters)
    offered = session.refresh_actions()
    candidates, omitted = concrete_actions(session.current_state, offered)
    assert not omitted
    assert {a.character_id for a in candidates} == {
        f'Patient {i}' for i in range(1, 11)}
    tools = {t.name: t for t in driver._build_tools(session)}
    args = dict(character_name='Patient 1', justification='Select a listed casualty')
    if tool_name == 'execute_available_action':
        args['action_id'] = 'evac_patient'
    tools[tool_name].invoke(args)
    selected, = session.scenario.executed
    assert selected.action_type == ActionTypeEnum.MOVE_TO_EVAC
    assert selected.action_id == 'evac_patient'
    assert selected.character_id == 'Patient 1'
    assert session.n_actions == 61
    assert 'Patient 1' in session.evac_patients
    # A successful submission alone must not manufacture scenario completion.
    assert not session.scenario_complete


def test_evacuation_context_uses_offered_actions_and_only_discloses_identity():
    session = stalled_session()
    session.current_state.meta_info.scene_id = 'arbitrary_scene_name'
    hidden = session.current_state.characters[0]
    hidden.unstructured = 'SECRET CLINICAL DESCRIPTION'
    hidden.injuries = [Model(name='SECRET INJURY')]
    hidden.vitals = Model(heart_rate='SECRET VITALS')
    session.refresh_actions()
    driver = ITMOpenWorldLangChainDriver()
    context = driver._progress_context(session)
    assert 'Evacuation candidates:' in context
    assert all(f'Patient {i} [' in context for i in range(1, 11))
    assert 'Casualties to reach:' not in context
    assert 'unseen or distant' in context
    observation = json.loads(driver._observation_text(session.current_state))
    assert observation['casualties'] == []
    assert 'SECRET' not in context + json.dumps(observation)


def test_targeted_evacuation_keeps_server_target_and_duplicate_guards():
    session = stalled_session()
    session.scenario.actions[0].character_id = 'Patient 1'
    driver = ITMOpenWorldLangChainDriver()
    session.refresh_actions()
    assert 'Patient 2 [' not in driver._progress_context(session)
    tools = {t.name: t for t in driver._build_tools(session)}
    for name in ['Patient 2', 'Invented patient']:
        tools['move_to_evac'].invoke(dict(character_name=name, justification='Attempt'))
    assert session.scenario.executed == []
    tools['move_to_evac'].invoke(dict(character_name='Patient 1', justification='Select'))
    assert len(session.scenario.executed) == 1
    assert session.refresh_actions() == []
    tools['move_to_evac'].invoke(dict(character_name='Patient 1', justification='Repeat'))
    assert len(session.scenario.executed) == 1


def test_unseen_clinical_actions_stay_unavailable_and_evacuation_must_be_offered():
    session = stalled_session()
    session.scenario.actions = [action(kind, kind) for kind in [
        'CHECK_VITALS', 'TREAT_PATIENT', 'TAG_CHARACTER']]
    assert session.refresh_actions() == []
    tools = {t.name: t for t in ITMOpenWorldLangChainDriver()._build_tools(session)}
    tools['move_to_evac'].invoke(dict(character_name='Patient 1', justification='Attempt'))
    assert session.scenario.executed == []


@pytest.mark.parametrize('reverse_order', [False, True])
def test_fallback_leaves_multiple_evacuation_choices_unresolved(reverse_order):
    session = stalled_session()
    session.current_state.meta_info.scene_id = 'arbitrary_scene'
    session.scenario.actions[0].action_id = 'transport-option'
    for i, patient in enumerate(session.current_state.characters):
        patient.id = f'casualty-{i * 7}'
        patient.name = f'Person {i}'
    if reverse_order:
        session.current_state.characters.reverse()
    # Even another offered action must not hide the unresolved evacuation.
    session.scenario.actions.append(action('search'))
    session.scenario.take_action = Mock()
    with pytest.raises(ScenarioStalledError) as caught:
        ITMOpenWorldLangChainDriver()._take_fallback_action(session)
    session.scenario.take_action.assert_not_called()
    diagnostics = caught.value.diagnostics
    assert diagnostics['reason'] == 'evacuation_choice_required'
    assert {a['character_id'] for a in diagnostics['evacuation_candidates']} == {
        patient.id for patient in session.current_state.characters}
    assert diagnostics['rejected'] == []
    assert session.n_actions == 60
    assert not session.scenario_complete


@pytest.mark.parametrize('rejected', [False, True])
def test_single_evacuation_fallback_preserves_server_rejection_handling(rejected):
    session = stalled_session()
    session.scenario.actions[0].character_id = 'Patient 1'
    attempted = []

    class Rejected(Exception):
        status = 400
        body = 'Evacuation rejected by server.'

    def take_action(selected):
        attempted.append(deepcopy(selected))
        if rejected:
            raise Rejected()
        return deepcopy(session.current_state)

    session.scenario.take_action = take_action
    driver = ITMOpenWorldLangChainDriver()
    if rejected:
        with pytest.raises(ScenarioStalledError) as caught:
            driver._take_fallback_action(session)
        assert len(caught.value.diagnostics['rejected']) == 1
        assert session.n_actions == 60
        assert session.evac_patients == {'Patient 11', 'Patient 12'}
    else:
        selected = driver._take_fallback_action(session)
        assert selected.character_id == attempted[-1].character_id
        assert session.n_actions == 61
    assert len(attempted) == 1
    assert attempted[0].character_id == 'Patient 1'
    assert all(a.action_type == ActionTypeEnum.MOVE_TO_EVAC for a in attempted)
    assert len({a.character_id for a in attempted}) == len(attempted)
    assert not session.scenario_complete


def test_unresolved_evacuation_saves_diagnostics_without_score_or_next_scenario(tmp_path, monkeypatch):
    scenario = stalled_session().scenario
    scenario.get_alignment_target = lambda: None
    scenario.take_action = Mock()
    interface = Model(start_scenario=Mock(side_effect=[scenario, None]),
                      get_session_alignment=Mock())
    driver = ITMOpenWorldLangChainDriver(
        chat_model=Actor([AIMessage(content='Still deciding')]),
        max_llm_calls_between_actions=1)
    with pytest.raises(ScenarioStalledError, match='did not choose'):
        drive(tmp_path, monkeypatch, interface, driver)
    scenario.take_action.assert_not_called()
    interface.start_scenario.assert_called_once()
    interface.get_session_alignment.assert_not_called()
    status, = json.loads((tmp_path / 'scoring_status.json').read_text())
    assert status['status'] == 'stalled' and not status['completed']
    assert status['stall']['reason'] == 'evacuation_choice_required'
    assert len(status['stall']['evacuation_candidates']) == 10
    assert json.loads((tmp_path / f"{status['episode_id']}.stall.json").read_text()) == status['stall']
    assert json.loads((tmp_path / 'scores.json').read_text()) == []


def test_recorded_stall_agent_loop_finishes_only_on_mock_server_completion():
    session = stalled_session()

    def take_action(selected):
        assert selected.character_id == 'Patient 1'
        following = deepcopy(session.current_state)
        # Synthetic response: verifies driver handling, not live API acceptance.
        following.scenario_complete = True
        session.scenario.executed.append(deepcopy(selected))
        return following

    session.scenario.take_action = take_action
    actor = Actor([AIMessage(content='', tool_calls=[
        action_call('evac_patient', character_name='Patient 1')])])
    driver = ITMOpenWorldLangChainDriver(chat_model=actor, max_actions_per_scenario=61)
    driver._run_agent_loop(session)
    assert session.scenario_complete and session.n_actions == 61
    assert len(session.scenario.executed) == 1
    assert 'Evacuation candidates:' in actor.messages[0][0].content
