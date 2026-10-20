"""Recorded reset ambiguity and generic navigation guards, without live inference."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from align_system.drivers.itm_open_world_langchain import (
    ITMOpenWorldLangChainDriver, ScenarioSession)
from align_system.tests.test_agent_assessments import Model, Scenario, action
from align_system.tests.test_agent_controls_and_events import Actor
from align_system.tests.test_agent_patient_progress import recorded_state
from align_system.cli.report_reset_progress import episode_metrics, report_run
from align_system.utils.agent_events import AgentEventLog


def recorded_session():
    fixture = json.loads((Path(__file__).parent / 'fixtures' / 'agent_reset_3965.json').read_text())
    scenario = Scenario([action(**a) for a in fixture['choices']])
    scenario.state = recorded_state(fixture['steps'][0])
    session = ScenarioSession(scenario, None, True, False, lambda record: None,
                              reset_awareness=True)
    return session, fixture['steps']


def advance(session, steps, start, end):
    for index in range(start, end):
        following = recorded_state(steps[index + 1])
        session.scenario.take_action = lambda selected: deepcopy(following)
        session.refresh_actions()
        session.execute(action(**steps[index]['action']))


def test_3965_reset_is_explicit_even_though_observations_still_say_treated():
    session, steps = recorded_session()
    advance(session, steps, 0, 17)
    driver = ITMOpenWorldLangChainDriver()
    observation = json.loads(driver._observation_text(session.current_state, session))
    progress = {p['patient_id']: p for p in observation['patient_progress']}
    assert not session.treated_patients
    for patient_id in ('Patient 5', 'Patient 9'):
        patient = next(c for c in observation['casualties'] if c['id'] == patient_id)
        assert 'treated' in patient['unstructured']
        assert progress[patient_id]['previously_treated']
        assert not progress[patient_id]['currently_treated']
        assert progress[patient_id]['last_treatment']['scene_id'] == 'treat_and_tag'
        assert progress[patient_id]['treatment_credit_reset']['reason'] == 'scene_change'
    assert progress['Patient 5']['tagged_current_scene']
    assert not progress['Patient 9']['tagged_current_scene']
    assert progress['Patient 9']['last_known_tag']['tag'] == 'IMMEDIATE'
    assert 'prior care no longer counts' in session.patient_progress.context(session).replace('\n', ' ')
    # Replaying the remaining original decisions is evidence of the deficit,
    # not a claim about how a new model would react to this observation.
    advance(session, steps, 17, 100)
    assert session.pending_patients() == {'Patient 5', 'Patient 9'}
    assert len(session.treated_patients) == 10


def navigation_session(enabled=True):
    scenario = Scenario([action('move', 'MOVE_TO'), action('treat', 'TREAT_PATIENT'),
                         action('tag', 'TAG_CHARACTER'), action('check', 'CHECK_VITALS')])
    scenario.state.characters = [Model(id=p, name=p, nearby=False, unseen=False,
                                      tag=None, vitals=Model(heart_rate=90))
                                 for p in ('alpha', 'beta')]
    session = ScenarioSession(scenario, None, True, False, lambda record: None,
                              reset_awareness=enabled, navigation_safeguards=enabled)

    def take_action(selected):
        state = deepcopy(session.current_state)
        if selected.action_type == 'MOVE_TO':
            for patient in state.characters:
                patient.nearby = patient.id == selected.character_id
        return state

    scenario.take_action = take_action
    driver = ITMOpenWorldLangChainDriver()
    return session, driver, {t.name: t for t in driver._build_tools(session)}


def move(tools, patient):
    return tools['move_to'].invoke({'character_name': patient, 'justification': 'Reach pending care'})


@pytest.mark.parametrize('enabled', [False, True])
def test_navigation_cycle_requires_a_pending_action_plan_only_when_enabled(enabled):
    session, driver, tools = navigation_session(enabled)
    assert move(tools, 'alpha').startswith('Action executed.')
    assert move(tools, 'beta').startswith('Action executed.')
    result = move(tools, 'alpha')
    if not enabled:
        assert result.startswith('Action executed.')
        assert 'reassess_navigation' not in tools
        assert 'patient_progress' not in json.loads(tools['observe_environment'].invoke({}))
        return
    assert 'Reassess your plan' in result
    assert session.n_actions == 2
    # Exact-ID actions and fallback must obey the same guard.
    assert 'Reassess your plan' in tools['execute_available_action'].invoke({
        'action_id': 'move', 'character_name': 'alpha', 'justification': 'Repeat'})
    assert not tools['reassess_navigation'].invoke({
        'character_name': 'alpha', 'next_action': 'move_to', 'justification': 'Cycle'}).startswith('Plan reassessed')
    assert tools['reassess_navigation'].invoke({
        'character_name': 'alpha', 'next_action': 'treat_patient',
        'justification': 'The previous visit did not perform required treatment'}).startswith('Plan reassessed')
    assert move(tools, 'alpha').startswith('Action executed.')
    move(tools, 'beta')
    assert 'Reassess your plan' in move(tools, 'alpha')  # Permission consumed once.
    fallback = driver._take_fallback_action(session)
    assert fallback.action_type != 'MOVE_TO'


def test_completed_patient_does_not_need_a_move_but_a_reset_reopens_it():
    session, _, tools = navigation_session()
    session.execute(action('treat', 'TREAT_PATIENT', character_id='alpha'))
    session.execute(action('tag', 'TAG_CHARACTER', character_id='alpha',
                           parameters={'category': 'IMMEDIATE'}))
    assert 'no pending offered care action' in move(tools, 'alpha')
    after = deepcopy(session.current_state)
    after.meta_info.scene_id = 'arbitrary-new-scene'
    session.scenario.take_action = lambda selected: deepcopy(after)
    session.execute(action('transition'))
    assert move(tools, 'alpha').startswith('Action executed.')


def test_clinical_change_reopens_treatment_without_scene_change_and_hidden_facts_do_not_leak():
    session, _, tools = navigation_session()
    session.execute(action('treat', 'TREAT_PATIENT', character_id='alpha'))
    changed = deepcopy(session.current_state)
    changed.characters[0].vitals.heart_rate = 140
    changed.characters[1].unseen = True
    changed.characters[1].vitals = Model(secret='NEVER DISCLOSED')
    session.scenario.take_action = lambda selected: deepcopy(changed)
    session.execute(action('search'))
    observation = tools['observe_environment'].invoke({})
    assert 'NEVER DISCLOSED' not in observation
    progress = json.loads(observation)['patient_progress'][0]
    assert not progress['currently_treated']
    assert progress['treatment_credit_reset']['reason'] == 'clinical_change'
    session.execute(action('treat', 'TREAT_PATIENT', character_id='alpha'))
    progress = session.patient_progress.snapshot(session)[0]
    assert progress['currently_treated'] and progress['treatment_credit_reset'] is None


def test_treatment_triggering_transition_is_history_only():
    session, _, _ = navigation_session()
    changed = deepcopy(session.current_state)
    changed.meta_info.scene_id = 'new-scene'
    session.scenario.take_action = lambda selected: deepcopy(changed)
    session.execute(action('treat', 'TREAT_PATIENT', character_id='alpha'))
    patient = session.patient_progress.snapshot(session)[0]
    assert patient['previously_treated'] and not patient['currently_treated']
    assert patient['last_treatment']['scene_id'] == 'scene'
    assert patient['treatment_credit_reset']['scene_id'] == 'new-scene'


def test_ledger_is_in_every_model_request_when_history_is_trimmed():
    session, _, _ = navigation_session()
    actor = Actor([AIMessage(content='', tool_calls=[{
        'id': patient, 'name': 'move_to',
        'args': {'character_name': patient, 'justification': 'Treat pending patient'}}])
                   for patient in ('alpha', 'beta')])
    driver = ITMOpenWorldLangChainDriver(chat_model=actor, max_messages_in_context=1,
                                        max_actions_per_scenario=2)
    driver._run_agent_loop(session)
    assert len(actor.messages) == 2
    for messages in actor.messages:
        assert 'CURRENT CARE CREDIT' in messages[0].content
        assert '"currently_treated": false' in messages[0].content


def test_recorded_metrics_do_not_count_pre_reset_treatment_as_current_coverage():
    session, steps = recorded_session()
    records = [{'input': {'full_state': recorded_state(step).to_dict(),
                          'choices': [a.to_dict() for a in session.scenario.actions]},
                'output': {'action': step['action']}}
               for step in steps[:-1]]
    metrics = episode_metrics(records)
    assert metrics['total_actions'] == 100
    assert metrics['movement_actions'] == 47
    assert metrics['longest_movement_sequence'] == 16
    assert metrics['revisits_without_patient_action'] == 28
    coverage = metrics['treatment_coverage_by_scene'][-1]
    assert coverage['after_scene_reset']
    assert coverage['treated'] == 10 and coverage['patients'] == 12
    assert coverage['without_treatment'] == ['Patient 5', 'Patient 9']


def test_metrics_include_zero_action_failures_and_flag_missing_trace_actions(tmp_path):
    with AgentEventLog(tmp_path / 'agent_events.jsonl') as events:
        with events.episode('failed-before-action'):
            events.emit('episode_finished', completed=False)
        with events.episode('completed-with-missing-trace-row'):
            events.emit('action_succeeded', decision_index=0, action=action('search').to_dict())
            events.emit('episode_finished', completed=True)
    report = report_run(tmp_path)
    assert report['episodes_started'] == 2
    assert report['completion_rate'] == 0.5
    assert not report['trace_metrics_complete']


def test_evacuation_only_scene_does_not_require_a_care_navigation_plan():
    session, _, tools = navigation_session()
    session.scenario.actions = [action('move', 'MOVE_TO'), action('evac', 'MOVE_TO_EVAC')]
    move(tools, 'alpha')
    move(tools, 'beta')
    assert move(tools, 'alpha').startswith('Action executed.')


def test_disclosed_clinical_change_allows_new_visit_without_reassessment():
    session, _, tools = navigation_session()
    move(tools, 'alpha')
    move(tools, 'beta')
    changed = deepcopy(session.current_state)
    changed.characters[0].vitals.heart_rate = 140
    session.scenario.take_action = lambda selected: deepcopy(changed)
    session.execute(action('search'))
    assert move(tools, 'alpha').startswith('Action executed.')
