"""Care survives partial observations within a scene, but resets between scenes."""
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage
from swagger_client.models import ActionTypeEnum

from align_system.drivers.itm_open_world_langchain import (
    ITMOpenWorldLangChainDriver, ScenarioSession)
from align_system.exceptions import ScenarioActionLimitError
from align_system.tests.test_agent_assessments import (
    Model, Scenario, action, action_call, session_for)
from align_system.tests.test_agent_controls_and_events import Actor
from align_system.tests.test_agent_scoring import drive, make_scenario
from align_system.tests.test_agent_stall_recovery import model


def care_actions():
    return [action('treat', ActionTypeEnum.TREAT_PATIENT),
            action('tag', ActionTypeEnum.TAG_CHARACTER),
            action('check', ActionTypeEnum.CHECK_VITALS),
            action('move', ActionTypeEnum.MOVE_TO),
            action('end', ActionTypeEnum.END_SCENE)]


def treat(session, patient='p1'):
    session.execute(action('treat', ActionTypeEnum.TREAT_PATIENT, character_id=patient))


def observe_after_action(session, state):
    session.scenario.take_action = lambda selected: deepcopy(state)
    session.execute(action('move', ActionTypeEnum.MOVE_TO, character_id='p1'))


def recorded_episode(run):
    return json.loads((Path(__file__).parent / 'fixtures' /
                       'agent_progress_3955_3963.json').read_text())['runs'][run]


def recorded_state(step, supplies=None):
    result = Model(unstructured='Recorded observation', scenario_complete=False,
                   meta_info=Model(scene_id=step['scene']),
                   characters=model(step['characters']), supplies=supplies)
    for patient in result.characters:
        patient.tag = getattr(patient, 'tag', None)
    return result


@pytest.mark.parametrize('unseen', [False, True])
def test_observed_tag_counts_for_completion_and_eligibility_after_visibility_loss(unseen):
    session, _ = session_for(care_actions())
    session.apply_action_filtering = True
    session.current_state.characters[0].tag = 'IMMEDIATE'
    treat(session)
    state = deepcopy(session.current_state)
    state.characters[0].tag = None
    state.characters[0].nearby = False
    state.characters[0].unseen = unseen
    observe_after_action(session, state)
    offered = session.refresh_actions()
    assert session.all_patients_handled()
    assert any(a.action_type == ActionTypeEnum.END_SCENE for a in offered)
    assert not any(a.action_type == ActionTypeEnum.TAG_CHARACTER for a in offered)
    assert 'Casualties to reach: none.' in ITMOpenWorldLangChainDriver._progress_context(session)


def test_seen_tag_does_not_make_an_unassessed_hidden_patient_complete():
    session, _ = session_for(care_actions())
    hidden = Model(id='new', name='New casualty', unseen=True, nearby=False, tag='IMMEDIATE')
    session.current_state.characters.append(hidden)
    # Even erroneous/hidden API tag fields are not observed evidence.
    session.treated_patients.update({'p1', 'new'})
    session.tagged_patients.add('p1')
    assert session.pending_patients() == {'new'}
    assert not session.all_patients_handled()
    assert 'Casualties to reach: New casualty.' in ITMOpenWorldLangChainDriver._progress_context(session)


@pytest.mark.parametrize('previous_scene,next_scene', [
    ('treat_and_tag', 'building_explosion'),
    ('treat_and_tag', 'unrelated'),
    ('scene-alpha', 'scene-beta'),
])
@pytest.mark.parametrize('trigger', [ActionTypeEnum.SEARCH, ActionTypeEnum.TREAT_PATIENT])
def test_every_scene_transition_resets_care_and_keeps_prior_tags_historical(
        previous_scene, next_scene, trigger):
    session, _ = session_for(care_actions())
    session.apply_action_filtering = True
    session.current_state.meta_info.scene_id = previous_scene
    session.current_state.characters[0].nearby = True
    session.current_state.characters[0].vitals = Model(heart_rate=90)
    treat(session)
    session.execute(action('check', ActionTypeEnum.CHECK_VITALS, character_id='p1'))
    session.execute(action('tag', ActionTypeEnum.TAG_CHARACTER, character_id='p1',
                           parameters={'category': 'IMMEDIATE'}))
    assert session.treated_patients == session.vitals_checked == {'p1'}
    assert session._treated_clinical and session._checked_clinical
    after = deepcopy(session.current_state)
    after.meta_info.scene_id = next_scene
    after.characters.append(Model(id='new', name='New casualty', tag=None, unseen=True, nearby=False))
    session.scenario.take_action = lambda selected: deepcopy(after)
    # Even a treatment that triggers a transition belongs to the prior scene.
    session.execute(action('transition', trigger,
                           character_id='p1' if trigger == ActionTypeEnum.TREAT_PATIENT else None))
    assert not session.treated_patients and not session.vitals_checked
    assert not session._treated_clinical and not session._checked_clinical
    assert not session.tagged_patients
    assert session.pending_patients() == {'p1', 'new'}
    assert not session.all_patients_handled()
    history, = session.tag_memory.snapshot()
    assert history['tag'] == 'IMMEDIATE' and history['scene_id'] == previous_scene
    assert '"same_scene": false' in session.tag_memory.context(session.current_state)
    offered = session.refresh_actions()
    assert {a.action_type for a in offered} >= {'CHECK_VITALS', 'TREAT_PATIENT', 'TAG_CHARACTER'}
    assert all(a.action_type != ActionTypeEnum.END_SCENE for a in offered)


def test_agent_can_check_and_treat_again_after_reset_without_repeating_within_scene():
    session, _ = session_for(care_actions())
    session.apply_action_filtering = True
    session.current_state.meta_info.scene_id = 'before-event'
    session.current_state.supplies = [Model(type='Blanket', quantity=3)]
    patient = session.current_state.characters[0]
    patient.nearby = True
    patient.tag = 'IMMEDIATE'
    treat(session)
    session.execute(action('check', ActionTypeEnum.CHECK_VITALS, character_id='p1'))
    after = deepcopy(session.current_state)
    after.meta_info.scene_id = 'after-event'
    after.characters[0].tag = None
    session.scenario.take_action = lambda selected: deepcopy(after)
    session.execute(action('transition', ActionTypeEnum.SEARCH))
    tools = {t.name: t for t in ITMOpenWorldLangChainDriver()._build_tools(session)}
    checked = tools['check_vitals'].invoke(dict(character_name='Alex', justification='Reassess'))
    assert checked.startswith('Action executed.')
    params = dict(character_name='Alex', treatment_supply='Blanket',
                  injury_location='unspecified', justification='Current scene decision')
    assert tools['treat_patient'].invoke(params).startswith('Action executed.')
    count = session.n_actions
    assert not tools['treat_patient'].invoke(params).startswith('Action executed.')
    assert session.n_actions == count
    assert not session.tagged_patients  # Old observed tag is still only history.


@pytest.mark.parametrize('change', ['injury', 'vitals', 'hidden', 'distant', 'unchanged'])
def test_new_clinical_facts_reopen_care_but_missing_or_hidden_fields_do_not(change):
    session, _ = session_for(care_actions())
    patient = session.current_state.characters[0]
    patient.nearby = True
    patient.tag = 'IMMEDIATE'
    patient.injuries = [Model(name='Laceration', location='left calf', status='treated')]
    patient.vitals = Model(heart_rate=90)
    treat(session)
    after = deepcopy(session.current_state)
    if change == 'injury':
        after.characters[0].injuries.append(Model(name='Puncture', location='chest', status='visible'))
    elif change == 'vitals':
        after.characters[0].vitals.heart_rate = 140
    elif change == 'hidden':
        after.characters[0].unseen = True
        after.characters[0].injuries.append(Model(name='Secret injury'))
        after.characters[0].vitals.heart_rate = 140
    elif change == 'distant':
        after.characters[0].nearby = False
        after.characters[0].injuries = None
        after.characters[0].vitals = Model()
        after.characters[0].tag = None
    observe_after_action(session, after)
    assert ('p1' in session.treated_patients) == (change not in {'injury', 'vitals'})
    if change == 'distant':
        # Reappearance must still compare with the last disclosed clinical facts.
        after.characters[0].nearby = True
        after.characters[0].vitals = Model(heart_rate=140)
        observe_after_action(session, after)
        assert 'p1' not in session.treated_patients


@pytest.mark.parametrize('run,actions_after,pending', [
    ('3955', 34, set()),
    ('3963', 48, {'Patient 1', 'Patient 2', 'Patient 3', 'Patient 5', 'Patient 6', 'Patient 8'}),
])
def test_recorded_progress_counts_only_care_performed_in_current_scene(run, actions_after, pending):
    fixture = recorded_episode(run)
    scenario = Scenario([action(**a) for a in fixture['choices']])
    scenario.state = recorded_state(fixture['steps'][0])
    session = ScenarioSession(scenario, None, True, False, lambda record: None)
    for index, step in enumerate(fixture['steps'][:-1]):
        session.refresh_actions()
        assert not session.scene_done, f'Premature completion after {index} actions'
        following = recorded_state(fixture['steps'][index + 1])
        scenario.take_action = lambda selected: deepcopy(following)
        session.execute(action(**step['action']))
        if run == '3963' and session.n_actions == 18:
            assert session.treated_patients == set()
            assert session._treated_clinical == {}
    offered = session.refresh_actions()
    assert session.n_actions == actions_after
    assert session.pending_patients() == pending
    assert session.all_patients_handled() == (not pending)
    assert any(a.action_type == ActionTypeEnum.END_SCENE for a in offered) == (not pending)
    context = ITMOpenWorldLangChainDriver._progress_context(session)
    if pending:
        assert all(name in context.split('Casualties to reach: ')[1] for name in pending)
    else:
        assert 'Casualties to reach: none.' in context


@pytest.mark.parametrize('ready_to_end', [False, True])
def test_cap_never_forces_transition_or_requests_next_scenario(tmp_path, monkeypatch, ready_to_end):
    scenario = Scenario([action('a', ActionTypeEnum.TREAT_PATIENT, character_id='p1',
                                parameters={'treatment': 'Blanket', 'location': 'unspecified'}),
                         action('end', ActionTypeEnum.END_SCENE)])
    scenario.state.supplies = [Model(type='Blanket', quantity=1)]
    scenario.get_alignment_target = lambda: None
    if ready_to_end:
        scenario.state.characters[0].tag = 'IMMEDIATE'
    interface = Model(start_scenario=Mock(side_effect=[scenario, None]),
                      get_session_alignment=Mock())
    driver = ITMOpenWorldLangChainDriver(chat_model=Actor(), max_actions_per_scenario=1)
    with pytest.raises(ScenarioActionLimitError, match='scenario remains incomplete') as caught:
        drive(tmp_path, monkeypatch, interface, driver)
    interface.start_scenario.assert_called_once()
    interface.get_session_alignment.assert_not_called()
    assert len(scenario.executed) == 1
    assert scenario.executed[0].action_type == ActionTypeEnum.TREAT_PATIENT
    status, = json.loads((tmp_path / 'scoring_status.json').read_text())
    assert status['status'] == 'incomplete' and not status['completed']
    assert 'final_state' not in status and 'api' not in status
    assert status['stall'] == caught.value.diagnostics
    assert status['stall']['reason'] == 'action_limit'
    assert status['stall']['decision_index'] == 1
    assert status['stall']['pending_patients'] == ([] if ready_to_end else ['p1'])
    assert json.loads((tmp_path / f"{status['episode_id']}.stall.json").read_text()) == status['stall']
    assert json.loads((tmp_path / 'scores.json').read_text()) == []


def test_scenario_completing_at_exact_limit_is_scored(tmp_path, monkeypatch):
    interface = Model(start_scenario=Mock(side_effect=[make_scenario(), None]),
                      get_session_alignment=Mock(return_value={'score': 0.5}))
    drive(tmp_path, monkeypatch, interface,
          ITMOpenWorldLangChainDriver(chat_model=Actor(), max_actions_per_scenario=1))
    assert interface.start_scenario.call_count == 2
    interface.get_session_alignment.assert_called_once()
    status, = json.loads((tmp_path / 'scoring_status.json').read_text())
    assert status['status'] == 'available' and status['completed']


@pytest.mark.parametrize('run,budget,expected_actions,completed', [
    ('3955', 60, 38, True), ('3963', 48, 48, False),
])
def test_recorded_agent_actions_respect_scene_reset_before_synthetic_evacuation(
        run, budget, expected_actions, completed):
    # Clinical actions/observations are recorded; the post-fix evacuation tail
    # is a local fake server, not a claim about a fresh live run or its score.
    # April's old trace has no fresh treatment for six reset patients; it must
    # remain in the care scene when its recorded actions run out.
    fixture = recorded_episode(run)
    steps = fixture['steps']
    supplies = [Model(type=supply, quantity=999) for supply in sorted({
        step['action']['parameters']['treatment'] for step in steps[:-1]
        if step['action']['action_type'] == ActionTypeEnum.TREAT_PATIENT})]
    scenario = Scenario([action(**a) for a in fixture['choices']])
    scenario.state = recorded_state(steps[0], supplies)
    evacs = []

    def take_action(selected):
        index = len(scenario.executed)
        if index < len(steps) - 1:
            expected = steps[index]['action']
            assert selected.action_id == expected['action_id']
            assert selected.character_id == expected.get('character_id')
            assert selected.parameters == (expected.get('parameters') or {})
            following = recorded_state(steps[index + 1], supplies)
        else:
            following = deepcopy(scenario.state)
            if index == len(steps) - 1:
                assert selected.action_type == ActionTypeEnum.END_SCENE
                following.meta_info.scene_id = 'evac_decision'
                # Run 3964 showed that evacuation preserves visibility:
                # distant patients remain unseen but still have listed IDs.
                if run == '3963':
                    assert any(patient.unseen for patient in following.characters)
                scenario.actions = [action('evac', ActionTypeEnum.MOVE_TO_EVAC)]
            else:
                assert selected.action_type == ActionTypeEnum.MOVE_TO_EVAC
                assert selected.character_id not in evacs
                evacs.append(selected.character_id)
                following.scenario_complete = len(evacs) == 3
        scenario.executed.append(deepcopy(selected))
        scenario.state = following
        return deepcopy(following)

    scenario.take_action = take_action
    responses = []
    for step in steps[:-1]:
        selected = step['action']
        args = {'character_name': selected['character_id']}
        params = selected.get('parameters') or {}
        if 'category' in params:
            args['triage_tag'] = params['category']
        if 'treatment' in params:
            args.update(treatment_supply=params['treatment'], injury_location=params['location'])
        responses.append(AIMessage(content='', tool_calls=[action_call(selected['action_id'], **args)]))
    responses.extend(AIMessage(content='', tool_calls=[action_call('evac', character_name=patient)])
                     for patient in ['Patient 1', 'Patient 3', 'Patient 8'])
    session = ScenarioSession(scenario, None, True, False, lambda record: None)
    driver = ITMOpenWorldLangChainDriver(chat_model=Actor(responses), max_actions_per_scenario=budget)
    driver._run_agent_loop(session)
    assert session.scenario_complete == completed
    assert session.n_actions == expected_actions
    assert len(evacs) == (3 if completed else 0)
    if not completed:
        assert not session.all_patients_handled()
        assert all(a.action_type != ActionTypeEnum.END_SCENE for a in scenario.executed)
