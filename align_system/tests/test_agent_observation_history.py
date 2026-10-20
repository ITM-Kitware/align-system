"""Scene resets preserve evidence while current care stays visible to the actor."""
from copy import deepcopy
import json

import pytest
from langchain_core.messages import AIMessage

from align_system.drivers.itm_open_world_langchain import (
    ActionRejectedException, ITMOpenWorldLangChainDriver)
from align_system.tests.test_agent_assessments import Model, action, session_for
from align_system.tests.test_agent_controls_and_events import Actor
from align_system.utils.patient_observation_history import PatientObservationHistory


def test_descriptions_keep_provenance_without_copying_hidden_or_current_fields():
    session, _ = session_for()
    state = deepcopy(session.current_state)
    state.characters[0].vitals = {'heart_rate': 90}
    state.characters[0].kdma_association = {'secret_label': 1}
    state.characters.append(Model(id='hidden', name='Alex', unseen=True,
                                  unstructured='SECRET DESCRIPTION'))
    history = PatientObservationHistory()
    history.observe(state, 0)
    history.observe(state, 1)
    state.meta_info.scene_id = 'next'
    state.characters[0].unstructured = 'Patient has been treated.'
    history.observe(state, 2)
    state.characters[0].unstructured = None
    history.observe(state, 3)
    snapshot = history.snapshot()
    descriptions = snapshot['p1']['descriptions']
    assert len(descriptions) == 2
    assert descriptions[0] == {
        'value': 'Soldier in your unit',
        'first_seen': {'scene_id': 'scene', 'after_action': 0},
        'last_seen': {'scene_id': 'scene', 'after_action': 1},
    }
    assert descriptions[1]['first_seen'] == {'scene_id': 'next', 'after_action': 2}
    assert all(text not in json.dumps(snapshot)
               for text in ('hidden', 'SECRET', 'secret_label', 'heart_rate'))
    descriptions[0]['value'] = 'corrupted'
    assert history.snapshot()['p1']['descriptions'][0]['value'] == 'Soldier in your unit'


def test_same_name_patients_and_repeated_scenarios_have_separate_history():
    session, _ = session_for()
    state = deepcopy(session.current_state)
    state.characters.append(Model(id='p2', name='Alex', unseen=False,
                                  unstructured='Civilian'))
    session.patient_history.observe(state, 1)
    assert session.patient_history.snapshot()['p2']['descriptions'][0]['value'] == 'Civilian'
    other, _ = session_for()
    assert 'p2' not in other.patient_history.snapshot()


def test_rejected_actions_leave_history_unchanged():
    session, _ = session_for()
    before = session.patient_history.snapshot()

    class Rejected(Exception):
        status = 400

    def reject(_):
        raise Rejected('Unavailable')

    session.scenario.take_action = reject
    with pytest.raises(ActionRejectedException):
        session.execute(action('a'))
    assert session.patient_history.snapshot() == before


@pytest.mark.parametrize('reset_awareness', [False, True])
def test_current_care_and_history_survive_conversation_trimming(reset_awareness):
    session, records = session_for()
    session.reset_awareness = reset_awareness
    session.execute(action('tag', 'TAG_CHARACTER', character_id='p1',
                           parameters={'category': 'IMMEDIATE'}))
    session.execute(action('treat', 'TREAT_PATIENT', character_id='p1'))
    session.evac_patients.add('p1')
    before_transition = session.patient_history.snapshot()
    changed = deepcopy(session.current_state)
    changed.meta_info.scene_id = 'next'
    changed.characters[0].unstructured = 'Patient has been treated.'
    changed.characters[0].kdma_association = {'secret_label': 999}
    session.scenario.take_action = lambda selected: deepcopy(changed)
    session.execute(action('a'))
    assert records[-1]['choice_info']['patient_history'] == before_transition
    assert session.actions_observed_at is None and session.assessments == {}
    assert not session.treated_patients | session.evac_patients | session.tagged_patients

    actor = Actor()
    driver = ITMOpenWorldLangChainDriver(
        chat_model=actor, max_messages_in_context=1, max_actions_per_scenario=6)
    driver._run_agent_loop(session)
    assert len(actor.messages) == 3
    for messages in actor.messages:
        system = messages[0].content
        context = json.loads(system.split('Current visible state and patient history:\n')[1]
                             .split('\n\nCurrent scene progress:')[0])
        assert context['scene_id'] == 'next'
        assert context['handled_patients'] == {'treated': [], 'evacuated': [], 'tagged': []}
        assert context['casualties'][0]['unstructured'] == 'Patient has been treated.'
        assert context['patient_history']['p1']['descriptions'][0]['value'] == 'Soldier in your unit'
        assert 'LAST-KNOWN PATIENT TAGS' in system and 'IMMEDIATE' in system
        assert 'secret_label' not in system
        assert ('patient_progress' in context) == reset_awareness


def test_automatic_scene_transition_refreshes_actions_before_next_model_request():
    session, _ = session_for([
        action('treat', 'TREAT_PATIENT', character_id='p1',
               parameters={'treatment': 'Pressure bandage', 'location': 'left calf'}),
        action('end', 'END_SCENE')])
    session.current_state.characters[0].tag = 'IMMEDIATE'

    def execute(selected):
        state = deepcopy(session.current_state)
        if selected.action_type == 'END_SCENE':
            state.meta_info.scene_id = 'evac'
            session.scenario.actions = [action('evac', 'MOVE_TO_EVAC', character_id='p1')]
        else:
            state.scenario_complete = selected.action_type == 'MOVE_TO_EVAC'
        return state

    session.scenario.take_action = execute
    actor = Actor([AIMessage(content='', tool_calls=[{
        'id': name, 'name': name,
        'args': {'character_name': 'p1', 'justification': 'Provide care'},
    }]) for name in ('treat_patient', 'move_to_evac')])
    driver = ITMOpenWorldLangChainDriver(chat_model=actor)
    driver._run_agent_loop(session)
    assert session.scenario_complete
    assert len(actor.messages) == 2
    assert 'evacuation is the only offered action' in actor.messages[1][0].content
    assert '"scene_id": "evac"' in actor.messages[1][0].content
