"""Memory survives partial observations; reminders never force CR inference."""
from copy import deepcopy
import json

import pytest
from swagger_client.models import ActionTypeEnum

from align_system.drivers.itm_open_world_langchain import (
    ActionRejectedException, ITMOpenWorldLangChainDriver)
from align_system.tests.test_agent_assessments import (
    Model, action, session_for)
from align_system.tests.test_agent_controls_and_events import Actor
from align_system.utils.patient_tag_memory import PatientTagMemory


def tag_action(patient='p1', category='IMMEDIATE', **kwargs):
    return action('tag', ActionTypeEnum.TAG_CHARACTER, character_id=patient,
                  parameters={'category': category}, **kwargs)


def test_tags_persist_when_missing_and_new_observations_replace_them():
    session, _ = session_for()
    state = session.current_state
    state.characters[0].tag = 'IMMEDIATE'
    state.characters.append(Model(id='hidden', name='Hidden', tag='EXPECTANT', unseen=True))
    memory = PatientTagMemory()
    memory.observe(state, 0)
    assert len(memory.snapshot()) == 1
    state.characters[0].tag = None
    before = deepcopy(state.to_dict())
    memory.observe(state, 1)
    text = memory.context(state)
    assert 'IMMEDIATE' in text and '"currently_observed": false' in text
    assert 'unknown, not untagged' in text
    assert 'hidden' not in text and 'EXPECTANT' not in text
    assert state.to_dict() == before
    state.characters[0].tag = 'DELAYED'
    memory.observe(state, 2)
    assert memory.snapshot()[0]['tag'] == 'DELAYED'
    assert memory.snapshot()[0]['source'] == 'observation'
    assert '"currently_observed": true' in memory.context(state)
    copy = memory.snapshot()
    copy[0]['tag'] = 'corrupted'
    assert memory.snapshot()[0]['tag'] == 'DELAYED'


def test_successful_tag_is_remembered_without_a_tag_in_the_response():
    session, records = session_for([tag_action()])
    session.execute(deepcopy(session.scenario.actions[0]))
    record = session.tag_memory.snapshot()[0]
    assert record == {'patient_id': 'p1', 'patient_name': 'Alex', 'tag': 'IMMEDIATE',
                      'scene_id': 'scene', 'source': 'successful_tag_action',
                      'after_action': 1}
    assert session.current_state.characters[0].tag is None
    assert records[0]['choice_info']['patient_tag_memory'] == []  # Before the action.
    session.execute(action('move', ActionTypeEnum.MOVE_TO, character_id='p1'))
    assert records[1]['choice_info']['patient_tag_memory'] == [record]


def test_rejected_and_intended_tags_do_not_become_successful_memories():
    session, _ = session_for([tag_action()])

    class Rejected(Exception):
        status = 400

    def reject(_):
        raise Rejected('Cannot tag this patient')

    session.scenario.take_action = reject
    with pytest.raises(ActionRejectedException):
        session.execute(tag_action())
    assert session.tag_memory.snapshot() == []
    memory = PatientTagMemory()
    memory.record_success(tag_action(intent_action=True), session.current_state, 1)
    assert memory.snapshot() == []


def test_prior_scene_tags_are_historical_until_observed_again():
    session, _ = session_for([tag_action()])
    session.execute(tag_action())
    session.scenario.state = deepcopy(session.current_state)
    session.scenario.state.meta_info.scene_id = 'new-scene'
    session.execute(action('move', ActionTypeEnum.MOVE_TO, character_id='p1'))
    assert session.tagged_patients == set()
    assert session.recent_actions == []
    text = session.tag_memory.context(session.current_state)
    assert '"same_scene": false' in text and '"scene_id": "scene"' in text
    session.current_state.characters[0].tag = 'DELAYED'
    session.tag_memory.observe(session.current_state, session.n_actions)
    record = session.tag_memory.snapshot()[0]
    assert record['scene_id'] == 'new-scene' and record['tag'] == 'DELAYED'
