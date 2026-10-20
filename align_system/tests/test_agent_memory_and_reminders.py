"""Memory survives partial observations; reminders never force CR inference."""
from copy import deepcopy
import json

import pytest
from swagger_client.models import ActionTypeEnum

from align_system.agent_tools.cr_reminder import CRReminder
from align_system.drivers.itm_open_world_langchain import (
    ActionRejectedException, ITMOpenWorldLangChainDriver)
from align_system.tests.test_agent_assessments import (
    Model, action, session_for, toolkit_for)
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


@pytest.mark.parametrize('with_cr', [False, True])
def test_memory_reaches_actor_after_tag_action_falls_out_of_history(with_cr):
    session, records = session_for()
    session.execute(tag_action())
    actor = Actor()
    events = []
    session.record_event = lambda event, **fields: events.append((event, fields))
    providers = {'cr': toolkit_for('comparative_regression', auto=False)} if with_cr else {}
    driver = ITMOpenWorldLangChainDriver(
        chat_model=actor, apply_action_filtering=False, extra_tools=providers,
        max_messages_in_context=2, max_actions_per_scenario=5)
    driver._run_agent_loop(session)
    assert len(actor.messages) == 4
    for messages in actor.messages:
        text = messages[0].content
        assert 'LAST-KNOWN PATIENT TAGS' in text
        assert '"patient_id": "p1"' in text and '"tag": "IMMEDIATE"' in text
        assert '"source": "successful_tag_action"' in text
        assert text.count('LAST-KNOWN PATIENT TAGS') == 1
    assert sum(event == 'patient_tag_memory' for event, _ in events) == 4
    if with_cr:
        assert providers['cr'].chat_model.calls == 0
    json.dumps(records)


def test_cr_reminders_have_a_cooldown_and_reset_at_scene_entry():
    session, _ = session_for()
    toolkit = toolkit_for('comparative_regression', auto=False)
    toolkit.build_tools(session)
    assert 'not consulted CR in this scene' in toolkit.decision_context(session)
    for count in range(5):
        session.n_actions = count
        assert toolkit.decision_context(session) is None
    session.n_actions = 5
    assert 'CR REMINDER' in toolkit.decision_context(session)
    assert toolkit.chat_model.calls == 0 and not session.assessments
    session.current_state.meta_info.scene_id = 'next'
    assert 'scene next' in toolkit.decision_context(session)
    assert toolkit.chat_model.calls == 0


@pytest.mark.parametrize('failed', [False, True])
def test_cr_requests_suppress_reminders_even_after_assessment_is_cleared(failed):
    session, _ = session_for()
    toolkit = toolkit_for('comparative_regression', auto=False)
    tool = toolkit.build_tools(session)[0]
    toolkit.chat_model.fail = failed
    session.n_actions = 2
    tool.invoke({'justification': 'Compare options'})
    calls = toolkit.chat_model.calls
    session.assessments.clear()
    for count in range(2, 7):
        session.n_actions = count
        assert toolkit.decision_context(session) is None
    session.n_actions = 7
    assert 'last CR request was 5' in toolkit.decision_context(session)
    assert toolkit.chat_model.calls == calls


def test_repeated_moves_trigger_a_stronger_reminder_without_automatic_scoring():
    session, _ = session_for()
    toolkit = toolkit_for('comparative_regression', auto=False)
    toolkit.decision_context(session)
    for patient in ('p1', 'p2', 'p1', 'p2'):
        session.execute(action('move', ActionTypeEnum.MOVE_TO, character_id=patient))
    text = toolkit.decision_context(session)
    assert 'last four actions repeatedly moved' in text
    assert 'last-known tags' in text
    assert toolkit.decision_context(session) is None
    assert toolkit.chat_model.calls == 0
    session.execute(action('move', ActionTypeEnum.MOVE_TO, character_id='p1'))
    assert toolkit.decision_context(session) is None


def test_reminders_can_be_disabled_and_require_an_alignment_target():
    session, _ = session_for()
    toolkit = toolkit_for('comparative_regression', auto=False, reminder_interval=0)
    assert toolkit.decision_context(session) is None
    toolkit = toolkit_for('comparative_regression', auto=False)
    session.alignment_target = None
    assert toolkit.decision_context(session) is None


@pytest.mark.parametrize('invalid', [-1, True, 2.5])
def test_reminder_interval_validation(invalid):
    with pytest.raises(ValueError, match='reminder_interval'):
        CRReminder(invalid)
