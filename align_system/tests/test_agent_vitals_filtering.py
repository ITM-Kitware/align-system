"""Vitals are checked once per patient per scene; the environment never stops offering it."""
from copy import deepcopy

from swagger_client.models import ActionTypeEnum

from align_system.drivers.itm_open_world_langchain import (
    ITMOpenWorldLangChainDriver, _filter_available_actions)
from align_system.tests.test_agent_assessments import Model, action, session_for


def vitals(action_id, character_id=None):
    return action(action_id, ActionTypeEnum.CHECK_VITALS, character_id=character_id,
                  unstructured=f'Check vitals {character_id or "(any)"}')


def test_targeted_and_generic_vitals_drop_once_every_patient_is_checked():
    state = Model(characters=[Model(id='p1', tag=None, unseen=False),
                              Model(id='p2', tag=None, unseen=False)])
    offered = [vitals('v1', 'p1'), vitals('v2', 'p2'), vitals('generic'), action('s')]

    def ids(checked):
        return [a.action_id for a in _filter_available_actions(
            state, offered, set(), set(), set(), checked)]

    assert ids(set()) == ['v1', 'v2', 'generic', 's']
    assert ids({'p1'}) == ['v2', 'generic', 's']
    assert ids({'p1', 'p2'}) == ['s']


def test_session_tracks_checks_and_resets_them_on_scene_change():
    session, _ = session_for([vitals('v', 'p1'), action('s')])
    session.apply_action_filtering = True
    assert [a.action_id for a in session.refresh_actions()] == ['v', 's']

    session.execute(deepcopy(session.scenario.actions[0]))
    assert session.vitals_checked == {'p1'}
    assert [a.action_id for a in session.refresh_actions()] == ['s']

    session.scenario.state = deepcopy(session.scenario.state)
    session.scenario.state.meta_info.scene_id = 'next-scene'
    session.execute(deepcopy(session.scenario.actions[1]))
    assert session.vitals_checked == set()
    assert [a.action_id for a in session.refresh_actions()] == ['v', 's']


def test_generic_check_vitals_tool_refuses_a_repeat_for_the_same_casualty():
    session, _ = session_for([vitals('generic'), action('s')])
    session.apply_action_filtering = True
    session.current_state.characters.append(
        Model(id='p2', name='Sam', tag=None, unseen=False, unstructured='Civilian'))
    driver = ITMOpenWorldLangChainDriver.__new__(ITMOpenWorldLangChainDriver)
    driver.apply_action_filtering = True
    driver._observation_text = lambda state, session=None: 'observation'
    first = driver._perform_typed_action(session, ActionTypeEnum.CHECK_VITALS, 'why',
                                         character_name='Alex')
    assert session.scenario.executed[-1].character_id == 'p1'
    repeat = driver._perform_typed_action(session, ActionTypeEnum.CHECK_VITALS, 'why',
                                          character_name='Alex')
    assert len(session.scenario.executed) == 1
    assert 'already checked' in repeat and 'already checked' not in str(first)
    driver._perform_typed_action(session, ActionTypeEnum.CHECK_VITALS, 'why', character_name='Sam')
    assert [a.character_id for a in session.scenario.executed] == ['p1', 'p2']
    assert [a.action_id for a in session.refresh_actions()] == ['s']
