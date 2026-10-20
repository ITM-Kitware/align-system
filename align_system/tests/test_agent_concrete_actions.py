"""Concrete CR candidates must survive scoring and execute unchanged."""
from copy import deepcopy
import json

import pytest

from align_system.agent_tools.concrete_actions import concrete_actions
from align_system.drivers.itm_open_world_langchain import ITMOpenWorldLangChainDriver
from align_system.tests.test_agent_assessments import (
    Model, action, session_for)


def patient(identifier, unseen=False):
    return Model(id=identifier, name=identifier, unseen=unseen, tag=None,
                 unstructured=f'Patient {identifier}', injuries=[Model(
                     name='Puncture', location='left thigh', severity='severe', status='visible')])


def treatment_session():
    session, _ = session_for([action('treat', 'TREAT_PATIENT')])
    session.current_state.characters = [patient('p1'), patient('p2'), patient('hidden', True)]
    session.current_state.supplies = [Model(type='Tourniquet', quantity=2)]
    return session


def test_expands_patients_and_treatment_options_without_mutation():
    session = treatment_session()
    offered = [action('vitals', 'CHECK_VITALS'), action('move', 'MOVE_TO'),
               *session.scenario.actions, action('end', 'END_SCENE')]
    before = deepcopy([a.to_dict() for a in offered])
    candidates, omitted = concrete_actions(session.current_state, offered)
    assert not omitted
    assert [(a.action_id, a.character_id) for a in candidates] == [
        ('vitals', 'p1'), ('vitals', 'p2'), ('move', 'p1'), ('move', 'p2'),
        ('move', 'hidden'), ('treat', 'p1'), ('treat', 'p2'), ('end', None)]
    assert all(a.parameters == {'treatment': 'Tourniquet', 'location': 'left thigh'}
               for a in candidates if a.action_type == 'TREAT_PATIENT')
    treatments = [a for a in candidates if a.action_type == 'TREAT_PATIENT']
    treatments[0].parameters['location'] = 'right thigh'
    assert treatments[1].parameters['location'] == 'left thigh'
    assert [a.to_dict() for a in offered] == before


def test_fixed_patient_and_parameters_are_preserved_and_duplicates_removed():
    session = treatment_session()
    fixed = action('treat', 'TREAT_PATIENT', character_id='p2',
                   parameters={'treatment': 'Tourniquet', 'location': 'left thigh', 'extra': 'keep'})
    candidates, _ = concrete_actions(session.current_state, [fixed, deepcopy(fixed)])
    assert len(candidates) == 1
    assert candidates[0].to_dict() == fixed.to_dict()
    session.current_state.supplies[0].quantity = 0
    assert concrete_actions(session.current_state, [fixed])[0] == []


def test_assessment_candidates_respect_proximity_and_discovery_destinations():
    session = treatment_session()
    nearby, distant, unseen = session.current_state.characters
    nearby.nearby = True
    distant.nearby = False
    unseen.nearby = False
    candidates, _ = concrete_actions(session.current_state, [
        action('check', 'CHECK_VITALS'), action('move', 'MOVE_TO'),
        action('treat', 'TREAT_PATIENT'), action('tag', 'TAG_CHARACTER'),
        action('evac', 'MOVE_TO_EVAC')])
    targets = lambda kind: {a.character_id for a in candidates if a.action_type == kind}
    assert targets('CHECK_VITALS') == targets('TREAT_PATIENT') == targets('TAG_CHARACTER') == {'p1'}
    assert targets('MOVE_TO') == {'p2', 'hidden'}
    assert targets('MOVE_TO_EVAC') == {'p1', 'p2', 'hidden'}


def test_concrete_candidates_honor_session_memory_and_fixed_ineligible_target():
    session = treatment_session()
    session.current_state.characters[0].nearby = True
    session.current_state.characters[1].nearby = False
    offered = [action('check', 'CHECK_VITALS'), action('move', 'MOVE_TO'),
               action('treat', 'TREAT_PATIENT'), action('tag', 'TAG_CHARACTER'),
               action('evac', 'MOVE_TO_EVAC'),
               action('fixed', 'CHECK_VITALS', character_id='hidden')]
    candidates, omitted = concrete_actions(
        session.current_state, offered, vitals_checked={'p1'},
        treated_patients={'p1'}, tagged_patients={'p1'}, evac_patients={'p2'})
    assert [(a.action_id, a.character_id) for a in candidates] == [
        ('move', 'hidden'), ('evac', 'p1'), ('evac', 'hidden')]
    assert any(text.startswith('fixed: no eligible patient') for text in omitted)


def test_tags_expand_and_non_patient_actions_remain_single_candidates():
    session = treatment_session()
    candidates, _ = concrete_actions(session.current_state, [
        action('tag', 'TAG_CHARACTER', character_id='p1'), action('search', 'SEARCH')])
    tags = [a for a in candidates if a.action_type == 'TAG_CHARACTER']
    assert len(tags) > 1
    assert all(a.character_id == 'p1' and a.parameters['category'] for a in tags)
    assert len({json.dumps(a.parameters) for a in tags}) == len(tags)
    assert len([a for a in candidates if a.action_type == 'SEARCH']) == 1
