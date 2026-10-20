"""Observed proximity and per-scene memory constrain patient actions."""
from types import SimpleNamespace

import pytest

from align_system.utils.agent_action_eligibility import eligible_characters


def patient(identifier, **kwargs):
    return SimpleNamespace(id=identifier, unseen=False, tag=None, **kwargs)


@pytest.fixture
def state():
    unseen = patient('unseen', nearby=False)
    unseen.unseen = True
    return SimpleNamespace(characters=[patient('near', nearby=True),
                                       patient('far', nearby=False),
                                       patient('unknown'), unseen])


@pytest.mark.parametrize('kind, expected', [
    ('CHECK_VITALS', {'near', 'unknown'}),
    ('TREAT_PATIENT', {'near', 'unknown'}),
    ('TAG_CHARACTER', {'near', 'unknown'}),
    ('MOVE_TO', {'far', 'unknown', 'unseen'}),
    ('MOVE_TO_EVAC', {'near', 'far', 'unknown', 'unseen'}),
    ('SEARCH', set()),
])
def test_eligibility_uses_visibility_and_known_proximity(state, kind, expected):
    assert {c.id for c in eligible_characters(state, kind)} == expected


@pytest.mark.parametrize('kind, memory', [
    ('CHECK_VITALS', 'vitals_checked'),
    ('TREAT_PATIENT', 'treated_patients'),
    ('TAG_CHARACTER', 'tagged_patients'),
    ('MOVE_TO_EVAC', 'evac_patients'),
])
def test_handled_patients_are_excluded_from_corresponding_actions(state, kind, memory):
    assert 'near' not in {c.id for c in eligible_characters(state, kind, **{memory: {'near'}})}


def test_reported_tag_and_evacuated_patients_are_respected(state):
    state.characters[0].tag = 'IMMEDIATE'
    assert {c.id for c in eligible_characters(state, 'TAG_CHARACTER')} == {'unknown'}
    assert {c.id for c in eligible_characters(
        state, 'MOVE_TO', evac_patients={'far'})} == {'unknown', 'unseen'}


@pytest.mark.parametrize('kind', ['MOVE_TO', 'MOVE_TO_EVAC'])
def test_missing_patient_identifier_is_not_a_destination(kind):
    state = SimpleNamespace(characters=[SimpleNamespace(unseen=True, nearby=False)])
    assert eligible_characters(state, kind) == []
