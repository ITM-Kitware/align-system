"""Concrete CR candidates must survive scoring and execute unchanged."""
from copy import deepcopy
import json

import pytest

from align_system.agent_tools.concrete_actions import concrete_actions
from align_system.drivers.itm_open_world_langchain import ITMOpenWorldLangChainDriver
from align_system.tests.test_agent_assessments import (
    Model, action, session_for, toolkit_for)


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


@pytest.mark.parametrize('change', ['empty_inventory', 'unknown_inventory', 'hidden_injury',
                                    'treated_injury', 'missing_injuries', 'fixed_conflict'])
def test_uncompletable_treatments_are_explained_not_scored(change):
    session = treatment_session()
    if change == 'empty_inventory':
        session.current_state.supplies[0].quantity = 0
    elif change == 'unknown_inventory':
        session.current_state.supplies = None
    elif change in ('hidden_injury', 'treated_injury'):
        for p in session.current_state.characters:
            p.injuries[0].status = 'hidden' if change == 'hidden_injury' else 'treated'
    elif change == 'missing_injuries':
        for p in session.current_state.characters:
            p.injuries = []
    else:
        session.scenario.actions[0].parameters = {'location': 'right thigh'}
    toolkit = toolkit_for('comparative_regression')
    text = toolkit.assessment_text(session)
    assert 'No concrete candidates' in text
    assert 'Not scored' in text
    assert toolkit.chat_model.calls == 0
    assert not session.assessments['comparative_regression']['rows']


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


def test_single_template_is_scored_for_each_patient_and_executes_exact_parameters():
    session = treatment_session()
    toolkit = toolkit_for('comparative_regression')
    driver = ITMOpenWorldLangChainDriver(extra_tools={'cr': toolkit}, apply_action_filtering=False)
    tools = {t.name: t for t in driver._build_tools(session)}
    text = toolkit.decision_context(session)
    assert 'Tourniquet' in text and 'left thigh' in text
    choices = list(toolkit.chat_model.schema['properties'])
    assert choices == ['C01', 'C02']
    query = toolkit.chat_model.messages[0][-1].content
    assert 'p1' in query and 'p2' in query
    assert 'Tourniquet' in query and 'left thigh' in query
    assert 'hidden' not in query
    tools['score_available_actions'].invoke({'justification': 'Review'})
    assert toolkit.chat_model.calls == 1
    scored = session.assessments['comparative_regression']['rows'][1]['action']
    tools['execute_available_action'].invoke({
        'action_id': scored['action_id'], 'character_name': scored['character_id'],
        'treatment_supply': scored['parameters']['treatment'],
        'injury_location': scored['parameters']['location'], 'justification': 'Choose scored action'})
    executed = session.scenario.executed[-1]
    assert executed.character_id == scored['character_id']
    assert executed.parameters == scored['parameters']
    assert executed.action_id == scored['action_id']


def test_filters_handled_patients_after_expanding_and_invalidates_changed_state():
    session = treatment_session()
    session.apply_action_filtering = True
    session.treated_patients.add('p1')
    toolkit = toolkit_for('comparative_regression')
    toolkit.assessment_text(session)
    rows = session.assessments['comparative_regression']['rows']
    assert {r['action']['character_id'] for r in rows} == {'p2'}
    session.current_state.characters[1].unstructured = 'Updated patient information'
    toolkit.assessment_text(session)
    assert toolkit.chat_model.calls == 2
    session.current_state.supplies[0].quantity = 0
    toolkit.assessment_text(session)
    assert session.assessments['comparative_regression']['rows'] == []


def test_tags_expand_and_non_patient_actions_remain_single_candidates():
    session = treatment_session()
    candidates, _ = concrete_actions(session.current_state, [
        action('tag', 'TAG_CHARACTER', character_id='p1'), action('search', 'SEARCH')])
    tags = [a for a in candidates if a.action_type == 'TAG_CHARACTER']
    assert len(tags) > 1
    assert all(a.character_id == 'p1' and a.parameters['category'] for a in tags)
    assert len({json.dumps(a.parameters) for a in tags}) == len(tags)
    assert len([a for a in candidates if a.action_type == 'SEARCH']) == 1


def test_fewshot_examples_reach_scorer_before_concrete_query_and_are_recorded():
    from align_system.data_models.dialog import DialogElement

    class Examples:
        calls = 0

        def run(self, state, choices, actions, alignment_target):
            self.calls += 1
            assert all(a.character_id in {'p1', 'p2'} for a in actions)
            assert all(a.parameters['treatment'] == 'Tourniquet' for a in actions)
            return ({'affiliation': [DialogElement(role='user', content='Dataset question'),
                                     DialogElement(role='assistant', content='Dataset scores')]},
                    {'affiliation': [{'prompt': 'Dataset question', 'response': 'Dataset scores'}]})

    examples = Examples()
    session = treatment_session()
    toolkit = toolkit_for('comparative_regression', icl=examples)
    original_batch = toolkit.chat_model.batch

    def capture(messages):
        dialog = messages[0]
        assert [m.type for m in dialog] == ['system', 'human', 'ai', 'human']
        assert dialog[1].content == 'Dataset question'
        assert dialog[2].content == 'Dataset scores'
        assert 'Tourniquet' in dialog[-1].content and 'p2' in dialog[-1].content
        return original_batch(messages)

    toolkit.chat_model.batch = capture
    toolkit.assessment_text(session)
    toolkit.assessment_text(session)
    assert examples.calls == 1
    assert session.assessments['comparative_regression']['icl_examples']['affiliation'][0][
        'prompt'] == 'Dataset question'


def test_missing_fewshot_dataset_does_not_silently_become_zero_shot():
    class MissingExamples:
        def run(self, *args):
            raise FileNotFoundError('Missing configured ICL dataset')

    toolkit = toolkit_for('comparative_regression', icl=MissingExamples())
    with pytest.raises(FileNotFoundError, match='ICL dataset'):
        toolkit.assessment_text(treatment_session())
    assert toolkit.chat_model.calls == 0


def test_ow_fewshot_uses_twenty_examples_with_reference_datasets_and_selection():
    from pathlib import Path
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    config_dir = str(Path(__file__).resolve().parents[1] / 'configs')
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        reference = compose(config_name='action_based', overrides=[
            '+experiment=phase2_feb_collab/phase2_pipeline_fewshot_comparative_regression_random_effects'])
        agent = compose(config_name='action_based', overrides=[
            '+experiment=phase2_feb_openworld/phase2_langchain_agent_ow_comparative_regression_live'])
    reference_icl = reference.adm.step_definitions.regression_icl
    agent_icl = agent.driver.extra_tools.comparative_regression.icl
    assert agent_icl._target_ == reference_icl._target_
    agent_settings = OmegaConf.to_container(agent_icl.icl_generator_partial, resolve=True)
    reference_settings = OmegaConf.to_container(reference_icl.icl_generator_partial, resolve=True)
    assert agent_settings['incontext_settings']['number'] == 20
    assert reference_settings['incontext_settings']['number'] == 20
    assert agent_settings.pop('scorer') == {
        '_target_': 'align_system.utils.bert_scorer.BatchedBERTScorer',
        'batch_size': 32, 'device': 'cuda'}
    assert agent_settings == reference_settings
    assert agent_icl.scenario_description_template._target_ == (
        'align_system.prompt_engineering.outlines_prompts.Phase2ScenarioDescription')
