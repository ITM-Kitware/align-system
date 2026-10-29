"""Offline contract tests for assessment modes and agent action execution."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from swagger_client.models import ActionTypeEnum, KDMAValue

from align_system.agent_tools.kaleido import KaleidoToolkit
from align_system.data_models.attribute import Attribute
from align_system.drivers.itm_open_world_langchain import (
    ITMOpenWorldLangChainDriver, ScenarioSession)
from align_system.utils.alignment_targets import (
    alignment_target_entries, alignment_target_values)


class Model(SimpleNamespace):
    def to_dict(self):
        def convert(value):
            if hasattr(value, 'to_dict'):
                return value.to_dict()
            if isinstance(value, list):
                return [convert(v) for v in value]
            if isinstance(value, dict):
                return {k: convert(v) for k, v in value.items()}
            return value
        return {k: convert(v) for k, v in vars(self).items()}


def action(action_id, action_type=ActionTypeEnum.SEARCH, **kwargs):
    return Model(**dict(dict(action_id=action_id, action_type=action_type,
                            unstructured='Same description', character_id=None,
                            parameters={}, justification=None,
                            intent_action=False, kdma_association=None), **kwargs))


def action_call(action_id='a', justification='Choose action', **kwargs):
    return {'id': f'call-{action_id}', 'name': 'execute_available_action',
            'args': {'action_id': action_id, 'justification': justification, **kwargs}}


class Scenario:
    def __init__(self, actions):
        self.actions = actions
        self.executed = []
        self.state = Model(
            unstructured='Two casualties await assessment.',
            scenario_complete=False, meta_info=Model(scene_id='scene'),
            characters=[Model(id='p1', name='Alex', tag=None, unseen=False,
                              unstructured='Soldier in your unit')],
            supplies=None)

    def id(self):
        return 'scenario'

    def get_state(self):
        return self.state

    def get_available_actions(self):
        return self.actions

    def take_action(self, selected):
        self.executed.append(deepcopy(selected))
        return deepcopy(self.state)


def session_for(actions=None):
    scenario = Scenario(actions if actions is not None else [action('a'), action('b')])
    records = []
    target = Model(id='target', kdma_values=[{'kdma': 'affiliation', 'value': 0.8}])
    session = ScenarioSession(scenario, target, False, False, records.append)
    return session, records


class FakeKaleido:
    def __init__(self):
        self.calls = 0
        self.explanation_calls = 0
        self.texts = []

    def get_relevance(self, texts, *args, **kwargs):
        self.calls += 1
        self.texts = texts
        return [[0.9, 0.1] for _ in texts]

    def get_valence(self, texts, *args, **kwargs):
        return [[0.8, 0.1, 0.1] for _ in texts]

    def get_explanation(self, texts, *args, **kwargs):
        self.explanation_calls += 1
        return ['Evidence explanation' for _ in texts]


def test_context_error_aborts_without_retry_or_fallback():
    from align_system.exceptions import ContextWindowError

    class OversizedAgent:
        calls = 0

        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            self.calls += 1
            raise ContextWindowError('Newest observation exceeds the context limit')

    agent = OversizedAgent()
    session, records = session_for()
    driver = ITMOpenWorldLangChainDriver(chat_model=agent)
    with pytest.raises(ContextWindowError, match='Newest observation'):
        driver._run_agent_loop(session)
    assert agent.calls == 1
    assert not records
    assert not session.scenario.executed


def test_message_window_keeps_complete_newest_tool_exchange():
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    opening = HumanMessage('Begin')
    calls = AIMessage(content='', tool_calls=[
        {'name': 'observe', 'args': {}, 'id': str(i)} for i in range(4)])
    latest = [calls, *[ToolMessage(content='Observation', tool_call_id=str(i))
                      for i in range(4)]]
    messages = [opening, AIMessage('Old turn'), *latest]
    driver = ITMOpenWorldLangChainDriver(max_messages_in_context=3)
    assert driver._trim_message_window(messages) == [opening, *latest]


def test_ambiguous_tool_requires_id_and_stale_id_is_rejected():
    session, _ = session_for([action('a', ActionTypeEnum.MESSAGE),
                              action('b', ActionTypeEnum.MESSAGE)])
    driver = ITMOpenWorldLangChainDriver(apply_action_filtering=False)
    tools = {t.name: t for t in driver._build_tools(session)}
    assert 'Multiple' in tools['send_message'].invoke({'justification': 'Communicate'})
    assert not session.scenario.executed
    assert 'unavailable' in tools['execute_available_action'].invoke(
        {'action_id': 'stale', 'justification': 'Communicate'})
    tools['execute_available_action'].invoke({'action_id': 'b', 'justification': 'Communicate'})
    assert session.scenario.executed[0].action_id == 'b'


def test_exact_action_preserves_parameters_and_allows_missing_patient():
    selected = action('a', ActionTypeEnum.TAG_CHARACTER, parameters={'category': 'IMMEDIATE'})
    session, records = session_for([selected])
    driver = ITMOpenWorldLangChainDriver(apply_action_filtering=False)
    executor = {t.name: t for t in driver._build_tools(session)}['execute_available_action']
    args = {'action_id': 'a', 'character_name': 'Alex', 'justification': 'Tag patient'}
    assert 'conflicts' in executor.invoke({**args, 'triage_tag': 'DELAYED'})
    assert not session.scenario.executed
    executor.invoke(args)
    assert session.scenario.executed[0].character_id == 'p1'
    assert session.scenario.executed[0].parameters == {'category': 'IMMEDIATE'}
    assert selected.character_id is None  # Template was not mutated.
    assert records[0]['output']['action']['character_id'] == 'p1'


def test_only_one_action_per_response_and_fallback_source_recorded():
    session, records = session_for()
    driver = ITMOpenWorldLangChainDriver(apply_action_filtering=False)
    tools = {t.name: t for t in driver._build_tools(session)}
    messages = []
    calls = [{'id': str(i), 'name': 'execute_available_action',
              'args': {'action_id': aid, 'justification': 'Search'}}
             for i, aid in enumerate(['a', 'b'])]
    assert driver._handle_tool_calls(session, tools, calls, messages)
    assert len(session.scenario.executed) == 1
    assert len(messages) == 2
    assert 'not executed' in messages[1].content
    driver._take_fallback_action(session)
    assert records[1]['choice_info']['langchain_agent']['decision_source'] == 'fallback'
    assert records[1]['choice_info']['assessments'] == {}


def test_missing_inventory_does_not_discard_explicit_treatment():
    session, _ = session_for()
    selected = action('a', ActionTypeEnum.TREAT_PATIENT)
    fill = ITMOpenWorldLangChainDriver._fill_treatment_parameters
    assert fill(session, selected, 'treat_patient', 'Tourniquet', 'left calf') is None
    assert selected.parameters == {'treatment': 'Tourniquet', 'location': 'left calf'}
    session.current_state.supplies = []
    assert 'No treatment supplies' in fill(session, selected, 'treat_patient', 'Tourniquet', '')


@pytest.mark.parametrize('representation', ['dict', 'omegaconf', 'swagger', 'object'])
def test_shared_target_normalization(representation):
    entries = [{'kdma': 'affiliation', 'value': 0.8},
               {'kdma': 'distribution', 'value': None, 'parameters': [{'x': 1}]}]
    target = {'kdma_values': entries}
    if representation == 'omegaconf':
        target = OmegaConf.create(target)
    elif representation == 'swagger':
        target = Model(kdma_values=[KDMAValue(kdma='affiliation', value=0.8)])
    elif representation == 'object':
        target = Model(kdma_values=[SimpleNamespace(**entry) for entry in entries])
    assert alignment_target_values(target) == {'affiliation': 0.8}
    if representation != 'swagger':
        assert alignment_target_entries(target)[1]['parameters'] == [{'x': 1}]
    assert alignment_target_values(None) == {}


@pytest.mark.parametrize('auto', [True, False])
@pytest.mark.parametrize('explain', [True, False])
def test_kaleido_original_assessment_with_current_driver(auto, explain):
    session, records = session_for([
        action('a', unstructured='Check Alex', character_id='p1'),
        action('b', unstructured='Check Jordan', character_id='p2')])
    backend = FakeKaleido()
    toolkit = KaleidoToolkit(backend, {'affiliation': {
        'description': 'Favor patients in your group', 'vrd': 'Value'}},
        auto_assess=auto, explain=explain, assessment_subject='actions')
    driver = ITMOpenWorldLangChainDriver(
        extra_tools={'kaleido': toolkit}, apply_action_filtering=False)
    tools = {tool.name: tool for tool in driver._build_tools(session)}
    context = toolkit.decision_context(session)
    assert (context is not None) == auto
    text = tools['assess_available_actions'].invoke({'justification': 'Assess choices'})
    assert 'estimate 0.85 vs target 0.80 (distance 0.05)' in text
    assert 'Closest to target overall:' in text
    assert ('Evidence explanation' in text) == explain
    assert backend.explanation_calls == int(explain)
    assert 'Choosing to "Check Alex" instead of "Check Jordan".' in backend.texts[0]
    assert 'action_id' not in backend.texts[0]
    assert session.assessments['kaleido']['targets'] == {'affiliation': 0.8}
    assert backend.calls == 1  # Automatic context and tool share the original cache.
    tools['execute_available_action'].invoke({'action_id': 'b', 'justification': 'Choose Jordan'})
    assert records[0]['output']['action']['action_id'] == 'b'
    assert records[0]['choice_info']['assessments']['kaleido']['targets'] == {'affiliation': 0.8}
    tools['assess_available_actions'].invoke({'justification': 'Assess next decision'})
    assert backend.calls == 2


def test_legacy_prompt_import_still_works():
    from align_system.prompt_engineering.ow_prompts import KaleidoAutoAssessHintPrompt
    assert 'closest' in KaleidoAutoAssessHintPrompt()()


def test_forced_end_is_labeled():
    session, records = session_for([action('end', ActionTypeEnum.END_SCENE)])
    session.refresh_actions()
    ITMOpenWorldLangChainDriver()._force_end_scene(session, 'Action limit reached')
    assert records[0]['choice_info']['langchain_agent']['decision_source'] == 'forced_end_scene'


def test_existing_attribute_target_conversion_preserves_parameters():
    from align_system.utils.alignment_utils import (
        alignment_target_to_attribute_targets, attributes_in_alignment_target)
    target = OmegaConf.create({'kdma_values': [
        {'kdma': 'affiliation', 'value': 0.8, 'parameters': [{'offset': 1}]}]})
    definitions = {'affiliation': Attribute(name='Affiliation', kdma='affiliation')}
    converted = alignment_target_to_attribute_targets(target, definitions)
    assert converted[0].value == 0.8
    assert converted[0].parameters == [{'offset': 1}]
    assert attributes_in_alignment_target(target) == ['affiliation']


class TargetTrackingKaleido(FakeKaleido):
    def __init__(self):
        super().__init__()
        self.descriptions = []

    def get_relevance(self, texts, vrds, descriptions, **kwargs):
        self.descriptions.append(list(descriptions))
        return super().get_relevance(texts, vrds, descriptions, **kwargs)


def target_tracking_kaleido(auto=True):
    backend = TargetTrackingKaleido()
    toolkit = KaleidoToolkit(backend, {
        'merit': {'description': 'Favor victims over perpetrators'},
        'affiliation': {'description': 'Favor patients in your group'},
    }, auto_assess=auto, assessment_subject='actions')
    return toolkit, backend


def assert_kaleido_target(text, attribute, value):
    assert f'Alignment target: {attribute}={value:.2f}' in text
    assert (f'estimate 0.85 vs target {value:.2f} '
            f'(distance {abs(0.85 - value):.2f})') in text


def test_kaleido_invalidates_cache_when_target_value_changes_in_place():
    toolkit, backend = target_tracking_kaleido()
    session, _ = session_for()
    entry = session.alignment_target.kdma_values[0]
    entry.update(kdma='merit', value=0.0)
    assert_kaleido_target(toolkit.assessment_text(session), 'merit', 0.0)

    entry['value'] = 1.0  # Same target ID, session, and action count.
    text = toolkit.assessment_text(session)
    assert_kaleido_target(text, 'merit', 1.0)
    assert backend.calls == 2
    assert toolkit.assessment_text(session) == text
    assert backend.calls == 2


def test_kaleido_invalidates_cache_when_target_attribute_changes():
    toolkit, backend = target_tracking_kaleido()
    session, _ = session_for()
    entry = session.alignment_target.kdma_values[0]
    entry.update(kdma='merit', value=0.4)
    assert_kaleido_target(toolkit.assessment_text(session), 'merit', 0.4)

    entry['kdma'] = 'affiliation'  # The scalar target is unchanged.
    assert_kaleido_target(toolkit.assessment_text(session), 'affiliation', 0.4)
    assert backend.calls == 2
    assert backend.descriptions == [
        ['Favor victims over perpetrators'] * 2,
        ['Favor patients in your group'] * 2,
    ]


@pytest.mark.parametrize('method', ['assessment_text', 'decision_context'])
def test_kaleido_cache_isolated_for_sessions_without_build_tools(method):
    toolkit, backend = target_tracking_kaleido()
    first, _ = session_for()
    second, _ = session_for()
    assert first.scenario.id() == second.scenario.id()
    assert first.n_actions == second.n_actions == 0
    # Identical targets ensure isolation depends on the actual session.
    assess = getattr(toolkit, method)
    for index, session in enumerate((first, second), start=1):
        text = assess(session)
        assert_kaleido_target(text, 'affiliation', 0.8)
        assert backend.calls == index
        assert assess(session) == text
        assert backend.calls == index


def test_kaleido_interleaved_tool_closures_keep_their_session_targets():
    toolkit, backend = target_tracking_kaleido()
    first, _ = session_for()
    second, _ = session_for()
    first.alignment_target.kdma_values[0].update(kdma='merit', value=0.0)
    second.alignment_target.kdma_values[0].update(kdma='affiliation', value=1.0)
    first_tool = toolkit.build_tools(first)[0]
    second_tool = toolkit.build_tools(second)[0]

    for bound_tool, attribute, value in (
            (first_tool, 'merit', 0.0),
            (second_tool, 'affiliation', 1.0),
            (first_tool, 'merit', 0.0)):
        text = bound_tool.invoke({'justification': 'Assess this session'})
        assert_kaleido_target(text, attribute, value)
    assert backend.calls >= 2


@pytest.mark.parametrize('auto', [True, False])
def test_kaleido_driver_24_session_merit_affiliation_sweep(auto):
    toolkit, backend = target_tracking_kaleido(auto)
    driver = ITMOpenWorldLangChainDriver(
        extra_tools={'kaleido': toolkit}, apply_action_filtering=False)
    completed = 0
    for scenario_id in ('desert', 'urban'):
        for attribute in ('merit', 'affiliation'):
            for value in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
                session, _ = session_for()
                session.scenario.id = lambda name=scenario_id: name
                session.alignment_target.kdma_values[0].update(
                    kdma=attribute, value=value)
                tools = {tool.name: tool for tool in driver._build_tools(session)}
                context = toolkit.decision_context(session)
                assert (context is not None) == auto
                if auto:
                    assert_kaleido_target(context, attribute, value)
                text = tools['assess_available_actions'].invoke({
                    'justification': 'Assess the current session target'})
                assert_kaleido_target(text, attribute, value)
                completed += 1
                # Automatic and explicit assessments share one computation
                # within each session, but never reuse a preceding session.
                assert backend.calls == completed
    assert completed == 24
