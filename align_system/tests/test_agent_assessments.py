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

from align_system.agent_tools.comparative_regression import (
    ComparativeRegressionToolkit, _choice_keys)
from align_system.data_models.attribute import Attribute
from align_system.drivers.itm_open_world_langchain import (
    ITMOpenWorldLangChainDriver, ScenarioSession)
from align_system.prompt_engineering.outlines_prompts import (
    ComparativeRegressionPrompt, ComparativeRegressionSchema,
    ComparativeRegressionSystemPromptWithTemplate)
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


class FakeScorer:
    def __init__(self):
        self.calls = 0
        self.fail = False

    def with_structured_output(self, schema, method):
        assert method == 'json_schema'
        self.schema = schema
        return self

    def batch(self, messages):
        self.calls += 1
        self.messages = messages
        if self.fail:
            raise RuntimeError('scorer unavailable')
        return [{key: {'score': 80, 'explanation': 'Evidence explanation'}
                 for key in self.schema['properties']}
                for _ in messages]


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


def toolkit_for(kind, mode='ranked', auto=True, **kwargs):
    assert kind == 'comparative_regression'
    return ComparativeRegressionToolkit(
        FakeScorer(), {'affiliation': Attribute(
            name='affiliation focus', kdma='affiliation', factor=100,
            description='Favor patients in your group', score_examples='A soldier scores 100.')},
        lambda scenario_state: scenario_state.unstructured,
        ComparativeRegressionPrompt(),
        ComparativeRegressionSchema({'affiliation focus': 100}),
        system_prompt_template=ComparativeRegressionSystemPromptWithTemplate(),
        target_attribute_names_override=['*'], assessment_mode=mode,
        auto_assess=auto, **kwargs)


@pytest.mark.parametrize('mode', ['evidence', 'ranked'])
@pytest.mark.parametrize('auto', [True, False])
def test_cr_modes_deliver_target_and_record_evidence(mode, auto):
    session, records = session_for()
    toolkit = toolkit_for('comparative_regression', mode, auto)
    tools = toolkit.build_tools(session)
    context = toolkit.decision_context(session)
    if auto:
        assert context is not None
    else:
        assert 'CR REMINDER' in context
        assert session.assessments == {}
    text = tools[0].invoke({'justification': 'Assess choices'})
    assert 'affiliation=0.80' in text
    assert 'action_id' in text
    assessment = session.assessments['comparative_regression']
    assert assessment['mode'] == mode
    if mode == 'evidence':
        assert 'Evidence explanation' in text
        assert 'distance' not in text.lower()
        assert 'Closest' not in text
        assert 'closest' not in toolkit.prompt_hint
        assert 'avg_distance' not in assessment['rows'][0]
        assert 'distance' not in assessment['rows'][0]['scores']['affiliation']
    else:
        assert 'Closest to target overall:' in text
        assert assessment['rows'][0]['avg_distance'] == pytest.approx(0)
    driver = ITMOpenWorldLangChainDriver(apply_action_filtering=False)
    executor = {t.name: t for t in driver._build_tools(session)}['execute_available_action']
    executor.invoke({'action_id': 'b', 'justification': 'My choice'})
    assert records[0]['output']['action']['action_id'] == 'b'
    assert records[0]['choice_info']['assessments']['comparative_regression'] == assessment
    assert records[0]['choice_info']['langchain_agent']['decision_source'] == 'agent'
    assert session.assessments == {}
    json.dumps(records)


def test_cr_cache_respects_actions_target_and_scenario():
    toolkit = toolkit_for('comparative_regression', 'evidence')
    model = toolkit.chat_model
    session, _ = session_for()
    toolkit.build_tools(session)
    toolkit.assessment_text(session)
    toolkit.assessment_text(session)
    assert model.calls == 1
    session.scenario.actions[0].parameters = {'message': 'changed'}
    toolkit.assessment_text(session)
    assert model.calls == 2
    session.alignment_target.kdma_values[0]['value'] = 0.2
    assert 'affiliation=0.20' in toolkit.assessment_text(session)
    assert model.calls == 3
    other, _ = session_for()
    toolkit.build_tools(other)
    toolkit.assessment_text(other)
    assert model.calls == 4


def test_regression_failure_is_logged_and_retried():
    toolkit = toolkit_for('comparative_regression', 'evidence')
    session, _ = session_for()
    toolkit.chat_model.fail = True
    assert 'scoring failed for: affiliation' in toolkit.assessment_text(session)
    assert session.assessments['comparative_regression']['failed'] == ['affiliation']
    toolkit.chat_model.fail = False
    assert 'Evidence explanation' in toolkit.assessment_text(session)
    assert session.assessments['comparative_regression']['failed'] == []
    assert toolkit.chat_model.calls == 4


def test_choice_keys_are_short_and_unique_even_with_duplicate_descriptions():
    session, _ = session_for()
    actions = [action('a', character_id='p1', parameters={'treatment': 'Tourniquet'}),
               action('b', unstructured='Same description (2)'), action('c')]
    keys = _choice_keys(actions, session.current_state)
    assert keys == ['C01', 'C02', 'C03']


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


def test_invalid_cr_mode_rejected():
    with pytest.raises(ValueError, match='assessment_mode'):
        toolkit_for('comparative_regression', 'unknown')


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


@pytest.mark.parametrize('mode', ['ranked', 'evidence'])
def test_cr_hydra_configs_instantiate_with_fake_backend(mode):
    config_dir = str(Path(__file__).resolve().parents[1] / 'configs')
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name='action_based', overrides=[
            '+experiment=phase2_feb_openworld/phase2_langchain_agent_ow_comparative_regression_live',
            f'driver.extra_tools.comparative_regression.assessment_mode={mode}',
            'driver.extra_tools.comparative_regression.auto_assess=false'])
        toolkit = instantiate(cfg.driver.extra_tools.comparative_regression, chat_model=FakeScorer())
        assert toolkit.assessment_mode == mode
        assert not toolkit.auto_assess


def test_agent_loop_refreshes_cr_evidence_after_each_action():
    from langchain_core.messages import AIMessage

    session, records = session_for()
    original_take_action = session.scenario.take_action

    def take_action(selected):
        state = original_take_action(selected)
        state.scenario_complete = len(session.scenario.executed) == 2
        return state

    session.scenario.take_action = take_action

    class Agent:
        def __init__(self):
            self.contexts = []

        def bind_tools(self, tools):
            assert 'execute_available_action' in [t.name for t in tools]
            return self

        def invoke(self, messages):
            self.contexts.append(deepcopy(messages))
            return AIMessage(content='', tool_calls=[{
                'id': f'call-{len(self.contexts)}',
                'name': 'execute_available_action',
                'args': {'action_id': 'b', 'justification': 'Interpreted the evidence'}}])

    agent = Agent()
    toolkit = toolkit_for('comparative_regression', 'evidence')
    driver = ITMOpenWorldLangChainDriver(chat_model=agent,
                                       extra_tools={'comparative_regression': toolkit},
                                       apply_action_filtering=False)
    driver._run_agent_loop(session)
    assert session.scenario_complete
    assert len(agent.contexts) == len(records) == 2
    for i, record in enumerate(records):
        evidence = record['choice_info']['assessments']['comparative_regression']
        assert evidence['n_actions'] == i
        assert 'Evidence explanation' in evidence['text']
        assert any(evidence['text'] in str(message.content)
                   for message in agent.contexts[i])
        assert 'closest' not in str(agent.contexts[i][0].content).lower()
    assert toolkit.chat_model.calls == 2


def test_rejected_action_retains_assessment_until_success():
    session, records = session_for()
    toolkit = toolkit_for('comparative_regression', 'evidence')
    toolkit.assessment_text(session)
    original_take_action = session.scenario.take_action

    class Rejected(Exception):
        status = 400

    def reject(selected):
        raise Rejected('Action rejected')

    session.scenario.take_action = reject
    driver = ITMOpenWorldLangChainDriver(apply_action_filtering=False)
    result = driver._perform_typed_action(session, ActionTypeEnum.SEARCH, 'Search', action_id='a')
    assert 'rejected' in result
    assert session.assessments and not records
    session.scenario.take_action = original_take_action
    driver._perform_typed_action(session, ActionTypeEnum.SEARCH, 'Search', action_id='a')
    assert records[0]['choice_info']['assessments']['comparative_regression']['n_actions'] == 0
    assert not session.assessments


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
