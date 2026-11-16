"""Offline role-alignment checks through proposal, review, and final execution."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import pytest
from swagger_client.models import Action, ActionTypeEnum

from align_system.algorithms.agents.agent_review import ActionReviewer
from align_system.algorithms.agents.langchain_agentic_adm import LangChainAgenticADM
from align_system.drivers.itm_open_world_agentic import (
    ITMOpenWorldAgenticDriver, ScenarioSession,
)


TARGET_ID = 'target-only-in-alignment-context'
TARGET_VALUE = 0.8
DEFINITION_SENTINEL = 'merit-definition-only-in-alignment-context'
DECISION_GUIDANCE = 'DECISION_ALIGNMENT_GUIDANCE_SENTINEL'
REVIEW_GUIDANCE = 'REVIEW_ALIGNMENT_GUIDANCE_SENTINEL'
REVIEW_DETAIL_GUIDANCE = 'REVIEW_ALIGNMENT_DETAIL_SENTINEL'
CURRENT_PREFIX = 'Current visible state, patient history, and cached provider context:\n'
GUIDE_PREFIX = '\nCitation guides (path lookup only; not evidence):\n'
EVIDENCE_PREFIX = '\nEvidence namespaces frozen for this decision-model invocation:\n'


class WireObject(SimpleNamespace):
    def to_dict(self):
        def convert(value):
            if hasattr(value, 'to_dict'):
                return value.to_dict()
            if isinstance(value, list):
                return [convert(item) for item in value]
            return value
        return {key: convert(value) for key, value in vars(self).items()}


class RecordingScenario:
    def __init__(self):
        self.state = WireObject(
            meta_info=WireObject(scene_id='scene-a'), scenario_complete=False,
            unstructured='A visible casualty needs assessment.', supplies=None,
            characters=[WireObject(id='patient-a', name='Patient A', unseen=False,
                                   tag=None, unstructured='Visible casualty.')])
        self.actions = [Action(
            action_id='check-a', action_type=ActionTypeEnum.CHECK_VITALS,
            character_id='patient-a', unstructured='Assess the casualty.')]
        self.submissions = []

    def id(self):
        return 'scenario-a'

    def get_state(self):
        return deepcopy(self.state)

    def get_available_actions(self):
        return deepcopy(self.actions)

    def take_action(self, action):
        self.submissions.append(deepcopy(action))
        result = deepcopy(self.state)
        result.scenario_complete = True
        return result


class ScriptedModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def bind_tools(self, tools):
        self.tools = tools
        return self

    def invoke(self, messages, **kwargs):
        self.calls.append(deepcopy(messages))
        return next(self.responses)


def attributes():
    return {
        'medical': {'name': 'Medical urgency', 'description': 'Visible medical need.',
                    'factor': 100, 'score_examples': 'Urgent need scores 100.'},
        'merit': {'name': 'Merit', 'description': DEFINITION_SENTINEL,
                  'factor': 100, 'score_examples': 'Visible intent affects merit.'},
    }


def decision(*, final=False, justification='Assess the visible casualty.',
             evidence_path='/review/proposal_tool/arguments/character_name',
             evidence_value='patient-a'):
    arguments = {'character_name': 'patient-a', 'justification': justification}
    if final:
        arguments['review_response'] = {
            'acknowledged_suggestion': 'keep',
            'response_to_advice': 'I acknowledge the advice and keep this action.',
            'evidence': [{'path': evidence_path, 'value': evidence_value}],
        }
    return AIMessage(content='', tool_calls=[{
        'name': 'check_vitals', 'args': arguments,
        'id': 'final-call' if final else 'proposal-call', 'type': 'tool_call'}])


def review(*, reason='The proposal names a visible casualty.',
           path='/proposal_tool/arguments/character_name', value='patient-a'):
    return AIMessage(content=json.dumps({
        'recommendation': 'keep', 'reason': reason,
        'alternative': None, 'confidence': None, 'estimated_improvement': None,
        'assessment': {'evidence': [{'path': path, 'value': value}]},
    }))


def run_episode(*, mode_kwargs=None, global_alignment=True,
                attribute_definitions=None, review_responses=None,
                decision_responses=None):
    scenario = RecordingScenario()
    target = WireObject(id=TARGET_ID, kdma_values=[{'kdma': 'merit', 'value': TARGET_VALUE}])
    records = []
    session = ScenarioSession(
        scenario, target, False, True, records.append,
        align_to_target=global_alignment, scene_end_policy='auto')
    tools = ITMOpenWorldAgenticDriver()._build_tools(session)
    decision_model = ScriptedModel(decision_responses or [decision(), decision(final=True)])
    reviewer_model = ScriptedModel(review_responses or [review()])
    reviewer = ActionReviewer(
        chat_model=reviewer_model, weighting={'influence': 0.5, 'swap_threshold': 0.15},
        alignment_guidance=REVIEW_GUIDANCE,
        alignment_review_guidance=REVIEW_DETAIL_GUIDANCE)
    adm = LangChainAgenticADM(
        chat_model=decision_model, reviewer=reviewer,
        context_budget={'enabled': False},
        attribute_definitions=(attributes() if attribute_definitions is None
                               else attribute_definitions),
        alignment_guidance=DECISION_GUIDANCE, **(mode_kwargs or {}))
    adm.run_scenario(session, tools)
    assert len(scenario.submissions) == len(records) == 1
    assert session.scenario_complete
    assert records[0]['input']['alignment_target_id'] == TARGET_ID
    assert session.log_context()['target_id'] == TARGET_ID
    assert records[0]['choice_info']['multi_agent']['driver_override'] is None
    return SimpleNamespace(session=session, records=records,
                           decision=decision_model.calls, reviewer=reviewer_model.calls)


def current_context(messages):
    message = next(message for message in reversed(messages)
                   if isinstance(message, HumanMessage)
                   and message.content.startswith(CURRENT_PREFIX))
    return json.loads(message.content.removeprefix(CURRENT_PREFIX))


def final_contexts(messages):
    reminder = messages[-1].content
    before, evidence = reminder.split(EVIDENCE_PREFIX)
    guide = before.split(GUIDE_PREFIX)[1]
    return json.loads(evidence), json.loads(guide)


def assert_role_contexts(result, decision_aligned, reviewer_aligned):
    assert len(result.decision) == 2
    for messages in result.decision:
        assert isinstance(messages[0], SystemMessage)
        assert (DECISION_GUIDANCE in messages[0].content) == decision_aligned
        opening = messages[1].content
        assert (TARGET_ID in opening) == decision_aligned
        assert (DEFINITION_SENTINEL in opening) == decision_aligned
    context = current_context(result.decision[0])
    assert ('alignment_context' in context) == decision_aligned
    assert bool(context['citation_guide']['kdmas']) == decision_aligned
    for messages in result.reviewer:
        assert isinstance(messages[0], SystemMessage)
        assert (REVIEW_GUIDANCE in messages[0].content) == reviewer_aligned
        assert (REVIEW_DETAIL_GUIDANCE in messages[0].content) == reviewer_aligned
        context = json.loads(messages[-1].content)
        assert ('alignment_context' in context) == reviewer_aligned
        assert bool(context['citation_guide']['kdmas']) == reviewer_aligned
        if reviewer_aligned:
            assert context['alignment_context']['target_id'] == TARGET_ID
            assert context['alignment_context']['kdmas'][0]['value'] == TARGET_VALUE
    evidence, guides = final_contexts(result.decision[-1])
    assert ('alignment_context' in evidence['current']) == decision_aligned
    assert ('alignment_context' in evidence['review']) == (decision_aligned and reviewer_aligned)
    assert bool(guides['current'].get('kdmas')) == decision_aligned
    assert bool(guides['review'].get('kdmas')) == (decision_aligned and reviewer_aligned)


@pytest.mark.parametrize('decision_aligned', [False, True])
@pytest.mark.parametrize('reviewer_aligned', [False, True])
def test_independent_alignment_modes_cover_proposal_review_and_final_turn(
        decision_aligned, reviewer_aligned):
    result = run_episode(mode_kwargs={
        'align_decision_to_target': decision_aligned,
        'align_reviewer_to_target': reviewer_aligned,
    })
    assert_role_contexts(result, decision_aligned, reviewer_aligned)
    assert result.session.scene_end_policy == 'agent_decides'
    assert result.records[0]['choice_info']['multi_agent']['calls'] == {
        'decision_agent': 2, 'reviewer': 1}
    # Neutral generated prose makes any target leakage a harness-injection bug.
    if not decision_aligned:
        serialized = json.dumps([message.content for call in result.decision for message in call])
        assert TARGET_ID not in serialized
        assert DEFINITION_SENTINEL not in serialized


def test_omitted_mode_flags_keep_both_agents_aligned():
    default = run_episode()
    explicit = run_episode(mode_kwargs={
        'align_decision_to_target': True, 'align_reviewer_to_target': True})
    assert_role_contexts(default, True, True)
    for role in ('decision', 'reviewer'):
        assert [[message.content for message in call] for call in getattr(default, role)] == [
            [message.content for message in call] for call in getattr(explicit, role)]


def test_global_alignment_false_overrides_both_role_flags():
    result = run_episode(global_alignment=False, attribute_definitions={}, mode_kwargs={
        'align_decision_to_target': True, 'align_reviewer_to_target': True})
    assert_role_contexts(result, False, False)


def test_both_baseline_does_not_require_alignment_definitions():
    result = run_episode(attribute_definitions={}, mode_kwargs={
        'align_decision_to_target': False, 'align_reviewer_to_target': False})
    assert_role_contexts(result, False, False)
    assert result.session.scene_end_policy == 'agent_decides'


@pytest.mark.parametrize(('decision_aligned', 'reviewer_aligned'), [(False, True), (True, False)])
def test_reviewer_repair_preserves_its_independent_alignment_mode(
        decision_aligned, reviewer_aligned):
    result = run_episode(mode_kwargs={
        'align_decision_to_target': decision_aligned,
        'align_reviewer_to_target': reviewer_aligned,
    }, review_responses=[AIMessage(content='not valid JSON'), review()])
    assert_role_contexts(result, decision_aligned, reviewer_aligned)
    assert len(result.reviewer) == 2
    assert 'repair_feedback' not in json.loads(result.reviewer[0][-1].content)
    assert 'repair_feedback' in json.loads(result.reviewer[1][-1].content)
    assert result.records[0]['choice_info']['multi_agent']['calls'] == {
        'decision_agent': 2, 'reviewer': 2}


def test_baseline_decision_can_receive_and_cite_aligned_reviewer_advice():
    reason = f'The reviewer considered {TARGET_ID} when recommending this action.'
    path = '/alignment_context/kdmas/0/value'
    result = run_episode(mode_kwargs={
        'align_decision_to_target': False, 'align_reviewer_to_target': True,
    }, review_responses=[review(reason=reason, path=path, value=TARGET_VALUE)],
        decision_responses=[decision(), decision(
            final=True, evidence_path='/review' + path, evidence_value=TARGET_VALUE)])
    assert_role_contexts(result, False, True)
    assert reason in result.decision[-1][-1].content
    assert '/review' + path in result.decision[-1][-1].content
    metadata = result.records[0]['choice_info']['multi_agent']
    assert metadata['advice']['assessment']['evidence'] == [{'path': path, 'value': TARGET_VALUE}]
    assert metadata['final_review_response_validation']['status'] == 'verified'


def test_baseline_reviewer_still_receives_the_aligned_agents_proposal_rationale():
    rationale = f'The decision agent considered {TARGET_ID} before choosing assessment.'
    result = run_episode(mode_kwargs={
        'align_decision_to_target': True, 'align_reviewer_to_target': False,
    }, decision_responses=[decision(justification=rationale), decision(final=True)])
    assert_role_contexts(result, True, False)
    context = json.loads(result.reviewer[0][-1].content)
    assert context['proposal_tool']['arguments']['justification'] == rationale


def forbid_reviewer(*args, **kwargs):
    pytest.fail('A no-reviewer run must not initialize or invoke a reviewer')


@pytest.mark.parametrize('decision_aligned', [False, True])
def test_no_reviewer_executes_one_decision_with_independent_alignment(
        decision_aligned, monkeypatch):
    monkeypatch.setattr(ActionReviewer, '__init__', forbid_reviewer)
    monkeypatch.setattr(ActionReviewer, 'review', forbid_reviewer)
    scenario = RecordingScenario()
    target = WireObject(id=TARGET_ID, kdma_values=[{'kdma': 'merit', 'value': TARGET_VALUE}])
    records = []
    session = ScenarioSession(
        scenario, target, False, True, records.append,
        align_to_target=True, scene_end_policy='agent_decides')
    tools = ITMOpenWorldAgenticDriver()._build_tools(session)
    model = ScriptedModel([decision()])
    adm = LangChainAgenticADM(
        chat_model=model, reviewer=None, context_budget={'enabled': False},
        # Baseline must not need target definitions even with global alignment on.
        attribute_definitions=attributes() if decision_aligned else {},
        alignment_guidance=DECISION_GUIDANCE,
        align_decision_to_target=decision_aligned)

    adm.run_scenario(session, tools)

    assert adm.reviewer is None
    assert adm.architecture is None
    assert len(model.calls) == len(scenario.submissions) == len(records) == 1
    assert session.scenario_complete
    assert session.scene_end_policy == 'agent_decides'
    assert records[0]['input']['alignment_target_id'] == TARGET_ID
    assert session.log_context()['target_id'] == TARGET_ID
    assert 'multi_agent' not in records[0]['choice_info']
    messages = model.calls[0]
    assert (DECISION_GUIDANCE in messages[0].content) == decision_aligned
    assert (TARGET_ID in messages[1].content) == decision_aligned
    context = current_context(messages)
    assert ('alignment_context' in context) == decision_aligned
    assert bool(context['citation_guide']['kdmas']) == decision_aligned
    if decision_aligned:
        assert context['alignment_context']['target_id'] == TARGET_ID
        assert context['alignment_context']['kdmas'][0]['value'] == TARGET_VALUE
    else:
        serialized = json.dumps([message.content for message in messages])
        assert TARGET_ID not in serialized
        assert DEFINITION_SENTINEL not in serialized
    for tool in model.tools:
        if tool.name in session.action_tool_names:
            assert 'review_response' not in tool.args_schema.model_fields


@pytest.mark.parametrize(('preset', 'decision_aligned'), [
    ('qwen_baseline_no_reviewer', False),
    ('qwen_aligned_no_reviewer', True),
])
def test_no_reviewer_presets_replace_inherited_reviewer_with_null(
        preset, decision_aligned, monkeypatch):
    monkeypatch.setattr(ActionReviewer, '__init__', forbid_reviewer)
    monkeypatch.setattr(ActionReviewer, 'review', forbid_reviewer)
    monkeypatch.setenv('QWEN_MODEL_REVISION', 'a' * 40)
    configs = Path(__file__).resolve().parents[1] / 'configs'
    with initialize_config_dir(version_base=None, config_dir=str(configs)):
        cfg = compose(config_name='action_based', overrides=[
            '+experiment=phase2_feb_openworld/qwen_ow_agentic_reviewer/' + preset])

    assert cfg.adm.instance.reviewer is None
    assert cfg.adm.instance.align_decision_to_target is decision_aligned
    assert cfg.align_to_target is True
    assert cfg.driver.scene_end_policy == 'agent_decides'
    model_config = cfg.adm.instance.chat_model
    assert model_config._convert_ == 'all'
    assert model_config.extra_body.chat_template_kwargs.reasoning_effort == 'low'
    adm = instantiate(cfg.adm.instance)
    assert adm.reviewer is None
    assert adm.align_decision_to_target is decision_aligned
    assert adm.load_model().extra_body['chat_template_kwargs']['reasoning_effort'] == 'low'
