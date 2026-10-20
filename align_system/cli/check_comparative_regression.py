"""Replay one saved CR tool call and check its full actor context on vLLM.

This command never creates an evaluation session or executes an environment
action. It uses an ``episode_context`` event from a previous agent run, invokes
the real scoring tool once (including its bounded retries), and asks vLLM's
tokenizer whether the resulting actor request fits. The actor does not generate.
"""

import argparse
import json
from pathlib import Path
from time import monotonic
from unittest.mock import patch

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from swagger_client.models import Action, AlignmentTarget, State

from align_system.drivers.itm_open_world_langchain import (
    ACTION_ASSESSMENT_GUIDANCE, ScenarioSession)


class SavedScenario:
    """A read-only scenario; accidental attempts to act fail immediately."""

    def __init__(self, event):
        self.event = event
        self.state = State(**event['full_state'])
        self.actions = [Action(**action) for action in event['choices']]

    def id(self):
        return self.event['scenario_id']

    def get_state(self):
        return self.state

    def get_available_actions(self):
        return self.actions

    def take_action(self, action):
        raise RuntimeError('The CR smoke check cannot execute environment actions.')


class CompletionEvidence(BaseCallbackHandler):
    """Save diagnostics only, without copying hidden reasoning into artifacts."""

    def __init__(self):
        self.completions = []

    def on_llm_end(self, response, **kwargs):
        for group in response.generations:
            for generation in group:
                message = generation.message
                self.completions.append({
                    'reasoning_content_present': bool(
                        message.additional_kwargs.get('reasoning_content')
                        or message.additional_kwargs.get('reasoning')),
                    'thinking_tag_present': '<think>' in str(message.content),
                    'finish_reason': message.response_metadata.get('finish_reason'),
                    'usage': message.usage_metadata,
                })


def _saved_episode(path, scenario_id=None):
    with path.open() as events:
        for line in events:
            event = json.loads(line)
            if (event.get('event') == 'episode_context'
                    and (scenario_id is None or event['scenario_id'] == scenario_id)):
                return event
    raise ValueError('No matching saved episode_context event found.')


def _tool_exchange(name, arguments, result, call_id):
    return [
        AIMessage(content='', tool_calls=[{
            'name': name, 'args': arguments, 'id': call_id}]),
        ToolMessage(content=result, name=name, tool_call_id=call_id),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--events', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--model', default='Qwen/Qwen3.8-27B-FP8')
    parser.add_argument('--scenario-id')
    parser.add_argument('--bert-device', default='cpu')
    parser.add_argument('--scorer-max-tokens', type=int, default=8192)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    event = _saved_episode(args.events, args.scenario_id)
    config_dir = Path(__file__).resolve().parents[1] / 'configs'
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        config = compose(config_name='action_based', overrides=[
            '+experiment=phase2_feb_openworld/'
            'phase2_langchain_agent_ow_comparative_regression_live',
            'driver/chat_model=vllm_qwen38_27b',
            f'driver.chat_model.model={args.model}',
            f'driver.chat_model.base_url={args.base_url}',
            'driver.max_messages_in_context=12',
            'driver.extra_tools.comparative_regression.auto_assess=false',
            'driver.extra_tools.comparative_regression.assessment_mode=ranked',
            'driver.extra_tools.comparative_regression.icl.'
            f'icl_generator_partial.scorer.device={args.bert_device}',
            '++driver.extra_tools.comparative_regression.'
            f'generation_kwargs.max_tokens={args.scorer_max_tokens}',
            '++driver.extra_tools.comparative_regression.generation_kwargs.'
            'extra_body.chat_template_kwargs.enable_thinking=false',
        ])
    # Instantiate only the driver; the configured live interface is never built.
    driver = instantiate(config.driver)
    toolkit = driver.extra_tools['comparative_regression']
    scenario = SavedScenario(event)
    target = AlignmentTarget(
        id=event['alignment_target_id'], kdma_values=event['target'])
    session = ScenarioSession(
        scenario, target, driver.apply_action_filtering,
        driver.sort_available_actions, lambda record: None)
    tools = driver._build_tools(session)
    by_name = {tool.name: tool for tool in tools}
    reminder = toolkit.decision_context(session)
    if toolkit.auto_assess or session.assessments:
        raise AssertionError('CR must remain optional in this smoke check.')

    observation = by_name['observe_environment'].invoke({})
    listing = by_name['list_available_actions'].invoke({})
    completions = CompletionEvidence()
    toolkit.chat_model.callbacks = [completions]
    scorer_requests = []
    model_type = type(toolkit.chat_model)
    original_count = model_type._count_prompt

    def count_scorer(model, payload):
        count, limit, output = original_count(model, payload)
        body = {**payload, **(payload.get('extra_body') or {})}
        scorer_requests.append({
            'prompt_tokens': count, 'context_limit': limit,
            'output_tokens': output,
            'message_count': len(body.get('messages', [])),
            'thinking_enabled': body.get('chat_template_kwargs', {}).get(
                'enable_thinking'),
        })
        return count, limit, output

    started = monotonic()
    justification = 'Compare the current concrete candidates with the alignment target.'
    with patch.object(model_type, '_count_prompt', count_scorer):
        result = by_name['score_available_actions'].invoke({
            'justification': justification})
    elapsed = monotonic() - started
    record = session.assessments['comparative_regression']
    (args.output_dir / 'assessment.json').write_text(
        json.dumps(record, indent=2, default=str) + '\n')
    (args.output_dir / 'tool_response.txt').write_text(result + '\n')

    system = '\n\n'.join([
        driver.system_prompt, ACTION_ASSESSMENT_GUIDANCE,
        *(provider.prompt_hint for provider in driver.extra_tools.values()
          if getattr(provider, 'prompt_hint', None)),
    ])
    tag_context = session.tag_memory.context(session.current_state)
    if tag_context:
        system += '\n\n' + tag_context
    opening = ('A new scenario has started.  Observe the environment and '
               'handle the casualties until the scenario is complete.')
    if driver.target_context is not None:
        opening += '\n\n' + driver.target_context.render(target)
    messages = [SystemMessage(content=system), HumanMessage(content=opening)]
    messages.extend(_tool_exchange('observe_environment', {}, observation, 'observe'))
    messages.extend(_tool_exchange('list_available_actions', {}, listing, 'list'))
    messages.extend(_tool_exchange(
        'score_available_actions', {'justification': justification}, result, 'cr'))
    actor = driver._resolve_chat_model()
    binding = actor.bind_tools(tools)
    payload = actor._get_request_payload(messages, **binding.kwargs)
    actor_tokens, limit, reserved = actor._count_prompt(payload)
    actor_fits = actor_tokens + reserved + actor.context_safety_margin <= limit
    (args.output_dir / 'actor_request.json').write_text(
        json.dumps(payload, indent=2, default=str) + '\n')

    example_counts = {
        name: len(examples) for name, examples in record['icl_examples'].items()}
    examples_visible = all(
        example['prompt'] in result and str(example['response']) in result
        for examples in record['icl_examples'].values() for example in examples)
    scorer = toolkit.icl.icl_generator_partial.keywords['scorer']
    expected_attributes = toolkit._attribute_names(record['targets'])
    complete = bool(record['rows']) and all(
        name in row['scores'] for row in record['rows']
        for name in expected_attributes)
    summary = {
        'scenario_id': scenario.id(), 'targets': record['targets'],
        'elapsed_seconds': round(elapsed, 2),
        'candidate_count': len(record['rows']),
        'expected_attributes': expected_attributes,
        'complete_scores': complete,
        'failed': record.get('failed'),
        'missing_scores': record.get('missing_scores'),
        'scoring_attempts': record.get('scoring_attempts'),
        'selected_examples_per_attribute': example_counts,
        'all_selected_examples_visible': examples_visible,
        'bert_device': scorer.device, 'bert_batch_size': scorer.batch_size,
        'tool_response_characters': len(result),
        'scorer_requests': scorer_requests,
        'completions': completions.completions,
        'actor_prompt_tokens': actor_tokens,
        'actor_output_tokens_reserved': reserved,
        'actor_safety_margin': actor.context_safety_margin,
        'server_context_limit': limit, 'actor_context_fits': actor_fits,
        'environment_actions_executed': session.n_actions,
        'auto_assess': toolkit.auto_assess,
        'initial_cr_reminder_present': bool(reminder),
    }
    (args.output_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2, default=str) + '\n')
    print(json.dumps(summary, indent=2, default=str))
    if (not complete or record.get('failed') or not actor_fits or not examples_visible
            or set(example_counts) != set(expected_attributes)
            or any(count != 20 for count in example_counts.values())
            or not scorer_requests
            or not completions.completions
            or any(request['thinking_enabled'] is not False
                   for request in scorer_requests)
            or any(item['reasoning_content_present'] or item['thinking_tag_present']
                   for item in completions.completions)):
        raise SystemExit('CR smoke check failed; inspect the saved artifacts.')


if __name__ == '__main__':
    main()
