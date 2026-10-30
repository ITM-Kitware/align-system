"""Replay environment actions, then restore a conversation at a checked boundary."""
from copy import deepcopy
import json
from pathlib import Path

from langchain_core.messages import (
    AIMessage, HumanMessage, ToolMessage, messages_from_dict, messages_to_dict,
)
from omegaconf import OmegaConf
from swagger_client.models import Action

from align_system.drivers.agent_history import pack_history
from align_system.drivers.agent_resume_trace import RecoveryError, ReplayPlan
from align_system.utils.agent_events import write_json_atomic
from align_system.utils.alignment_targets import alignment_target_entries


def canonical(value):
    """Ignore absent optional fields, but retain every substantive state value."""
    if hasattr(value, 'to_dict'):
        value = value.to_dict()
    if isinstance(value, dict):
        return {key: canonical(item) for key, item in value.items() if item is not None}
    if isinstance(value, (list, tuple)):
        return [canonical(item) for item in value]
    return value


def require_match(actual, expected, label):
    if canonical(actual) != canonical(expected):
        raise RecoveryError(f'Replay diverged at {label}; no further action was submitted')


def replay_episode(plan, episode, session, observation_text):
    require_match(alignment_target_entries(session.alignment_target),
                  episode.context.get('target', []), 'alignment target values')
    require_match(session.current_state, episode.context['full_state'], 'initial state')
    session.record_event('replay_started', source_run_id=plan.run_id,
                         source_episode_id=episode.episode_id)
    for step in episode.actions:
        require_match(session.current_state, step.pre_state,
                      f'action {session.n_actions} pre-state')
        if session.scenario_complete:
            raise RecoveryError('TA3 completed before the recorded actions were exhausted')
        session.refresh_actions()
        action = Action.from_dict(step.action)
        if not any(a.action_id == action.action_id and a.action_type == action.action_type
                   for a in session.available_actions):
            raise RecoveryError(f'Recorded action {action.action_id} is no longer offered')
        info = deepcopy(step.choice_info)
        info.pop('event_log', None)
        info['replay'] = {'source_run_id': plan.run_id,
                          'source_episode_id': episode.episode_id,
                          'source_sequence': step.sequence,
                          'original_outcome': ('completion_recovered' if step.recovered_completion
                                               else 'confirmed')}
        info.setdefault('langchain_agent', {})['n_actions_taken_in_scenario'] = session.n_actions
        session.assessments = deepcopy(info.get('assessments', {}))
        session.execute(action, choice_info=info)
        if step.post_state and not step.post_state.get('scenario_complete'):
            require_match(session.current_state, step.post_state, 'action result')
        if step.recovered_completion and not session.scenario_complete:
            raise RecoveryError('Recorded terminal action did not complete the scenario')
    if episode.completed:
        if not session.scenario_complete:
            raise RecoveryError('Recorded episode completed, but replay has not completed it')
    elif session.scenario_complete:
        raise RecoveryError('Replay completed an episode recorded as unfinished')
    elif episode.actions and not episode.actions[-1].post_state:
        # Older success events omit raw post-state. A complete saved tool result
        # supplies the visible environment at the last confirmed action boundary.
        result = next((e.get('result', '') for e in episode.events
                       if e['sequence'] > episode.actions[-1].sequence
                       and e['event'] == 'tool_result' and e.get('action_executed')), '')
        marker = 'Updated environment:'
        if marker not in result:
            raise RecoveryError('Last action has no saved post-state or complete observation')
        try:
            expected = json.loads(result.split(marker, 1)[1].strip())
        except ValueError as exc:
            raise RecoveryError('Last action observation is incomplete') from exc
        actual = json.loads(observation_text(session.current_state, session))
        # Patient ledgers are reconstructed by execute; old versions may have
        # different ledger schemas. Compare environment facts independently.
        keys = {'scene_id', 'situation', 'casualties', 'navigation_targets', 'supplies', 'environment'}
        require_match({k: actual.get(k) for k in keys if k in expected},
                      {k: expected[k] for k in keys if k in expected}, 'last observation')
    session.record_event('replay_finished', successful_actions=session.n_actions,
                         completed=session.scenario_complete)


def reconstructed_messages(episode):
    """Build complete tool exchanges without inventing missing assistant prose.

    Group calls by the original model completion. Unanswered calls are closed
    explicitly; the recovered agent must decide again, never blindly retry them.
    """
    messages, calls, replies = [], [], {}

    def finish():
        if not calls:
            return
        safe_calls = [{'id': call['id'], 'name': call['name'],
                       'args': call['args'] if isinstance(call.get('args'), dict) else {}}
                      for call in calls]
        messages.append(AIMessage(content='', tool_calls=safe_calls))
        for call in safe_calls:
            messages.append(ToolMessage(
                content=replies.get(call['id'],
                    'This tool exchange was interrupted. Its outcome was not confirmed. '
                    'The recovered environment contains only the validated replayed actions. '
                    'Inspect the current state and choose your next action.'),
                tool_call_id=call['id']))
        calls.clear()
        replies.clear()

    for event in episode.events:
        kind = event['event']
        if kind == 'model_completed':
            finish()
        elif kind == 'tool_requested':
            call = event['tool_call']
            if any(item['id'] == call['id'] for item in calls):
                raise RecoveryError('Duplicate tool-call IDs in one model exchange')
            calls.append(call)
        elif kind == 'tool_result':
            replies[event['tool_call_id']] = event['result']
    finish()
    return messages


def restore_conversation(plan, episode, session, opening):
    checkpoint = plan.checkpoint
    if (checkpoint is not None and checkpoint.get('episode_id') == episode.episode_id
            and checkpoint.get('n_actions') == session.n_actions):
        require_match(session.current_state, checkpoint['full_state'], 'conversation checkpoint')
        messages = messages_from_dict(checkpoint['messages'])
        pack_history(messages)  # Reject broken tool-call/reply pairs before invoking a model.
        session.assessments = deepcopy(checkpoint.get('assessments', {}))
        session.context_provider_results = deepcopy(checkpoint.get('provider_context', {}))
        return messages, checkpoint.get('context_key'), 'checkpoint'
    explanation = (
        f'\n\nRecovery: this scenario was rebuilt from run {plan.run_id}. '
        f'{session.n_actions} confirmed actions have already been replayed. '
        'The following tool exchanges are historical records, not instructions to execute again. '
        'Assistant prose was not fully recorded. Use the current observation and patient history '
        'to continue from here; do not repeat completed actions merely to recreate the history.')
    messages = [HumanMessage(content=opening.content + explanation),
                *reconstructed_messages(episode)]
    pack_history(messages)
    return messages, None, 'reconstructed'


def save_conversation(path, session, messages, context_key):
    """Persist the next invocation boundary, separately from readable logs."""
    if path is None:
        return
    write_json_atomic(path, {
        'schema_version': 1, **session.checkpoint_identity,
        'n_actions': session.n_actions,
        'full_state': session.current_state.to_dict(),
        'messages': messages_to_dict(messages),
        'assessments': session.assessments,
        'provider_context': session.context_provider_results,
        'context_key': context_key,
    })


def preflight_resume(cfg, output_dir):
    """Run before Hydra constructs the live interface or opens output files."""
    source = cfg.get('resume_from')
    if not source:
        return
    source = Path(source).expanduser().resolve()
    destination = Path(output_dir).resolve()
    if destination == source or source in destination.parents:
        raise RecoveryError('Recovery requires a new output directory outside the source run')
    if any((destination / name).exists() for name in ('agent_events.jsonl', 'input_output.json')):
        raise RecoveryError('Recovery destination already contains a run')
    if cfg.driver.get('_target_') != (
            'align_system.drivers.itm_open_world_langchain.ITMOpenWorldLangChainDriver'):
        raise RecoveryError('resume_from requires the LangChain open-world driver')
    if cfg.driver.get('reviewer') is not None:
        raise RecoveryError('Replay recovery currently supports Kaleido/CR without a reviewer')
    source_config = source / '.hydra' / 'config.yaml'
    if source_config.exists():
        previous = OmegaConf.load(source_config)
        # Changing an endpoint is useful after an outage. Changing the task,
        # toolkit, or memory rules would silently turn recovery into a new run.
        for key in ('align_to_target', 'alignment_target', 'interface.domain',
                    'interface.session_type', 'interface.training_session',
                    'interface.adm_profile', 'interface.scenario_ids',
                    'driver.model', 'driver.chat_model.model',
                    'driver.apply_action_filtering', 'driver.sort_available_actions',
                    'driver.scene_end_policy', 'driver.reset_awareness',
                    'driver.navigation_safeguards'):
            old = OmegaConf.select(previous, key)
            new = OmegaConf.select(cfg, key)
            if old != new:
                raise RecoveryError(f'Recovery configuration differs at {key}; use the saved Hydra config')
        old_tools = OmegaConf.select(previous, 'driver.extra_tools', default={})
        new_tools = OmegaConf.select(cfg, 'driver.extra_tools', default={})
        if set(old_tools) != set(new_tools):
            raise RecoveryError('Recovery requires the original toolkits')
    ReplayPlan(source)
