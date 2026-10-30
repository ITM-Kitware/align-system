"""Read-only recovery plans from the agent journal and successful-action trace."""
from dataclasses import dataclass, field
import json
from pathlib import Path

from align_system.utils.agent_actions import action_identity


class RecoveryError(ValueError):
    """Recovery cannot establish a consistent replay boundary."""


def read_journal(path):
    events = []
    truncated = False
    for index, line in enumerate(Path(path).read_text().splitlines(keepends=True)):
        try:
            event = json.loads(line)
        except ValueError:
            if not line.endswith('\n'):
                truncated = True
                break
            raise RecoveryError(f'Corrupt journal at line {index + 1}') from None
        if (not isinstance(event, dict) or event.get('schema_version') != 1
                or type(event.get('sequence')) is not int or event['sequence'] != index
                or not isinstance(event.get('run_id'), str)
                or not isinstance(event.get('event'), str)):
            raise RecoveryError(f'Invalid journal envelope at line {index + 1}')
        if events and (event['run_id'] != events[0]['run_id']
                       or events[-1]['event'] in {'run_finished', 'run_aborted'}
                       or event['event'] == 'run_started'):
            raise RecoveryError('Journal mixes runs or continues after termination')
        events.append(event)
    if not events or events[0]['event'] != 'run_started':
        raise RecoveryError('Journal is missing run_started')
    return events, truncated


@dataclass
class ReplayAction:
    action: dict
    pre_state: dict
    sequence: int
    choice_info: dict = field(default_factory=dict)
    post_state: dict | None = None
    recovered_completion: bool = False


@dataclass
class ReplayEpisode:
    episode_id: str
    scenario_id: str
    events: list = field(default_factory=list)
    actions: list[ReplayAction] = field(default_factory=list)
    context: dict = field(default_factory=dict)
    completed: bool = False

    @property
    def target_id(self):
        return self.context.get('alignment_target_id')

    @property
    def confirmed_actions(self):
        return sum(not action.recovered_completion for action in self.actions)


class ReplayPlan:
    def __init__(self, directory):
        self.directory = Path(directory).expanduser().resolve()
        events, self.truncated_journal = read_journal(self.directory / 'agent_events.jsonl')
        self.run_id = events[0]['run_id']
        self.finished = events[-1]['event'] == 'run_finished'
        self.episodes = []
        by_id = {}
        for event in events:
            eid = event.get('episode_id')
            if event['event'] == 'episode_started':
                if not isinstance(eid, str) or eid in by_id:
                    raise RecoveryError('Invalid or repeated episode ID')
                episode = ReplayEpisode(eid, event['scenario_id'])
                self.episodes.append(episode)
                by_id[eid] = episode
            if eid is not None:
                if eid not in by_id:
                    raise RecoveryError('Journal event refers to an unknown episode')
                by_id[eid].events.append(event)

        records = json.loads((self.directory / 'input_output.json').read_text())
        if not isinstance(records, list):
            raise RecoveryError('input_output.json must contain an action list')
        indexed = {}
        for row in records:
            info = row.get('choice_info', {})
            binding = info.get('event_log', {})
            key = (binding.get('episode_id'),
                   info.get('langchain_agent', {}).get('n_actions_taken_in_scenario'))
            if (binding.get('run_id') != self.run_id or key in indexed
                    or key[0] not in by_id or type(key[1]) is not int):
                raise RecoveryError('Invalid action trace binding or duplicate decision')
            indexed[key] = row
        for episode in self.episodes:
            self._load_episode(episode, indexed)
        if indexed:
            raise RecoveryError('Action trace contains actions absent from the journal')
        if any(not ep.completed for ep in self.episodes[:-1]):
            raise RecoveryError('An unfinished episode precedes another episode')
        self.checkpoint = self._read_checkpoint()
        self.position = 0

    def _load_episode(self, episode, indexed):
        attempt = None
        tool_call = None
        for event in episode.events:
            kind = event['event']
            if kind == 'episode_context':
                if episode.context:
                    raise RecoveryError('Repeated episode_context')
                episode.context = event
            elif kind == 'tool_requested':
                tool_call = event.get('tool_call')
            elif kind == 'action_attempted':
                attempt = event
            elif kind == 'action_succeeded':
                index = event.get('decision_index')
                if index != episode.confirmed_actions or episode.completed:
                    raise RecoveryError('Noncontiguous successful actions')
                row = indexed.pop((episode.episode_id, index), None)
                if row is not None:
                    if (row['input']['scenario_id'] != episode.scenario_id
                            or row['input'].get('alignment_target_id') != episode.target_id
                            or action_identity(row['output']['action']) != event['action']):
                        raise RecoveryError('Action trace does not match the journal')
                    action, pre = row['output']['action'], row['input']['full_state']
                    info = row['choice_info']
                elif (attempt and attempt.get('full_action') is not None
                      and attempt.get('pre_action_state') is not None
                      and attempt.get('decision_index') == index
                      and attempt['action'] == event['action']):
                    # The durable success event can precede the trace-file update.
                    action, pre, info = attempt['full_action'], attempt['pre_action_state'], {}
                else:
                    raise RecoveryError('Successful action is missing its full payload/state')
                episode.actions.append(ReplayAction(
                    action, pre, event['sequence'], info, event.get('full_state')))
                attempt = None
            elif kind == 'scenario_completion_recovered':
                if (event.get('source') != 'action' or not event.get('action')
                        or not event.get('pre_action_state')
                        or not (event.get('final_state') or {}).get('scenario_complete')
                        or event.get('decision_index') != episode.confirmed_actions
                        or attempt is None or attempt.get('action') != event['action']):
                    raise RecoveryError('Recovered completion lacks a replayable terminal action')
                action = dict((attempt or {}).get('full_action') or event['action'])
                if tool_call:
                    action.setdefault('justification', tool_call.get('args', {}).get('justification'))
                episode.actions.append(ReplayAction(
                    action, event['pre_action_state'], event['sequence'],
                    post_state=event['final_state'], recovered_completion=True))
                episode.completed = True
            elif kind == 'episode_finished':
                episode.completed = event.get('completed') is True
        if not episode.context or not episode.context.get('full_state'):
            raise RecoveryError('Episode lacks its initial state and target context')
        if episode.actions and (episode.actions[-1].post_state or {}).get('scenario_complete'):
            episode.completed = True

    def _read_checkpoint(self):
        path = self.directory / 'agent_checkpoint.json'
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        if (not isinstance(data, dict) or data.get('schema_version') != 1
                or data.get('run_id') != self.run_id
                or data.get('episode_id') not in {ep.episode_id for ep in self.episodes}
                or type(data.get('n_actions')) is not int
                or not isinstance(data.get('full_state'), dict)
                or not isinstance(data.get('messages'), list)):
            raise RecoveryError('Conversation checkpoint belongs to a different run or schema')
        return data

    def next_episode(self, scenario_id, target_id):
        if self.position == len(self.episodes):
            return None
        episode = self.episodes[self.position]
        if (scenario_id, target_id) != (episode.scenario_id, episode.target_id):
            raise RecoveryError(
                f'Replay episode {self.position + 1} expected '
                f'{episode.scenario_id} / {episode.target_id}, got {scenario_id} / {target_id}')
        self.position += 1
        return episode

    def assert_consumed(self):
        if self.position != len(self.episodes):
            raise RecoveryError('TA3 ended before all recorded episodes could be replayed')

    def summary(self):
        return {
            'source': str(self.directory), 'run_id': self.run_id,
            'source_finished': self.finished, 'truncated_journal': self.truncated_journal,
            'episodes': [{'scenario_id': ep.scenario_id, 'alignment_target_id': ep.target_id,
                          'confirmed_actions': ep.confirmed_actions,
                          'recovered_terminal_actions': len(ep.actions) - ep.confirmed_actions,
                          'completed': ep.completed} for ep in self.episodes],
            'conversation_checkpoint': self.checkpoint is not None,
        }
