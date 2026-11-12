"""Validate episode boundaries before recovery submits environment actions."""
import json

import pytest

from align_system.drivers.agent_resume_trace import RecoveryError, read_journal


def write_journal(path, entries):
    events = [dict(schema_version=1, sequence=index, run_id='run',
                   event=kind, episode_id=episode)
              for index, (kind, episode) in enumerate(entries)]
    path.write_text(''.join(json.dumps(event) + '\n' for event in events))
    return events


@pytest.mark.parametrize('ending', ['episode_finished', 'episode_aborted'])
def test_episode_closes_before_run_termination(tmp_path, ending):
    path = tmp_path / 'journal.jsonl'
    expected = write_journal(path, [
        ('run_started', None), ('run_resumed', None),
        ('episode_started', 'first'), ('episode_context', 'first'),
        (ending, 'first'),
        ('run_aborted' if ending == 'episode_aborted' else 'run_finished', None),
    ])
    assert read_journal(path) == (expected, False)


def test_interrupted_open_episode_and_general_run_events_are_allowed(tmp_path):
    path = tmp_path / 'journal.jsonl'
    expected = write_journal(path, [
        ('run_started', None), ('model_initialized', None),
        ('episode_started', 'first'), ('episode_context', 'first'),
        ('action_attempted', 'first'),
    ])
    assert read_journal(path) == (expected, False)
    with path.open('a') as stream:
        stream.write('{"interrupted":')
    assert read_journal(path) == (expected, True)


@pytest.mark.parametrize('invalid_entries', [
    [('episode_started', 'first'), ('episode_started', 'second')],
    [('episode_started', 'first'), ('action_succeeded', 'second')],
    [('episode_started', 'first'), ('action_succeeded', None)],
    [('episode_started', 'first'), ('episode_finished', 'first'),
     ('tool_result', 'first')],
    [('episode_started', 'first'), ('episode_finished', 'first'),
     ('episode_finished', 'first')],
    [('episode_started', 'first'), ('episode_aborted', 'first'),
     ('episode_aborted', 'first')],
    [('episode_started', 'first'), ('episode_finished', 'first'),
     ('episode_started', 'second'), ('action_succeeded', 'first')],
    [('episode_started', 'first'), ('run_finished', None)],
    [('episode_started', 'first'), ('run_aborted', None)],
    [('action_succeeded', None)],
    [('episode_finished', None)],
])
def test_invalid_episode_transitions_are_rejected(tmp_path, invalid_entries):
    path = tmp_path / 'journal.jsonl'
    write_journal(path, [('run_started', None), *invalid_entries])
    with pytest.raises(RecoveryError):
        read_journal(path)
