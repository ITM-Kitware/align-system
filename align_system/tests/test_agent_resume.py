"""Recovery exercises real driver bookkeeping with an offline TA3 substitute."""
from copy import deepcopy
import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from omegaconf import OmegaConf
import pytest

from align_system.drivers.agent_history import pack_history
from align_system.drivers.agent_resume import preflight_resume, reconstructed_messages
from align_system.drivers.agent_resume_trace import RecoveryError, ReplayPlan
from align_system.drivers.itm_open_world_langchain import ITMOpenWorldLangChainDriver
from align_system.tests.test_agent_assessments import Model, Scenario, action, action_call
from align_system.tests.test_agent_controls_and_events import Actor, journal_result


class ReplayScenario(Scenario):
    def __init__(self, finish_after=3, target='target', drift_at=None, terminal_error=False):
        super().__init__([action('a')])
        self.state.elapsed_time = 0
        self.finish_after = finish_after
        self.target = target
        self.drift_at = drift_at
        self.terminal_error = terminal_error

    def get_alignment_target(self):
        return Model(id=self.target, kdma_values=[{'kdma': 'affiliation', 'value': 0.8}])

    def take_action(self, selected):
        self.executed.append(deepcopy(selected))
        self.state = deepcopy(self.state)
        self.state.elapsed_time += 1
        self.state.unstructured = f'Observation after action {len(self.executed)}'
        self.state.scenario_complete = len(self.executed) == self.finish_after
        if self.drift_at == len(self.executed):
            self.state.unstructured = 'Different server state'
        if self.terminal_error and self.state.scenario_complete:
            error = RuntimeError('Lost final response')
            error.status = 500
            raise error
        return deepcopy(self.state)


class AutoToolkit:
    prompt_hint = 'Use the cached assessment.'

    def __init__(self):
        self.calls = 0

    def build_tools(self, session):
        return []

    def decision_context(self, session):
        self.calls += 1
        session.record_assessment('example', {'score': 0.75})
        return f'Cached assessment at {session.n_actions}'


def drive(path, monkeypatch, scenarios, actor, resume=None, toolkit=None):
    path.mkdir(exist_ok=True)
    iterator = iter([*scenarios, None])
    cfg = OmegaConf.create({
        'interface': Model(start_scenario=lambda: next(iterator)),
        'align_to_target': True, 'save_input_output': True,
        'save_scoring_output': False, 'save_alignment_targets': False,
        'save_timing': False, 'resume_from': str(resume) if resume else None,
    }, flags={'allow_objects': True})
    monkeypatch.setattr('hydra.core.hydra_config.HydraConfig.get',
                        lambda: Model(runtime=Model(output_dir=str(path))))
    driver = ITMOpenWorldLangChainDriver(
        chat_model=actor, max_actions_per_scenario=10,
        extra_tools={'example': toolkit} if toolkit else {},
        scene_end_policy='agent_decides', max_messages_in_context=40)
    driver.drive(cfg)


def interrupted(path, monkeypatch, toolkit=None):
    actor = Actor([
        AIMessage(content='Original explanation one', tool_calls=[action_call()]),
        AIMessage(content='Original explanation two', tool_calls=[action_call()]),
        *[RuntimeError('Model unavailable')] * 3,
    ])
    with pytest.raises(RuntimeError, match='3 consecutive'):
        drive(path, monkeypatch, [ReplayScenario()], actor, toolkit=toolkit)
    return actor


def edit_events(path, edit):
    journal = path / 'agent_events.jsonl'
    events = [json.loads(line) for line in journal.read_text().splitlines()]
    events = edit(events)
    for i, event in enumerate(events):
        event['sequence'] = i
    journal.write_text(''.join(json.dumps(event) + '\n' for event in events))


@pytest.mark.parametrize('legacy', [False, True])
def test_replay_then_continue_preserves_context_and_new_output(tmp_path, monkeypatch, legacy):
    source, destination = tmp_path / 'source', tmp_path / 'resumed'
    interrupted(source, monkeypatch)
    if legacy:
        (source / 'agent_checkpoint.json').unlink()
        def remove_new_fields(events):
            for event in events:
                if event['event'] in {'action_attempted', 'action_succeeded'}:
                    for key in ('full_action', 'pre_action_state', 'full_state'):
                        event.pop(key, None)
            return events
        edit_events(source, remove_new_fields)
    original = {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    actor, scenario = Actor(), ReplayScenario()
    drive(destination, monkeypatch, [scenario], actor, resume=source)
    assert len(scenario.executed) == 3
    assert len(actor.messages) == 1  # Two actions replayed without inference.
    request = actor.messages[0]
    assert any(isinstance(m, ToolMessage) and 'Observation after action 2' in m.content
               for m in request)
    if not legacy:
        assert any(m.content == 'Original explanation two' for m in request)
    else:
        assert 'Recovery:' in request[1].content
    records, report, events = journal_result(destination)
    assert len(records) == 3 and report['run_status'] == 'finished'
    assert report['successful_actions_missing_from_trace'] == 0
    assert sum('replay' in r['choice_info'] for r in records) == 2
    restored = next(e for e in events if e['event'] == 'conversation_restored')
    assert restored['source'] == ('reconstructed' if legacy else 'checkpoint')
    assert original == {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    assert ReplayPlan(destination).finished


def test_checkpoint_reuses_auto_assessment_at_same_decision(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    interrupted(source, monkeypatch, toolkit=AutoToolkit())
    toolkit, actor = AutoToolkit(), Actor()
    drive(tmp_path / 'resumed', monkeypatch, [ReplayScenario()], actor,
          resume=source, toolkit=toolkit)
    assert toolkit.calls == 0
    assert any('Cached assessment at 2' in m.content for m in actor.messages[0])


@pytest.mark.parametrize('terminal_error', [False, True])
def test_completed_episodes_replay_without_actor_and_keep_terminal_attempt(tmp_path, monkeypatch,
                                                                         terminal_error):
    source = tmp_path / 'source'
    drive(source, monkeypatch, [ReplayScenario(2, terminal_error=terminal_error)], Actor())
    plan = ReplayPlan(source)
    assert len(plan.episodes[0].actions) == 2
    assert plan.episodes[0].actions[-1].recovered_completion == terminal_error
    fresh, actor = ReplayScenario(2), Actor([])
    drive(tmp_path / 'resumed', monkeypatch, [fresh], actor, resume=source)
    assert len(fresh.executed) == 2 and not actor.messages
    records, report, _ = journal_result(tmp_path / 'resumed')
    assert len(records) == 2 and report['run_status'] == 'finished'


@pytest.mark.parametrize('mismatch', ['initial', 'target', 'during'])
def test_replay_stops_on_divergence_before_further_mutations(tmp_path, monkeypatch, mismatch):
    source = tmp_path / 'source'
    interrupted(source, monkeypatch)
    scenario = ReplayScenario(target='other' if mismatch == 'target' else 'target',
                              drift_at=1 if mismatch == 'during' else None)
    if mismatch == 'initial':
        scenario.state.unstructured = 'Different initial state'
    actor = Actor([])
    with pytest.raises(RecoveryError, match='expected|diverged'):
        drive(tmp_path / 'resumed', monkeypatch, [scenario], actor, resume=source)
    assert len(scenario.executed) == (1 if mismatch == 'during' else 0)
    assert not actor.messages


def test_replay_detects_early_end_of_server_sequence(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    interrupted(source, monkeypatch)
    with pytest.raises(RecoveryError, match='TA3 ended'):
        drive(tmp_path / 'resumed', monkeypatch, [], Actor([]), resume=source)


def test_durable_success_recovers_trace_write_gap(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    interrupted(source, monkeypatch)
    records = json.loads((source / 'input_output.json').read_text())
    (source / 'input_output.json').write_text(json.dumps(records[:-1]))
    plan = ReplayPlan(source)
    assert plan.episodes[0].confirmed_actions == 2
    fresh = ReplayScenario()
    drive(tmp_path / 'resumed', monkeypatch, [fresh], Actor(), resume=source)
    assert len(fresh.executed) == 3


def test_unconfirmed_attempt_is_not_replayed_and_tool_exchange_is_closed(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    interrupted(source, monkeypatch)
    (source / 'agent_checkpoint.json').unlink()
    def append_unconfirmed(events):
        events = events[:-2]  # Remove episode_aborted and run_aborted.
        envelope = {k: events[-1][k] for k in ('schema_version', 'run_id', 'episode_id', 'timestamp')}
        events.extend([
            dict(envelope, event='model_completed', decision_index=2, tool_calls=[action_call()]),
            dict(envelope, event='tool_requested', decision_index=2, tool_call=action_call()),
            dict(envelope, event='action_attempted', decision_index=2,
                 action={'action_id': 'a', 'action_type': 'SEARCH'}),
        ])
        return events
    edit_events(source, append_unconfirmed)
    with (source / 'agent_events.jsonl').open('a') as stream:
        stream.write('{"unfinished":')
    plan = ReplayPlan(source)
    assert plan.truncated_journal and plan.episodes[0].confirmed_actions == 2
    messages = reconstructed_messages(plan.episodes[0])
    pack_history([HumanMessage(content='Recovered'), *messages])
    assert 'outcome was not confirmed' in messages[-1].content
    actor, fresh = Actor(), ReplayScenario()
    drive(tmp_path / 'resumed', monkeypatch, [fresh], actor, resume=source)
    assert len(fresh.executed) == 3 and len(actor.messages) == 1


def test_preflight_rejects_source_reuse_and_unsupported_driver(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    interrupted(source, monkeypatch)
    cfg = OmegaConf.create({'resume_from': str(source), 'driver': {
        '_target_': 'align_system.drivers.itm_open_world_langchain.ITMOpenWorldLangChainDriver',
        'reviewer': None}})
    with pytest.raises(RecoveryError, match='new output directory'):
        preflight_resume(cfg, source)
    preflight_resume(cfg, tmp_path / 'fresh')
    cfg.driver['_target_'] = 'different_driver'
    with pytest.raises(RecoveryError, match='LangChain'):
        preflight_resume(cfg, tmp_path / 'fresh')


def test_tampered_trace_is_rejected_before_replay(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    interrupted(source, monkeypatch)
    rows = json.loads((source / 'input_output.json').read_text())
    rows[0]['output']['action']['action_id'] = 'different'
    (source / 'input_output.json').write_text(json.dumps(rows))
    with pytest.raises(RecoveryError, match='does not match'):
        ReplayPlan(source)
