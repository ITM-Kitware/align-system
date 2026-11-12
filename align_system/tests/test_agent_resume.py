"""Exercise replay and conversation recovery without a model or TA3 server."""
from copy import deepcopy
import json
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from omegaconf import OmegaConf
import pytest
from swagger_client.models import ActionTypeEnum

from align_system.algorithms.agents.agent_history import pack_history
from align_system.drivers.agent_resume import preflight_resume, reconstructed_messages
from align_system.drivers.agent_resume_trace import RecoveryError, ReplayPlan
from align_system.drivers.itm_open_world_langchain import ITMOpenWorldLangChainDriver


class Model(SimpleNamespace):
    def to_dict(self):
        def convert(value):
            if hasattr(value, 'to_dict'):
                return value.to_dict()
            if isinstance(value, list):
                return [convert(item) for item in value]
            if isinstance(value, dict):
                return {key: convert(item) for key, item in value.items()}
            return value
        return {key: convert(value) for key, value in vars(self).items()}


def action_call():
    return {'id': 'search-call', 'name': 'search',
            'args': {'justification': 'Look for casualties'}}


class Actor:
    def __init__(self, responses=None):
        self.responses = iter(responses) if responses is not None else None
        self.messages = []

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.messages.append(deepcopy(messages))
        response = (next(self.responses) if self.responses is not None else
                    AIMessage(content='Continue searching', tool_calls=[action_call()]))
        if isinstance(response, Exception):
            raise response
        return response


class ReplayScenario:
    def __init__(self, finish_after=3, target='target', drift_at=None,
                 terminal_error=False):
        self.executed = []
        self.finish_after = finish_after
        self.target = target
        self.drift_at = drift_at
        self.terminal_error = terminal_error
        self.state = Model(
            unstructured='A casualty awaits assessment.', elapsed_time=0,
            scenario_complete=False, meta_info=Model(scene_id='scene'),
            characters=[Model(id='p1', name='Alex', tag=None, unseen=False)],
            supplies=None)
        self.actions = [Model(
            action_id='a', action_type=ActionTypeEnum.SEARCH,
            unstructured='Search for casualties', character_id=None,
            parameters={}, justification=None, intent_action=False,
            kdma_association=None)]

    def id(self):
        return 'scenario'

    def get_alignment_target(self):
        return Model(id=self.target,
                     kdma_values=[{'kdma': 'affiliation', 'value': 0.8}])

    def get_state(self):
        return deepcopy(self.state)

    def get_available_actions(self):
        return deepcopy(self.actions)

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


def drive(path, monkeypatch, scenarios, actor, resume=None, toolkit=None,
          align_to_target=True, save_agent_checkpoint=True):
    path.mkdir(exist_ok=True)
    iterator = iter([*scenarios, None])
    cfg = OmegaConf.create({
        'interface': Model(start_scenario=lambda: next(iterator)),
        'align_to_target': align_to_target, 'save_input_output': True,
        'save_scoring_output': False, 'save_alignment_targets': False,
        'save_timing': False, 'resume_from': str(resume) if resume else None,
        'save_agent_checkpoint': save_agent_checkpoint,
    }, flags={'allow_objects': True})
    monkeypatch.setattr('hydra.core.hydra_config.HydraConfig.get',
                        lambda: Model(runtime=Model(output_dir=str(path))))
    driver = ITMOpenWorldLangChainDriver(
        chat_model=actor, max_actions_per_scenario=10,
        extra_tools={'example': toolkit} if toolkit else {},
        apply_action_filtering=False)
    driver.drive(cfg)


def interrupted(path, monkeypatch, toolkit=None, align_to_target=True,
                save_agent_checkpoint=True):
    actor = Actor([
        AIMessage(content='Original explanation one', tool_calls=[action_call()]),
        AIMessage(content='Original explanation two', tool_calls=[action_call()]),
        *[RuntimeError('Model unavailable')] * 3,
    ])
    with pytest.raises(RuntimeError, match='3 consecutive'):
        drive(path, monkeypatch, [ReplayScenario()], actor, toolkit=toolkit,
              align_to_target=align_to_target,
              save_agent_checkpoint=save_agent_checkpoint)
    return actor


def journal(path):
    return [json.loads(line) for line in
            (path / 'agent_events.jsonl').read_text().splitlines()]


def edit_events(path, edit):
    events = edit(journal(path))
    for index, event in enumerate(events):
        event['sequence'] = index
    (path / 'agent_events.jsonl').write_text(
        ''.join(json.dumps(event) + '\n' for event in events))


@pytest.mark.parametrize('legacy', [False, True])
def test_replay_then_continue_preserves_context_and_source(tmp_path, monkeypatch, legacy):
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
    assert len(actor.messages) == 1
    request = actor.messages[0]
    assert any(isinstance(message, ToolMessage)
               and 'Observation after action 2' in message.content
               for message in request)
    if legacy:
        assert 'Recovery:' in request[1].content
    else:
        assert any(message.content == 'Original explanation two' for message in request)
    records = json.loads((destination / 'input_output.json').read_text())
    assert len(records) == 3
    assert sum('replay' in row['choice_info'] for row in records) == 2
    restored = next(event for event in journal(destination)
                    if event['event'] == 'conversation_restored')
    assert restored['source'] == ('reconstructed' if legacy else 'checkpoint')
    assert original == {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    assert ReplayPlan(destination).finished


def test_checkpoint_reuses_assessment_at_same_decision(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    interrupted(source, monkeypatch, toolkit=AutoToolkit())
    toolkit, actor = AutoToolkit(), Actor()
    drive(tmp_path / 'resumed', monkeypatch, [ReplayScenario()], actor,
          resume=source, toolkit=toolkit)
    assert toolkit.calls == 0
    assert any('Cached assessment at 2' in message.content
               for message in actor.messages[0])
    checkpoint = json.loads((tmp_path / 'resumed' / 'agent_checkpoint.json').read_text())
    assert checkpoint['assessments']['example']['score'] == 0.75


def test_unaligned_baseline_resumes_without_an_alignment_target(tmp_path, monkeypatch):
    source, destination = tmp_path / 'source', tmp_path / 'resumed'
    interrupted(source, monkeypatch, align_to_target=False)
    assert ReplayPlan(source).episodes[0].target_id is None
    actor, scenario = Actor(), ReplayScenario(target='different-unused-target')
    drive(destination, monkeypatch, [scenario], actor, resume=source,
          align_to_target=False)
    assert len(scenario.executed) == 3 and len(actor.messages) == 1
    records = json.loads((destination / 'input_output.json').read_text())
    assert all(row['input']['alignment_target_id'] is None for row in records)


def test_disabling_checkpoint_uses_journal_reconstruction(tmp_path, monkeypatch):
    source, destination = tmp_path / 'source', tmp_path / 'resumed'
    interrupted(source, monkeypatch, save_agent_checkpoint=False)
    assert not (source / 'agent_checkpoint.json').exists()
    actor, scenario = Actor(), ReplayScenario()
    drive(destination, monkeypatch, [scenario], actor, resume=source,
          save_agent_checkpoint=False)
    assert len(scenario.executed) == 3 and len(actor.messages) == 1
    assert 'Recovery:' in actor.messages[0][1].content
    assert not (destination / 'agent_checkpoint.json').exists()
    assert next(event for event in journal(destination)
                if event['event'] == 'conversation_restored')['source'] == 'reconstructed'


@pytest.mark.parametrize('terminal_error', [False, True])
def test_completed_episodes_replay_without_actor(tmp_path, monkeypatch, terminal_error):
    source = tmp_path / 'source'
    drive(source, monkeypatch, [ReplayScenario(2, terminal_error=terminal_error)], Actor())
    plan = ReplayPlan(source)
    assert len(plan.episodes[0].actions) == 2
    assert plan.episodes[0].actions[-1].recovered_completion == terminal_error
    fresh, actor = ReplayScenario(2), Actor([])
    drive(tmp_path / 'resumed', monkeypatch, [fresh], actor, resume=source)
    assert len(fresh.executed) == 2 and not actor.messages
    assert ReplayPlan(tmp_path / 'resumed').finished


def test_completed_episode_replays_before_unfinished_episode(tmp_path, monkeypatch):
    source, destination = tmp_path / 'source', tmp_path / 'resumed'
    responses = [AIMessage(content=f'Explanation {index}', tool_calls=[action_call()])
                 for index in range(4)]
    responses.extend([RuntimeError('Model unavailable')] * 3)
    with pytest.raises(RuntimeError, match='3 consecutive'):
        drive(source, monkeypatch, [ReplayScenario(2), ReplayScenario(3)], Actor(responses))
    plan = ReplayPlan(source)
    assert [episode.completed for episode in plan.episodes] == [True, False]
    fresh, actor = [ReplayScenario(2), ReplayScenario(3)], Actor()
    drive(destination, monkeypatch, fresh, actor, resume=source)
    assert [len(scenario.executed) for scenario in fresh] == [2, 3]
    assert len(actor.messages) == 1
    records = json.loads((destination / 'input_output.json').read_text())
    assert len(records) == 5
    assert sum('replay' in row['choice_info'] for row in records) == 4


@pytest.mark.parametrize('mismatch', ['initial', 'target', 'during'])
def test_replay_stops_before_further_mutations_on_divergence(tmp_path, monkeypatch, mismatch):
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


def test_replay_rejects_terminal_state_drift(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    drive(source, monkeypatch, [ReplayScenario(2)], Actor())
    actor, scenario = Actor([]), ReplayScenario(2, drift_at=2)
    with pytest.raises(RecoveryError, match='diverged'):
        drive(tmp_path / 'resumed', monkeypatch, [scenario], actor, resume=source)
    assert len(scenario.executed) == 2 and not actor.messages


def test_replay_detects_early_end_of_server_sequence(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    interrupted(source, monkeypatch)
    with pytest.raises(RecoveryError, match='TA3 ended'):
        drive(tmp_path / 'resumed', monkeypatch, [], Actor([]), resume=source)


def test_durable_success_recovers_trace_write_gap(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    interrupted(source, monkeypatch)
    rows = json.loads((source / 'input_output.json').read_text())
    (source / 'input_output.json').write_text(json.dumps(rows[:-1]))
    assert ReplayPlan(source).episodes[0].confirmed_actions == 2
    fresh = ReplayScenario()
    drive(tmp_path / 'resumed', monkeypatch, [fresh], Actor(), resume=source)
    assert len(fresh.executed) == 3


def test_unconfirmed_attempt_is_not_replayed_and_exchange_is_closed(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    interrupted(source, monkeypatch)
    (source / 'agent_checkpoint.json').unlink()

    def append_unconfirmed(events):
        events = events[:-2]
        envelope = {key: events[-1][key]
                    for key in ('schema_version', 'run_id', 'episode_id', 'timestamp')}
        events.extend([
            dict(envelope, event='model_completed', decision_index=2,
                 tool_calls=[action_call()]),
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


def test_preflight_rejects_changed_toolkit_settings(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    interrupted(source, monkeypatch)
    cfg = OmegaConf.create({'resume_from': str(source), 'driver': {
        '_target_': 'align_system.drivers.itm_open_world_langchain.ITMOpenWorldLangChainDriver',
        'extra_tools': {'example': {'mode': 'automatic'}}}})
    (source / '.hydra').mkdir()
    OmegaConf.save(cfg, source / '.hydra' / 'config.yaml')
    preflight_resume(cfg, tmp_path / 'fresh')
    cfg.driver.extra_tools.example.mode = 'on_demand'
    with pytest.raises(RecoveryError, match='toolkit'):
        preflight_resume(cfg, tmp_path / 'fresh')


def test_preflight_accepts_identical_custom_toolkit_references(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    interrupted(source, monkeypatch)
    cfg = OmegaConf.create({'resume_from': str(source), 'driver': {
        '_target_': 'align_system.drivers.itm_open_world_langchain.ITMOpenWorldLangChainDriver',
        'chat_model': {'model': 'example-model'},
        'extra_tools': {'example': {'_target_': 'example.Toolkit',
                                    'chat_model': '${ref:driver.chat_model}'}}}})
    (source / '.hydra').mkdir()
    OmegaConf.save(cfg, source / '.hydra' / 'config.yaml')
    preflight_resume(cfg, tmp_path / 'fresh')


def test_tampered_trace_is_rejected_before_replay(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    interrupted(source, monkeypatch)
    rows = json.loads((source / 'input_output.json').read_text())
    rows[0]['output']['action']['action_id'] = 'different'
    (source / 'input_output.json').write_text(json.dumps(rows))
    with pytest.raises(RecoveryError, match='does not match'):
        ReplayPlan(source)
