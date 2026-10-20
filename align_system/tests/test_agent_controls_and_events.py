"""Offline experiments cover failures as well as successful decisions."""
from copy import deepcopy
import json

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from langchain_core.messages import AIMessage
from omegaconf import OmegaConf
import pytest

from align_system.algorithms.target_context import TargetContext
from align_system.cli.agent_event_summary import load_event_report, summarize_event_reports
from align_system.cli.compare_agent_alignment import compare_study
from align_system.cli.summarize_agent_alignment import main as summarize_main
from align_system.drivers.itm_open_world_langchain import ITMOpenWorldLangChainDriver
from align_system.exceptions import ScenarioActionLimitError
from align_system.tests.test_agent_assessments import (
    Model, Scenario, action, action_call, session_for)
from align_system.tests.test_agent_launchers import ROOT
from align_system.utils.agent_events import AgentEventLog


class Actor:
    def __init__(self, responses=None):
        self.responses = iter(responses) if responses is not None else None
        self.messages = []

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.messages.append(deepcopy(messages))
        result = next(self.responses) if self.responses is not None else AIMessage(
            content='', tool_calls=[action_call()])
        if isinstance(result, BaseException):
            raise result
        return result


def drive_fixture(tmp_path, monkeypatch, driver, scenario=None):
    scenario = scenario or Scenario([action('a'), action('b')])
    scenarios = iter([scenario, None])
    scenario.get_alignment_target = lambda: Model(
        id='target', kdma_values=[{'kdma': 'affiliation', 'value': 0.8}])
    interface = Model(start_scenario=lambda: next(scenarios))
    cfg = OmegaConf.create(dict(
        interface=interface, align_to_target=True, save_input_output=True,
        save_scoring_output=False, save_alignment_targets=False, save_timing=False),
        flags={'allow_objects': True})
    monkeypatch.setattr(
        'hydra.core.hydra_config.HydraConfig.get',
        lambda: Model(runtime=Model(output_dir=str(tmp_path))))
    driver.drive(cfg)
    return scenario


def journal_result(path):
    records = json.loads((path / 'input_output.json').read_text())
    report = load_event_report(path / 'input_output.json', records)
    events = [json.loads(line) for line in (path / 'agent_events.jsonl').read_text().splitlines()]
    return records, report, events


def test_target_context_is_pinned_without_model_calls_or_hidden_labels():
    session, records = session_for([action('a', kdma_association={'hidden': 999})])
    session.alignment_target.kdma_values = [{'kdma': 'affiliation', 'distribution': {'low': 0.3}}]
    target = TargetContext({'affiliation': {'description': 'Group membership'},
                            'unrelated': {'description': 'unused'}})
    actor = Actor()
    driver = ITMOpenWorldLangChainDriver(
        chat_model=actor, target_context=target, apply_action_filtering=False,
        max_messages_in_context=2, max_actions_per_scenario=3)
    driver._run_agent_loop(session)
    assert len(actor.messages) == len(records) == 3
    for messages in actor.messages:
        assert 'distribution' in messages[0].content
        assert 'Group membership' in messages[0].content
        assert 'unrelated' not in messages[0].content
        assert 'hidden' not in messages[0].content


def test_direct_target_experiment_config():
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / 'align_system/configs')):
        cfg = compose(config_name='action_based', overrides=[
            '+experiment=phase2_feb_openworld/phase2_langchain_agent_ow_target_live'])
    assert isinstance(instantiate(cfg.driver.target_context), TargetContext)


def test_unreachable_actor_has_three_attempts_and_no_actions(tmp_path, monkeypatch):
    driver = ITMOpenWorldLangChainDriver(chat_model=Actor([RuntimeError('down')] * 3))
    with pytest.raises(RuntimeError, match='3 consecutive'):
        drive_fixture(tmp_path, monkeypatch, driver)
    records, report, _ = journal_result(tmp_path)
    assert not records
    assert report['event_counts']['model_started'] == report['event_counts']['model_failed'] == 3
    assert summarize_event_reports([report])['episode_completion_rate'] == 0


def test_invalid_proposal_and_rejection_survive_success(tmp_path, monkeypatch):
    class Rejected(Exception):
        status = 400

    scenario = Scenario([action('a')])
    attempts = []

    def take_action(selected):
        attempts.append(selected)
        if len(attempts) == 1:
            raise Rejected('too distant')
        state = deepcopy(scenario.state)
        state.scenario_complete = True
        return state

    scenario.take_action = take_action
    actor = Actor([AIMessage(content='', tool_calls=[action_call('missing')]),
                   AIMessage(content='', tool_calls=[action_call()]),
                   AIMessage(content='', tool_calls=[action_call()])])
    drive_fixture(tmp_path, monkeypatch, ITMOpenWorldLangChainDriver(chat_model=actor), scenario)
    records, report, events = journal_result(tmp_path)
    assert len(records) == 1
    assert report['episode_outcomes'] == {'completed': 1}
    assert report['event_counts']['action_attempted'] == 2
    assert report['event_counts']['action_rejected'] == 1
    assert report['event_counts']['tool_requested'] == 3
    assert any(e['event'] == 'tool_result' and not e['action_executed'] for e in events)
    assert report['successful_actions_missing_from_trace'] == 0
    assert report['decisions_without_executed_action'] == 0


def test_initialization_failure_and_output_directory_reuse(tmp_path, monkeypatch):
    driver = ITMOpenWorldLangChainDriver()
    with pytest.raises(ValueError):
        drive_fixture(tmp_path, monkeypatch, driver)
    records, report, _ = journal_result(tmp_path)
    assert records == [] and report['run_status'] == 'aborted'
    assert report['episodes_started'] == 0
    previous = (tmp_path / 'agent_events.jsonl').read_bytes()
    with pytest.raises(FileExistsError):
        drive_fixture(tmp_path, monkeypatch, driver)
    assert previous == (tmp_path / 'agent_events.jsonl').read_bytes()


def test_unfinished_episode_and_cli_reliability(tmp_path, monkeypatch, capsys):
    with pytest.raises(ScenarioActionLimitError):
        drive_fixture(tmp_path, monkeypatch, ITMOpenWorldLangChainDriver(
            chat_model=Actor(), max_actions_per_scenario=1))
    records, report, _ = journal_result(tmp_path)
    assert len(records) == 1 and report['run_status'] == 'aborted'
    assert report['episode_outcomes'] == {'aborted': 1}
    summarize_main([str(tmp_path / 'input_output.json')])
    result = json.loads(capsys.readouterr().out)
    assert result['reliability']['episode_completion_rate'] == 0
    assert result['overall']['preference_agreement'] is None


def test_interrupted_journal_retains_pending_decision_and_flags_partial_line(tmp_path):
    with AgentEventLog(tmp_path / 'agent_events.jsonl') as journal:
        with journal.episode('scene') as emit:
            emit('decision_started', decision_index=0)
    path = tmp_path / 'agent_events.jsonl'
    lines = path.read_text().splitlines(True)
    path.write_text(''.join(lines[:-1]) + '{"event":')
    report = load_event_report(tmp_path / 'input_output.json', [])
    assert report['truncated_last_line']
    assert report['run_status'] == 'unfinished'
    assert report['episode_outcomes'] == {'unfinished': 1}
    assert report['decisions_without_executed_action'] == 1


def test_journal_mismatch_and_mid_file_corruption_fail(tmp_path, monkeypatch):
    with pytest.raises(ScenarioActionLimitError):
        drive_fixture(tmp_path, monkeypatch, ITMOpenWorldLangChainDriver(
            chat_model=Actor(), max_actions_per_scenario=1))
    records, _, _ = journal_result(tmp_path)
    records[0]['choice_info']['event_log']['run_id'] = 'another-run'
    with pytest.raises(ValueError, match='does not match'):
        load_event_report(tmp_path / 'input_output.json', records)
    path = tmp_path / 'agent_events.jsonl'
    path.write_text(path.read_text() + 'broken\n{}\n')
    with pytest.raises(ValueError, match='Invalid event JSON'):
        load_event_report(tmp_path / 'input_output.json', [])


def test_comparison_includes_aborted_variant_with_no_successes(tmp_path, monkeypatch):
    runs = []
    for variant, actor in [('baseline', Actor()), ('failed', Actor([RuntimeError('down')] * 3))]:
        directory = tmp_path / variant
        directory.mkdir()
        driver = ITMOpenWorldLangChainDriver(chat_model=actor, max_actions_per_scenario=1)
        if variant == 'failed':
            with pytest.raises(RuntimeError):
                drive_fixture(directory, monkeypatch, driver)
        else:
            with pytest.raises(ScenarioActionLimitError):
                drive_fixture(directory, monkeypatch, driver)
        runs.append({'variant': variant, 'replicate': '0',
                     'input_output': f'{variant}/input_output.json'})
    report = compare_study({'reference': 'baseline', 'runs': runs}, tmp_path)
    failed = report['variants']['failed']
    assert failed['reliability']['zero_action_episodes'] == 1
    assert failed['reliability']['run_outcomes'] == {'aborted': 1}
    assert failed['overall']['preference_agreement'] is None
    assert report['paired_comparisons']['failed']['matched_initial_episodes'] == 0


def test_legacy_runs_have_unknown_reliability(tmp_path):
    assert summarize_event_reports([load_event_report(tmp_path / 'input_output.json', [])]) == {
        'runs_with_events': 0, 'runs_missing_events': 1, 'episode_completion_rate': None}


def test_copied_zero_action_journal_cannot_duplicate_study_runs(tmp_path):
    with AgentEventLog(tmp_path / 'agent_events.jsonl'):
        pass
    runs = []
    for name in ('first', 'copied'):
        (tmp_path / f'{name}.json').write_text('[]')
        runs.append({'variant': name, 'replicate': '0', 'input_output': f'{name}.json'})
    with pytest.raises(ValueError, match='listed more than once'):
        compare_study({'reference': 'first', 'runs': runs}, tmp_path)


def test_execution_before_trace_write_failure_is_visible(tmp_path, monkeypatch):
    from align_system.drivers import itm_open_world_langchain as driver_module
    # The environment may have acted even when the durable action trace fails.
    original_write = driver_module.write_json_atomic

    def fail_after_initialization(path, value):
        if value:
            raise OSError('simulated full disk')
        original_write(path, value)

    monkeypatch.setattr(driver_module, 'write_json_atomic', fail_after_initialization)
    with pytest.raises(OSError):
        drive_fixture(tmp_path, monkeypatch, ITMOpenWorldLangChainDriver(chat_model=Actor()))
    records, report, _ = journal_result(tmp_path)
    assert not records and report['run_status'] == 'aborted'
    assert report['successful_actions_missing_from_trace'] == 1
    assert report['zero_action_episodes'] == 0
