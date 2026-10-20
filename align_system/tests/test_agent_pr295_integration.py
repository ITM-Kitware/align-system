"""Exercise review/execution boundaries and ambiguous live completion responses."""
from copy import deepcopy
import json

import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from langchain_core.messages import AIMessage

from align_system.drivers.agent_review import ReviewWeighting
from align_system.drivers.itm_open_world_langchain import ITMOpenWorldLangChainDriver
from align_system.tests.test_agent_assessments import (
    Scenario, action, action_call, session_for,
)
from align_system.tests.test_agent_controls_and_events import Actor, drive_fixture
from align_system.tests.test_agent_launchers import ROOT


class HTTPError(Exception):
    def __init__(self, status, body):
        self.status = status
        self.body = body
        super().__init__(body)


class Reviewer:
    response_contract = 'grounded_v2'

    def __init__(self, scenario, weighted=False):
        self.scenario = scenario
        self.calls = 0
        self.weighting = ReviewWeighting() if weighted else None

    def review(self, context, default_chat_model):
        self.calls += 1
        # Proposals must never reach the server before the reviewer runs.
        assert not self.scenario.executed
        assert 'hidden_label' not in json.dumps(context)
        advice = {
            'recommendation': 'keep', 'reason': 'Search is offered.',
            'assessment': {'evidence': [
                {'path': '/observation/scene_id', 'value': 'scene'}]},
        }
        if self.weighting is not None:
            advice.update(alternative=None, confidence=None, estimated_improvement=None)
        return advice


@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('exact_id', [False, True])
def test_review_gates_execution_and_requires_verified_acknowledgment(weighted, exact_id):
    session, records = session_for([action('a', kdma_association={'hidden_label': 999})])
    session.alignment_target = None
    reviewer = Reviewer(session.scenario, weighted)
    proposal = (action_call() if exact_id else
                {'id': 'search', 'name': 'search', 'args': {'justification': 'Search'}})
    invalid_final = deepcopy(proposal)
    invalid_final['args']['review_response'] = {
        'acknowledged_suggestion': 'keep' if weighted else 'none',
        'response_to_advice': 'The current scene supports searching.',
        'evidence': [{'path': '/current/observation/scene_id', 'value': 'invented'}],
    }
    final = deepcopy(invalid_final)
    final['args']['review_response']['evidence'][0]['value'] = 'scene'
    if exact_id:
        # Qwen XML parsers can serialize a nested acknowledgment as JSON text.
        final['args']['review_response'] = json.dumps(final['args']['review_response'])
    actor = Actor([AIMessage(content='', tool_calls=[call])
                   for call in (proposal, invalid_final, final)])
    driver = ITMOpenWorldLangChainDriver(
        chat_model=actor, reviewer=reviewer, max_actions_per_scenario=1)
    driver._run_agent_loop(session)
    assert reviewer.calls == 1
    assert len(actor.messages) == 3
    assert len(session.scenario.executed) == len(records) == 1
    review = records[0]['choice_info']['multi_agent']
    assert review['review_status'] == 'reviewed'
    assert review['final_review_response_validation']['status'] == 'verified'
    assert review['calls'] == {'decision_agent': 3, 'reviewer': 1}
    if weighted:
        assert review['weighting']['suggestion'] == 'keep'


@pytest.mark.parametrize('status,body', [(500, 'Internal Server Error'),
                                       (400, '"Scenario Complete"')])
def test_final_action_error_recovers_score_and_continues_run(tmp_path, monkeypatch, status, body):
    scenario = Scenario([action('a')])
    submissions = []

    def submit(selected):
        submissions.append(selected)
        scenario.state = deepcopy(scenario.state)
        scenario.state.scenario_complete = True
        scenario.state.unstructured = 'Session alignment score = 0.8'
        raise HTTPError(status, body)

    scenario.take_action = submit
    actor = Actor([AIMessage(content='', tool_calls=[action_call()])])
    drive_fixture(tmp_path, monkeypatch, ITMOpenWorldLangChainDriver(chat_model=actor), scenario)
    assert len(submissions) == 1
    records = json.loads((tmp_path / 'input_output.json').read_text())
    assert not records  # An ambiguous submission is not a confirmed executed action.
    events = [json.loads(line) for line in (tmp_path / 'agent_events.jsonl').read_text().splitlines()]
    recovery = next(e for e in events if e['event'] == 'scenario_completion_recovered')
    assert recovery['action_outcome'] == 'unconfirmed'
    assert recovery['pre_action_state']['scenario_complete'] is False
    assert not any(e['event'] == 'action_succeeded' for e in events)
    score = next(e for e in events if e['event'] == 'alignment_scoring')
    assert score['completed'] is True
    assert score['final_state']['score'] == 0.8
    assert any(e['event'] == 'episode_finished' and e['completed'] for e in events)


def test_completed_action_listing_reconciles_without_another_model_call():
    session, _ = session_for()
    session.scenario.state = deepcopy(session.current_state)
    session.scenario.state.scenario_complete = True

    def actions():
        raise HTTPError(400, '"Scenario Complete"')

    session.scenario.get_available_actions = actions
    actor = Actor([])
    ITMOpenWorldLangChainDriver(chat_model=actor)._run_agent_loop(session)
    assert session.scenario_complete
    assert not actor.messages
    assert not session.scenario.executed


@pytest.mark.parametrize('readback_fails', [False, True])
def test_unconfirmed_server_error_aborts_without_resubmission(readback_fails):
    session, records = session_for([action('a')])
    submissions = []
    original = HTTPError(500, 'Internal Server Error')

    def submit(selected):
        submissions.append(selected)
        raise original

    session.scenario.take_action = submit
    if readback_fails:
        def state():
            raise HTTPError(503, 'Unavailable')
        session.scenario.get_state = state
    actor = Actor([AIMessage(content='', tool_calls=[action_call()])])
    with pytest.raises(HTTPError) as caught:
        ITMOpenWorldLangChainDriver(chat_model=actor)._run_agent_loop(session)
    assert (caught.value.__cause__ if readback_fails else caught.value) is original
    assert len(submissions) == 1
    assert not session.scenario_complete
    assert not records


@pytest.mark.parametrize('profile', ['weighted', 'qualitative'])
def test_existing_kaleido_experiment_can_select_pr_reviewer(profile):
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / 'align_system/configs')):
        cfg = compose(config_name='action_based', overrides=[
            '+experiment=phase2_feb_openworld/phase2_langchain_agent_ow_kaleido_live',
            f'driver/reviewer={profile}'])
    reviewer = instantiate(cfg.driver.reviewer)
    assert reviewer.response_contract == 'grounded_v2'
    assert 'kaleido' in cfg.driver.extra_tools
    assert cfg.driver.context_budget.enabled
    assert 'medical' in cfg.driver.attribute_definitions
    assert cfg.driver.scene_end_policy == 'auto'


def test_mandatory_context_overflow_never_falls_back_to_an_action():
    from align_system.drivers.agent_context_budget import ContextBudgetError

    session, records = session_for()
    actor = Actor([])
    driver = ITMOpenWorldLangChainDriver(chat_model=actor, context_budget={
        'decision': {'counter': lambda payload: 5000,
                     'context_window': 4096, 'output_tokens': 100}})
    with pytest.raises(ContextBudgetError, match='Mandatory decision context'):
        driver._run_agent_loop(session)
    assert not actor.messages
    assert not records
    assert not session.scenario.executed
