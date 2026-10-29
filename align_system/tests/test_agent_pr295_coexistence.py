"""Keep the existing driver and the separate agentic ADM usable together."""
from copy import deepcopy
from io import StringIO
import logging
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from langchain_core.messages import AIMessage
import pytest

from align_system.algorithms.agents.langchain_agentic_adm import LangChainAgenticADM
from align_system.drivers.itm_open_world_agentic import (
    ITMOpenWorldAgenticDriver, ScenarioSession,
)
from align_system.tests.test_agent_assessments import Scenario, action
from align_system.tests.test_agent_controls_and_events import Actor
from align_system.utils import agent_logging
from align_system.utils import logging as stdlib_logging
from align_system.utils.agent_transcript import AgentTranscriptFileHandler
from align_system.utils.logging import (
    agent_log_call, agent_log_event, agent_log_private, agent_log_scope,
)


CONFIG_DIR = Path(__file__).resolve().parents[1] / 'configs'


@pytest.mark.parametrize('mode', ['live', 'replay'])
@pytest.mark.parametrize('architecture', ['langchain_agent', 'agentic_adm'])
def test_reviewer_experiments_keep_separate_architectures(mode, architecture):
    experiment = f'phase2_{architecture}_ow_review_{mode}'
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name='action_based', overrides=[
            f'+experiment=phase2_feb_openworld/{experiment}',
        ], return_hydra_config=True)
    assert cfg.agent_log_rendering
    assert cfg.raw_log_mode == 'w'
    if architecture == 'langchain_agent':
        assert cfg.get('adm') is None
        assert cfg.driver._target_.endswith('.ITMOpenWorldLangChainDriver')
        assert not cfg.save_agent_transcript
        reviewer = instantiate(cfg.driver.reviewer)
        assert reviewer.response_contract == 'grounded_v2'
    else:
        assert cfg.driver._target_.endswith('.ITMOpenWorldAgenticDriver')
        assert cfg.adm.instance._target_.endswith('.LangChainAgenticADM')
        assert cfg.save_agent_transcript
        assert cfg.hydra.job_logging.handlers.console['class'] == (
            'align_system.utils.agent_logging.AgentRichHandler')
        reviewer = instantiate(cfg.adm.instance.reviewer)
        assert reviewer.response_contract == 'visible_fact_review'


def test_legacy_logging_exports_capture_full_transcripts_at_info(tmp_path):
    assert stdlib_logging is logging
    logger = logging.Logger('coexistence', level=logging.INFO)
    console = StringIO()
    logger.addHandler(logging.StreamHandler(console))
    raw_path = tmp_path / 'raw.log'
    transcript_path = tmp_path / 'interactions.log'
    handler = AgentTranscriptFileHandler(raw_path, transcript_path)
    logger.addHandler(handler)
    payload = 'Complete tool output ' + 'x' * 500
    try:
        # Old imports and new imports must share correlation and redaction state.
        with agent_log_scope(reset_calls=True, scenario_id='shared-scenario'):
            agent_log_private('private transport text')
            with agent_log_call():
                agent_log_event(logger, 'TOOL', 'OUTPUT', 'Legacy tool', payload=payload)
                agent_logging.agent_log_event(
                    logger, 'TOOL', 'OUTPUT', 'New tool',
                    payload='New output: private transport text')
    finally:
        handler.close()
    raw = raw_path.read_text()
    transcript = transcript_path.read_text()
    assert payload in raw
    # Transcripts wrap long lines for readability without truncating their text.
    assert ''.join(payload.split()) in ''.join(transcript.split())
    assert 'shared-scenario' in raw and 'call=1' in raw
    assert 'New output: <exception details omitted>' in transcript
    assert 'private transport text' not in raw + transcript
    assert payload not in console.getvalue()


@pytest.mark.parametrize('reviewed', [False, True])
def test_new_adm_runs_without_changing_existing_driver_contract(reviewed):
    scenario = Scenario([action('search')])
    records = []

    def submit(selected):
        scenario.executed.append(deepcopy(selected))
        scenario.state = deepcopy(scenario.state)
        scenario.state.scenario_complete = True
        return scenario.state

    scenario.take_action = submit
    session = ScenarioSession(scenario, None, False, False, records.append)
    proposal = {'id': 'proposal', 'name': 'search',
                'args': {'justification': 'Search the offered area.'}}
    calls = [proposal]

    class Reviewer:
        response_contract = 'visible_fact_review'
        weighting = None
        calls = 0

        def review(self, context, default_chat_model):
            assert not scenario.executed
            self.calls += 1
            return {'recommendation': 'keep', 'reason': 'Search is offered.',
                    'assessment': {'evidence': [
                        {'path': '/observation/scene_id', 'value': 'scene'}]}}

    reviewer = Reviewer() if reviewed else None
    if reviewed:
        final = deepcopy(proposal)
        final['id'] = 'final'
        final['args']['review_response'] = {
            'acknowledged_suggestion': 'none',
            'response_to_advice': 'The current scene supports searching.',
            'evidence': [{'path': '/current/observation/scene_id', 'value': 'scene'}],
        }
        calls.append(final)
    actor = Actor([AIMessage(content='', tool_calls=[call]) for call in calls])
    adm = LangChainAgenticADM(chat_model=actor, reviewer=reviewer,
                             context_budget={'enabled': False})
    driver = ITMOpenWorldAgenticDriver(apply_action_filtering=False)
    adm.run_scenario(session, driver._build_tools(session))
    assert session.scenario_complete
    assert len(records) == len(scenario.executed) == 1
    assert len(actor.messages) == (2 if reviewed else 1)
    if reviewed:
        assert reviewer.calls == 1
        assert records[0]['choice_info']['multi_agent'][
            'final_review_response_validation']['status'] == 'verified'
