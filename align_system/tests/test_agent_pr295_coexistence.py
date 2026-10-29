"""Compatibility checks for agent logging and interaction transcripts."""
from io import StringIO
import logging

from align_system.utils import agent_logging
from align_system.utils import logging as stdlib_logging
from align_system.utils.agent_transcript import AgentTranscriptFileHandler
from align_system.utils.logging import (
    agent_log_call, agent_log_event, agent_log_private, agent_log_scope,
)


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
