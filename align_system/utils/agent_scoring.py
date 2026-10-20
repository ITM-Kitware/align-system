"""Save server scoring results and explain missing scores per episode."""
from contextlib import contextmanager
import math
from pathlib import Path
import re

from align_system.interfaces.ta3_caci_action_based_service import (
    TA3CACIActionBasedServiceInterface)
from align_system.exceptions import ScenarioActionLimitError, ScenarioStalledError
from align_system.utils import logging
from align_system.utils.agent_events import write_json_atomic


log = logging.getLogger(__name__)


def final_state_score(text):
    """Keep text-derived scores separate from the server's structured results."""
    result = {'status': 'missing', 'score': None, 'text': text}
    match = re.search(r'Session alignment score\s*=\s*([^\r\n]+)', text or '')
    if match is None:
        return result
    raw = match.group(1).strip()
    result['raw_value'] = raw
    try:
        score = float(raw)
    except ValueError:
        score = None
    if score is not None and math.isfinite(score):
        result.update(status='available', score=score)
    else:
        result['status'] = 'unavailable'
    return result


class AgentScoreRecorder:
    def __init__(self, output_dir, interface, events, enabled=True):
        self.output_dir = Path(output_dir)
        self.interface = interface
        self.events = events
        self.enabled = enabled
        self.scores = []
        self.statuses = []
        if enabled:
            write_json_atomic(self.output_dir / 'scores.json', [])
            write_json_atomic(self.output_dir / 'scoring_status.json', [])

    def _save_status(self):
        if self.enabled:
            write_json_atomic(self.output_dir / 'scoring_status.json', self.statuses)

    @contextmanager
    def episode(self, scenario_id):
        record = {
            'run_id': self.events.run_id,
            'episode_id': self.events.episode_id,
            'scenario_id': scenario_id,
            'session_id': getattr(self.interface, 'session_id', None),
            'alignment_target_id': None,
            'status': 'pending',
            'completed': False,
        }
        self.statuses.append(record)
        self._save_status()
        try:
            yield record
        except BaseException as exc:
            # Do not make remote scoring calls while unwinding a failed run.
            status = ('incomplete' if isinstance(exc, ScenarioActionLimitError)
                      else 'stalled' if isinstance(exc, ScenarioStalledError)
                      else 'aborted')
            record.update(status=status, error_type=type(exc).__name__,
                          error=str(exc))
            if isinstance(exc, ScenarioStalledError):
                record['stall'] = exc.diagnostics
            try:
                self._save_status()
                self.events.emit('alignment_scoring', **record)
            except Exception:
                # A failing disk/journal must not replace the original error.
                log.exception('Could not save scoring diagnostics for aborted scenario %s',
                              scenario_id)
            raise

    def finish(self, record, session, alignment_target):
        record['completed'] = session.scenario_complete
        if not session.scenario_complete:
            record.update(status='incomplete',
                          reason='Scenario did not complete; no final score requested.')
        else:
            record['final_state'] = final_state_score(
                getattr(session.current_state, 'unstructured', None))
            api, response = self._get_api_score(alignment_target)
            record['api'] = api
            if response is not None:
                # Preserve the existing scores.json API response schema.
                self.scores.append(response)
                if self.enabled:
                    write_json_atomic(self.output_dir / 'scores.json', self.scores)
            available = (api['status'] == 'available'
                         or record['final_state']['status'] == 'available')
            record['status'] = 'available' if available else 'unavailable'
        self._save_status()
        self.events.emit('alignment_scoring', **record)
        if record['status'] == 'available':
            log.info('Alignment scoring for %s: %s', record['scenario_id'], record)
        else:
            log.warning('Alignment scoring for %s: %s', record['scenario_id'], record)

    def _get_api_score(self, alignment_target):
        if alignment_target is None:
            return {'status': 'skipped', 'reason': 'No alignment target configured.'}, None
        if (isinstance(self.interface, TA3CACIActionBasedServiceInterface)
                and self.interface.training_session != 'full'):
            return {
                'status': 'skipped',
                'reason': 'TA3 score API is only called for full training sessions.',
                'training_session': self.interface.training_session,
            }, None
        getter = getattr(self.interface, 'get_session_alignment', None)
        if getter is None:
            return {'status': 'skipped', 'reason': 'Interface has no score retrieval method.'}, None
        try:
            response = getter(alignment_target)
            if response is None:
                return {'status': 'unavailable', 'reason': 'Score API returned no result.'}, None
            result = response if isinstance(response, dict) else response.to_dict()
            log.info('*TA1 Alignment Score*: %s', result)
            return {'status': 'available', 'result': result}, result
        except Exception as exc:
            log.warning('Alignment score retrieval failed for target %s: %s: %s',
                        getattr(alignment_target, 'id', None), type(exc).__name__, exc,
                        exc_info=True)
            return {'status': 'error', 'error_type': type(exc).__name__,
                    'error': str(exc), 'http_status': getattr(exc, 'status', None)}, None
