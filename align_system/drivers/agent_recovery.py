"""Reconcile ambiguous TA3 responses without resubmitting state-changing actions."""

import json


def is_completion_response(error):
    if getattr(error, 'status', None) != 400:
        return False
    body = getattr(error, 'body', None)
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except ValueError:
            pass
    return body == 'Scenario Complete'


def completed_state_after_error(scenario, error):
    """Return a verified final state; never infer completion from an HTTP code.

    A failed POST can have committed on the server. Read back the state once
    before deciding what happened. If readback fails, preserve the original
    error as the cause; neither retry the action nor manufacture a final score.
    """
    if getattr(error, 'status', None) != 500 and not is_completion_response(error):
        return None
    try:
        state = scenario.get_state()
    except Exception as read_error:
        raise read_error from error
    return state if state.scenario_complete else None
