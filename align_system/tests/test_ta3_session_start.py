"""A gateway error page must never be sent back to TA3 as a session ID."""
from unittest.mock import Mock

import pytest

from align_system.interfaces.ta3_caci_action_based_service import (
    TA3CACIActionBasedServiceInterface,
)


BLOCKED_RESPONSE = (
    '<p>The requested URL was rejected by the security policy.</p>'
    '<p><strong>Your support ID is: 1285540422468984403</strong></p>')


@pytest.mark.parametrize('response,reason', [
    (BLOCKED_RESPONSE, "security gateway rejected the request"),
    ('<html><body>Service unavailable</body></html>', "HTML instead"),
    ('', "invalid session ID"),
    ('  ', "invalid session ID"),
    (None, "invalid session ID"),
    ({'error': 'Unavailable'}, "invalid session ID"),
])
def test_invalid_start_response_stops_before_scenario(monkeypatch, response, reason):
    connection = Mock()
    connection.start_session.return_value = response
    monkeypatch.setattr(
        'swagger_client.ItmTa2EvalApi', Mock(return_value=connection))
    with pytest.raises(RuntimeError, match=reason) as caught:
        TA3CACIActionBasedServiceInterface()
    assert 'No scenario request was sent' in str(caught.value)
    assert '<' not in str(caught.value)
    if response == BLOCKED_RESPONSE:
        assert '1285540422468984403' in str(caught.value)
    connection.start_session.assert_called_once()
    connection.start_scenario.assert_not_called()


@pytest.mark.parametrize('session_id', [
    'f9d53495-5eb9-4666-9a5e-0793be2fd7c6', 'opaque-session-token',
])
def test_valid_id_is_forwarded_unchanged(monkeypatch, session_id):
    connection = Mock()
    connection.start_session.return_value = session_id
    monkeypatch.setattr(
        'swagger_client.ItmTa2EvalApi', Mock(return_value=connection))
    interface = TA3CACIActionBasedServiceInterface(
        scenario_ids=['April2026-OW_desert3'])
    assert interface.session_id == session_id
    interface.start_scenario()
    connection.start_scenario.assert_called_once_with(
        session_id=session_id, scenario_id='April2026-OW_desert3')
