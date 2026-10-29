"""Logging defaults and compatibility exports for existing agent callers.

Both driver architectures share agent event scopes and transcript capture through
agent_logging. Keep these imports available for existing drivers and Hydra configs.
"""
import logging

from .agent_logging import (
    AgentPlainFormatter, AgentRichHandler,
    agent_log_call, agent_log_event, agent_log_private, agent_log_scope,
    message_payload, preview, visible_text,
)

EXPLAIN_LOG_LEVEL_NUM = 15  # Between DEBUG and INFO
logging.addLevelName(EXPLAIN_LOG_LEVEL_NUM, "EXPLAIN")
LOGGING_FORMAT = "%(message)s"


logging.basicConfig(
    level="EXPLAIN",
    format=LOGGING_FORMAT,
    datefmt="[%X]",
    handlers=[AgentRichHandler()])

# See: https://stackoverflow.com/a/13638084

def explain(self, message, *args, **kws):
    if self.isEnabledFor(EXPLAIN_LOG_LEVEL_NUM):
        # Yes, logger takes its '*args' as 'args'.
        self._log(EXPLAIN_LOG_LEVEL_NUM, message, args, **kws)


logging.Logger.explain = explain
