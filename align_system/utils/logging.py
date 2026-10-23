import logging

from rich.logging import RichHandler

# Default logging levels:
# https://docs.python.org/3/howto/logging.html#logging-levels
# CRITICAL: 50
# ERROR: 40
# WARNING: 30
# INFO: 20
# DEBUG: 10
# NOTSET: 0

EXPLAIN_LOG_LEVEL_NUM = 15  # Between DEBUG and INFO
logging.addLevelName(EXPLAIN_LOG_LEVEL_NUM, "EXPLAIN")

LOGGING_FORMAT = "%(message)s"
logging.basicConfig(
    level="EXPLAIN",
    format=LOGGING_FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler()])

# See: https://stackoverflow.com/a/13638084

def explain(self, message, *args, **kws):
    if self.isEnabledFor(EXPLAIN_LOG_LEVEL_NUM):
        # Yes, logger takes its '*args' as 'args'.
        self._log(EXPLAIN_LOG_LEVEL_NUM, message, args, **kws)


logging.Logger.explain = explain
