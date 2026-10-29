"""Custom exceptions for the align-system."""


class ContextWindowError(RuntimeError):
    """The prompt cannot be safely budgeted; repeating it will not help."""


class ScenarioStalledError(RuntimeError):
    """No valid recovery action succeeded in the current scenario.

    ``diagnostics`` contains replayable state and rejected action details.
    A stalled scenario has no final score and must not silently advance a
    server session that offers no explicit abort operation.
    """

    def __init__(self, message, diagnostics=None):
        self.diagnostics = diagnostics if diagnostics is not None else {}
        super().__init__(message)


class ScenarioActionLimitError(ScenarioStalledError):
    """The action budget ended with an active scenario; do not advance it."""


class SceneSkipException(Exception):
    """
    Exception raised when a scene should be skipped and the system should move to the next scene.

    This exception is typically raised when a component (e.g., ExpressStageComponent) fails
    after all retry attempts and cannot recover. Instead of crashing the entire run, the
    system will skip the current scene/probe and continue with the next one.

    Attributes:
        message: Explanation of why the scene is being skipped
        component_name: Name of the component that raised the exception
        last_error: The underlying error that caused the skip (optional)
    """

    def __init__(self, message: str, component_name: str | None, last_error: Exception | None):
        self.message = message
        self.component_name = component_name
        self.last_error = last_error
        super().__init__(self.message)

    def __str__(self):
        if self.component_name:
            return f"[{self.component_name}] {self.message}"
        return self.message


class ActionRejectedException(Exception):
    """A recoverable action rejection reported by an environment harness."""
