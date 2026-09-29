"""Contract between the LangChain agent drivers and optional
"toolkits" that add tools to the agent.

A toolkit is anything with a ``build_tools(session)`` method returning
a list of LangChain tools; the driver calls it once per scenario with
its per-scenario session object (the current environment state, the
alignment target, and ``refresh_actions()`` for the currently
available actions), and binds the returned tools to the chat model
alongside its own.  An optional ``prompt_hint`` is appended to the
agent's system prompt so the agent knows the extra tools exist and
when to use them.

A toolkit may also implement ``decision_context(session)``: the driver
calls it once per decision point (before each LLM call whose outcome
could be an environment action) and hands any returned text to the
agent as a message.  This is how a toolkit guarantees its information
is in front of the agent at every decision, rather than hoping the
agent remembers to call a tool.

Toolkits are configured under ``configs/driver/tools/`` and attached
to a driver through its ``extra_tools`` mapping, e.g. in an experiment
config:

    defaults:
      - /driver/tools@driver.extra_tools.kaleido: kaleido

Toolkits should defer any heavy model loading until a tool is first
invoked so that composing / instantiating configs stays cheap.
"""


class AgentToolProvider:
    """Base class for optional agent toolkits (see module docstring).
    Subclassing is optional; any object with the same interface works.
    """

    # Text appended to the agent's system prompt when this toolkit is
    # attached; None for no addition
    prompt_hint = None

    def build_tools(self, session):
        """Return a list of LangChain tools bound to the given
        per-scenario driver session."""
        raise NotImplementedError

    def decision_context(self, session):
        """Text to hand to the agent before it decides its next action
        (called once per decision point), or None for nothing."""
        return None
