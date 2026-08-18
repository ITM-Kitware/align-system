import json
import re
from copy import deepcopy
from timeit import default_timer as timer

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool, ToolException
from pydantic import ValidationError
from rich.highlighter import JSONHighlighter
from swagger_client.models import ActionTypeEnum

from align_system.drivers.itm_open_world import (
    ITMOpenWorldDriver,
    as_dict,
    make_input_output_entry,
)
from align_system.utils import logging
from align_system.utils.action_completion import (
    DEFAULT_TAGS,
    complete_action_parameters,
)


log = logging.getLogger(__name__)
JSON_HIGHLIGHTER = JSONHighlighter()


class _ActionRejectedException(Exception):
    """The environment refused an action (e.g. HTTP 400 from the live
    TA3 server); recoverable by choosing a different action."""


# Action types the (live) environment rejects without a character_id
CHARACTER_REQUIRED_ACTIONS = {
    ActionTypeEnum.CHECK_VITALS,
    ActionTypeEnum.TREAT_PATIENT,
    ActionTypeEnum.MOVE_TO,
    ActionTypeEnum.MOVE_TO_EVAC,
    ActionTypeEnum.TAG_CHARACTER,
}


DEFAULT_LANGCHAIN_AGENT_SYSTEM_PROMPT = """\
You are an autonomous medical triage agent operating in an open-world
mass-casualty simulation.  You interact with the environment
exclusively through the provided tools:

- observe_environment: look at the current scene and casualties
- list_available_actions: see what actions you can currently take
- take_action: carry out one of the listed actions (by its number)

Work in a loop: observe the environment, list the available actions,
reason about which action best serves the casualties, then take it.
After each action, re-observe before deciding what to do next --
the environment changes as you act.

Triage guidance: assess and tag untagged casualties, treat the most
urgent injuries first, and evacuate patients when appropriate.  Always
provide a brief clinical justification when taking an action.  When an
action targets a specific casualty, pass their name as take_action's
character_name argument; when applying a triage tag, pass the category
as the triage_tag argument.

Continue taking actions until you are told the scenario is complete."""


def _format_action_choices(actions):
    """Number the actions for the agent; take_action's action_index
    refers back to this numbering."""
    return "\n".join(f"{idx}: {a.unstructured}"
                     for idx, a in enumerate(actions))


class _AgentScenarioSession:
    """Mutable per-scenario state for the agent: the current environment
    state, manual treated/evac'd patient tracking (see the base driver's
    filtering HACK note), the most recently listed actions, and
    input/output bookkeeping for each executed action."""

    def __init__(self, driver, scenario, alignment_target,
                 sort_available_actions, record_input_output):
        self.driver = driver
        self.scenario = scenario
        self.alignment_target = alignment_target
        self.sort_available_actions = sort_available_actions
        self.record_input_output = record_input_output

        self.current_state = scenario.get_state()
        self.scenario_complete = self.current_state.scenario_complete
        self.treated_patients = set()
        self.evac_patients = set()
        self.available_actions = []
        self.actions_filtered = []
        self.n_actions = 0
        self.times_s = []
        self.decision_start = timer()

    def refresh_actions(self):
        """Re-fetch, expand, and filter the environment's available
        actions, updating `available_actions` / `actions_filtered`."""
        available_actions = self.scenario.get_available_actions()

        if self.sort_available_actions:
            available_actions = sorted(
                available_actions, key=lambda a: a.unstructured)

        expanded, filtered = self.driver._get_expanded_and_filtered_actions(
            self.current_state,
            available_actions,
            self.treated_patients,
            self.evac_patients)

        if len(filtered) == 0:
            # END_SCENE is excluded from the filtered list; once
            # nothing else remains it's the only sensible choice
            filtered = [self.driver._end_scene_fallback_action(expanded)]

        self.available_actions = available_actions
        self.actions_filtered = filtered

        return filtered

    def execute(self, action_to_take, justification=None):
        """Submit an action to the environment, record it, and update
        the session state; raises _ActionRejectedException when the
        environment refuses the action (recoverable by choosing a
        different action)."""
        if justification and getattr(
                action_to_take, 'justification', None) is None:
            action_to_take.justification = justification

        log.info("[bold]*ACTION BEING TAKEN*[/bold]",
                 extra={"markup": True})
        log.info(json.dumps(as_dict(action_to_take), indent=4),
                 extra={"highlighter": JSON_HIGHLIGHTER})

        try:
            if getattr(action_to_take, "intent_action", False):
                current_state = self.scenario.intend_action(action_to_take)
            else:
                current_state = self.scenario.take_action(action_to_take)
        except Exception as e:
            if hasattr(e, 'json'):
                log.info(e.json(indent=2))
            else:
                log.info(str(e))

            if getattr(e, 'status', None) == 400:
                # The environment refused the action (e.g. the
                # targeted character is too far away); recoverable
                # by choosing differently
                raise _ActionRejectedException(
                    str(getattr(e, 'body', e))) from e
            raise e

        # Only successfully executed actions are recorded
        self._record_action(action_to_take)

        if action_to_take.action_type == ActionTypeEnum.TREAT_PATIENT:
            self.treated_patients.add(action_to_take.character_id)
        if action_to_take.action_type == ActionTypeEnum.MOVE_TO_EVAC:
            self.evac_patients.add(action_to_take.character_id)

        self.current_state = current_state
        self.scenario_complete = current_state.scenario_complete
        self.n_actions += 1
        # Listed actions are stale after the environment changes
        self.actions_filtered = []
        self.decision_start = timer()

        return current_state

    def _record_action(self, action_to_take):
        # Called before the session state is updated, so the recorded
        # state/choices are the ones the decision was made against
        self.times_s.append(timer() - self.decision_start)

        action_choice_idx = None
        for i, a in enumerate(self.available_actions):
            if a.action_id == action_to_take.action_id:
                action_choice_idx = i
                break

        choice_info = {
            'langchain_agent': {
                'justification': getattr(action_to_take, 'justification', None),
                'n_actions_taken_in_scenario': self.n_actions}}

        self.record_input_output(make_input_output_entry(
            scenario_id=self.scenario.id(),
            alignment_target_id=(self.alignment_target.id
                                 if self.alignment_target is not None
                                 else None),
            current_state=self.current_state,
            available_actions=self.available_actions,
            choice_info=choice_info,
            action_choice_idx=action_choice_idx,
            action_to_take=action_to_take))


class ITMOpenWorldLangChainDriver(ITMOpenWorldDriver):
    """Open world driver where a LangChain tool-calling agent drives the
    scenario directly.

    Instead of delegating each decision to an ADM, the driver exposes the
    environment to a LangChain tool-calling agent as tools (observe /
    list actions / take action) and lets the agent run its own
    observe -> decide -> act loop until the scenario is complete.  The
    loop is implemented directly with LangChain primitives
    (``chat_model.bind_tools`` plus explicit message handling); no
    cfg.adm is required.

    The underlying LLM is any LangChain chat model: pass an instantiated
    ``chat_model`` (e.g. via a hydra ``_target_``), or a ``model``
    string resolvable by ``langchain.chat_models.init_chat_model`` (e.g.
    ``"ollama:llama3.1"``, ``"openai:gpt-4o"``,
    ``"anthropic:claude-sonnet-4-5"``).  The model must support tool
    calling.
    """

    driver_name = "langchain_agent"

    def __init__(self,
                 chat_model=None,
                 model=None,
                 system_prompt=None,
                 max_actions_per_scenario=100,
                 max_llm_calls_between_actions=8,
                 max_messages_in_context=40,
                 apply_action_filtering=True,
                 expand_actions=False,
                 expand_tagging=False,
                 sort_available_actions=False):
        super().__init__(
            apply_action_filtering=apply_action_filtering,
            expand_actions=expand_actions,
            expand_tagging=expand_tagging,
            sort_available_actions=sort_available_actions)

        # Model resolution is deferred to drive() so that composing /
        # instantiating configs never imports a provider package
        self._chat_model = chat_model
        self._model = model

        if system_prompt is None:
            system_prompt = DEFAULT_LANGCHAIN_AGENT_SYSTEM_PROMPT
        self.system_prompt = system_prompt

        self.max_actions_per_scenario = max_actions_per_scenario
        self.max_llm_calls_between_actions = max_llm_calls_between_actions
        self.max_messages_in_context = max_messages_in_context

    def _resolve_chat_model(self):
        """Resolve and return the configured chat model (idempotent).
        An explicit `model` string takes precedence over a `chat_model`
        block, so `driver.model=...` on the command line overrides a
        config-supplied chat model."""
        if self._model is not None:
            if self._chat_model is not None:
                log.info(f"`model` ({self._model}) overriding the "
                         "configured `chat_model` "
                         f"({type(self._chat_model).__name__})")
                self._chat_model = None

            from langchain.chat_models import init_chat_model
            self._chat_model = init_chat_model(self._model)
            self._model = None

        if self._chat_model is None:
            raise ValueError(
                "No chat model configured for the LangChain agent "
                "driver; set `driver.model` to an init_chat_model "
                "string (e.g. 'openai:gpt-4o') or provide a "
                "`driver.chat_model` block instantiating any "
                "LangChain chat model that supports tool calling")

        return self._chat_model

    def _initialize_run(self, cfg):
        # Resolve the chat model up front so a missing/misconfigured
        # model fails fast, before any session is started
        self._resolve_chat_model()

    @staticmethod
    def _describe_character(character):
        char = as_dict(character)
        # Drop null / internal-only fields to keep observations compact
        return {k: v for k, v in char.items()
                if v is not None and k not in {'has_blanket'}}

    def _observation_text(self, current_state):
        observation = {
            'scene_id': current_state.meta_info.scene_id,
            'situation': current_state.unstructured,
            'casualties': [self._describe_character(c)
                           for c in current_state.characters
                           if not getattr(c, 'unseen', False)],
        }
        if getattr(current_state, 'environment', None) is not None:
            env = current_state.environment
            env_dict = env.to_dict() if hasattr(env, 'to_dict') else env
            observation['environment'] = env_dict

        return json.dumps(observation, indent=2, default=str)

    def _build_tools(self, session):
        """Build the LangChain tools through which the agent interacts
        with the per-scenario `session`."""

        @tool
        def observe_environment() -> str:
            """Observe the current scene: the situation description and
            the casualties (with their injuries, vitals, and triage
            tags)."""
            log.info("[bold]*AGENT OBSERVING ENVIRONMENT*[/bold]",
                     extra={"markup": True})
            return self._observation_text(session.current_state)

        @tool
        def list_available_actions() -> str:
            """List the actions currently available in the environment,
            numbered.  Use the number with take_action to carry one
            out."""
            listing = _format_action_choices(session.refresh_actions())

            log.info("[bold]*AGENT LISTING AVAILABLE ACTIONS*[/bold]",
                     extra={"markup": True})
            log.info(listing)

            return listing

        @tool
        def take_action(action_index: int,
                        justification: str,
                        character_name: str = "",
                        triage_tag: str = "") -> str:
            """Take one of the currently available actions.

            Args:
                action_index: the number of the action from the most
                    recent list_available_actions call
                justification: brief clinical reasoning for why this
                    action was chosen
                character_name: the casualty to target, when the action
                    requires one and doesn't already name a specific
                    casualty
                triage_tag: for tagging actions, the triage category to
                    apply (MINIMAL, DELAYED, IMMEDIATE, or EXPECTANT)
            """
            if not session.actions_filtered:
                return ("No current action list; call "
                        "list_available_actions first (the available "
                        "actions change after every action taken).")

            if not (0 <= action_index < len(session.actions_filtered)):
                return (f"Invalid action_index {action_index}; must be "
                        f"between 0 and {len(session.actions_filtered) - 1}. "
                        "Call list_available_actions to see the current "
                        "options.")

            action_to_take = deepcopy(session.actions_filtered[action_index])

            # Complete required action parameters from the agent's
            # arguments, asking the agent to retry when something the
            # environment requires is missing
            if (action_to_take.action_type in CHARACTER_REQUIRED_ACTIONS
                    and action_to_take.character_id is None):
                visible_characters = [
                    c for c in session.current_state.characters
                    if not getattr(c, 'unseen', False)]

                if not character_name:
                    names = ", ".join(c.name for c in visible_characters)
                    return ("This action requires a target casualty; "
                            "call take_action again with character_name "
                            f"set to one of: {names}")

                matched_character = next(
                    (c for c in visible_characters
                     if character_name.lower() in (c.name.lower(),
                                                   c.id.lower())),
                    None)

                if matched_character is None:
                    names = ", ".join(c.name for c in visible_characters)
                    return (f"Unknown casualty '{character_name}'; "
                            f"valid casualties are: {names}")

                action_to_take.character_id = matched_character.id

            if action_to_take.action_type == ActionTypeEnum.TAG_CHARACTER:
                if action_to_take.parameters is None:
                    action_to_take.parameters = {}

                if 'category' not in action_to_take.parameters:
                    if not triage_tag:
                        return ("Tagging requires a triage category; "
                                "call take_action again with triage_tag "
                                "set to one of: "
                                f"{', '.join(DEFAULT_TAGS)}")

                    matched_tag = next(
                        (t for t in DEFAULT_TAGS
                         if t.lower() == triage_tag.lower()),
                        None)

                    if matched_tag is None:
                        return (f"Unknown triage_tag '{triage_tag}'; "
                                "valid tags are: "
                                f"{', '.join(DEFAULT_TAGS)}")

                    action_to_take.parameters['category'] = matched_tag

            try:
                current_state = session.execute(action_to_take, justification)
            except _ActionRejectedException as e:
                return (f"The environment rejected this action: {e}  "
                        "Choose a different action (for example, you "
                        "may need to move to a casualty before "
                        "assessing or treating them).")

            if current_state.scenario_complete:
                return "Action executed.  SCENARIO COMPLETE -- you are done."

            return ("Action executed.  Updated environment:\n"
                    + self._observation_text(current_state))

        return [observe_environment, list_available_actions, take_action]

    @staticmethod
    def _parse_text_tool_calls(content):
        """Recover tool calls that the model emitted as plain JSON text
        (e.g. '{"name": "take_action", "parameters": {...}}') instead
        of as structured tool calls; some smaller models fall back to
        this style mid-conversation."""
        if isinstance(content, list):
            content = "\n".join(
                part if isinstance(part, str) else part.get('text', '')
                for part in content)
        if not content:
            return []

        text = re.sub(r'```(?:json)?', '', content)

        # Extract top-level {...} blocks with a simple depth counter
        candidates = []
        depth = 0
        start = None
        for i, ch in enumerate(text):
            if ch == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == '}' and depth > 0:
                depth -= 1
                if depth == 0:
                    candidates.append(text[start:i + 1])
                    start = None

        tool_calls = []
        for idx, candidate in enumerate(candidates):
            try:
                obj = json.loads(candidate)
            except json.JSONDecodeError:
                continue

            if not isinstance(obj, dict) or 'name' not in obj:
                continue

            args = obj.get('parameters',
                           obj.get('arguments', obj.get('args', {})))
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    continue
            if not isinstance(args, dict):
                continue

            tool_calls.append({'name': obj['name'],
                               'args': args,
                               'id': f'text-tool-call-{idx}'})

        return tool_calls

    def _trim_message_window(self, messages):
        """Keep the conversation within `max_messages_in_context`
        messages (the system prompt is handled separately by the
        caller).  The window must not start with a ToolMessage (which
        would be an orphaned reply to a trimmed-out assistant
        message)."""
        if len(messages) <= self.max_messages_in_context:
            return list(messages)

        window = list(messages[-self.max_messages_in_context:])
        while window and isinstance(window[0], ToolMessage):
            window.pop(0)

        return window

    def _take_fallback_action(self, session):
        """Take the first available action the environment will accept
        (with heuristically completed parameters), for when the agent
        is spinning without acting."""
        for fallback_candidate in session.refresh_actions():
            fallback_action = complete_action_parameters(
                session.current_state, deepcopy(fallback_candidate),
                character_required_actions=CHARACTER_REQUIRED_ACTIONS)
            try:
                session.execute(
                    fallback_action,
                    justification=("Fallback selection: agent "
                                   "made no progress"))
                return fallback_action
            except _ActionRejectedException as e:
                log.warning("Fallback action rejected by "
                            f"environment: {e}")

        raise RuntimeError("Environment rejected every fallback action")

    def _run_agent_loop(self, session):
        """Run the agent's observe -> decide -> act loop for a single
        scenario using plain LangChain tool calling: the chat model is
        bound to the environment tools and invoked in an explicit
        message loop until the scenario completes (or limits are
        hit)."""
        tools = self._build_tools(session)
        tools_by_name = {t.name: t for t in tools}
        llm_with_tools = self._resolve_chat_model().bind_tools(tools)

        system_message = SystemMessage(content=self.system_prompt)
        messages = [HumanMessage(content=(
            "A new scenario has started.  Observe the environment and "
            "handle the casualties until the scenario is complete."))]

        llm_calls_since_action = 0
        consecutive_llm_failures = 0

        while (not session.scenario_complete
               and session.n_actions < self.max_actions_per_scenario):
            message_window = self._trim_message_window(messages)

            try:
                ai_message = llm_with_tools.invoke(
                    [system_message, *message_window])
                consecutive_llm_failures = 0
            except Exception as e:
                log.error(f"Agent LLM invocation failed: {e}")
                consecutive_llm_failures += 1
                if consecutive_llm_failures >= 3:
                    # The model is unreachable/broken, not merely
                    # indecisive; abort rather than blindly driving
                    # the scenario with fallback actions
                    raise RuntimeError(
                        "Agent LLM unreachable/failing "
                        f"({consecutive_llm_failures} consecutive "
                        f"failures; last error: {e}).  Check that the "
                        "model backend is running and reachable "
                        "(e.g. `ollama serve` for ollama models).") from e
                llm_calls_since_action += 1
                ai_message = None

            if ai_message is not None:
                messages.append(ai_message)

                if ai_message.content:
                    log.info("[bold]*AGENT*[/bold]: {}".format(
                        ai_message.content), extra={"markup": True})

                tool_calls = ai_message.tool_calls
                recovered_from_text = False
                if not tool_calls:
                    tool_calls = self._parse_text_tool_calls(
                        ai_message.content)
                    recovered_from_text = bool(tool_calls)
                    if recovered_from_text:
                        log.info(f"Recovered {len(tool_calls)} tool "
                                 "call(s) from plain-text agent response")

                if not tool_calls:
                    llm_calls_since_action += 1

                    # Some models narrate instead of calling tools;
                    # put the concrete options in front of them
                    choices_block = _format_action_choices(
                        session.refresh_actions())

                    messages.append(HumanMessage(content=(
                        "You did not call any tool, so nothing happened "
                        "in the environment.  The scenario is not yet "
                        "complete.  The currently available actions "
                        f"are:\n{choices_block}\n\nCall the take_action "
                        "tool with the action_index of your chosen "
                        "action (or observe_environment to look "
                        "around).")))
                else:
                    acted = False
                    for tool_call in tool_calls:
                        # Every tool call needs a reply message, even
                        # after the scenario completes mid-batch
                        if session.scenario_complete:
                            result = "Scenario is already complete."
                        elif tool_call['name'] not in tools_by_name:
                            result = (f"Unknown tool: {tool_call['name']}. "
                                      "Available tools: "
                                      f"{', '.join(tools_by_name)}")
                        else:
                            n_actions_before = session.n_actions
                            try:
                                result = tools_by_name[
                                    tool_call['name']].invoke(
                                        tool_call['args'])
                            except (ValidationError, TypeError,
                                    ToolException) as e:
                                # Malformed arguments are fed back to
                                # the agent; environment errors
                                # propagate (as in the base driver)
                                result = f"Tool call failed: {e}"
                            if session.n_actions > n_actions_before:
                                acted = True

                        if recovered_from_text:
                            # Without a structured tool call to reply
                            # to, return the result as a user message
                            messages.append(HumanMessage(content=(
                                f"Result of {tool_call['name']}: "
                                f"{result}")))
                        else:
                            messages.append(ToolMessage(
                                content=str(result),
                                tool_call_id=tool_call['id']))

                    llm_calls_since_action =\
                        0 if acted else llm_calls_since_action + 1

            if (not session.scenario_complete
                    and llm_calls_since_action
                    >= self.max_llm_calls_between_actions):
                # The agent is spinning without acting; take the first
                # available action so a live session can't stall
                log.warning(
                    f"Agent made no progress in "
                    f"{llm_calls_since_action} LLM calls; taking first "
                    "available action as fallback")
                fallback_action = self._take_fallback_action(session)

                llm_calls_since_action = 0
                messages.append(HumanMessage(content=(
                    "You were not making progress, so the following "
                    "action was taken on your behalf: "
                    f"{fallback_action.unstructured}.  Re-observe the "
                    "environment and continue.")))

    def _run_scenario(self, cfg, scenario, alignment_target,
                      sort_available_actions, record_input_output):
        # The agent doesn't align to KDMA targets; the alignment target
        # is only recorded for scoring purposes
        session = _AgentScenarioSession(
            driver=self,
            scenario=scenario,
            alignment_target=alignment_target,
            sort_available_actions=sort_available_actions,
            record_input_output=record_input_output)

        self._run_agent_loop(session)

        if session.n_actions >= self.max_actions_per_scenario:
            log.warning(f"Hit max_actions_per_scenario "
                        f"({self.max_actions_per_scenario}) before "
                        "scenario completion")

        return session.times_s, session.current_state, session.scenario_complete
