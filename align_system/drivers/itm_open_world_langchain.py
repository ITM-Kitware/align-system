import json
from copy import deepcopy
from timeit import default_timer as timer

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool, ToolException
from pydantic import ValidationError
from rich.highlighter import JSONHighlighter
from swagger_client.models import ActionTypeEnum

from align_system.drivers.itm_open_world import (
    ActionRejectedException,
    ITMOpenWorldDriver,
    ScenarioSession,
    as_dict,
)
from align_system.utils import logging
from align_system.utils.action_completion import (
    DEFAULT_TAGS,
    VALID_INJURY_LOCATIONS,
    complete_action_parameters,
    in_stock_supplies,
)
from align_system.utils.text_tool_calls import parse_text_tool_calls


log = logging.getLogger(__name__)
JSON_HIGHLIGHTER = JSONHighlighter()


# Action types the (live) environment rejects without a character_id
CHARACTER_REQUIRED_ACTIONS = {
    ActionTypeEnum.CHECK_VITALS,
    ActionTypeEnum.TREAT_PATIENT,
    ActionTypeEnum.MOVE_TO,
    ActionTypeEnum.MOVE_TO_EVAC,
    ActionTypeEnum.TAG_CHARACTER,
}


# The agent-facing tool name for each action type in the pipeline
TOOL_NAMES_BY_ACTION_TYPE = {
    ActionTypeEnum.CHECK_VITALS: 'check_vitals',
    ActionTypeEnum.TREAT_PATIENT: 'treat_patient',
    ActionTypeEnum.TAG_CHARACTER: 'tag_character',
    ActionTypeEnum.MOVE_TO: 'move_to',
    ActionTypeEnum.MOVE_TO_EVAC: 'move_to_evac',
    ActionTypeEnum.SEARCH: 'search',
    ActionTypeEnum.MESSAGE: 'send_message',
    ActionTypeEnum.END_SCENE: 'end_scene',
}


DEFAULT_LANGCHAIN_AGENT_SYSTEM_PROMPT = """\
You are an autonomous medical triage agent operating in an open-world
mass-casualty simulation.  You interact with the environment
exclusively through the provided tools:

- observe_environment: look at the current scene and casualties
- list_available_actions: see which actions the environment currently
  offers
- check_vitals(character_name): assess a casualty's vitals
- treat_patient(character_name, treatment_supply, injury_location):
  treat a casualty's injuries with a supply from your inventory
- tag_character(character_name, triage_tag): apply a triage tag
  (MINIMAL, DELAYED, IMMEDIATE, or EXPECTANT)
- move_to(character_name): move to a casualty
- move_to_evac(character_name): move a casualty to evacuation
- search: search the area for additional casualties
- send_message: deliver the currently offered message/communication
- end_scene: end the current scene once all casualties are handled

Every action tool also takes a justification argument -- always
provide a brief clinical justification for the action you choose.

Not every action is available at every moment; use
list_available_actions when unsure, and if a tool reports that it is
unavailable, choose among the actions it says are available.

Work in a loop: observe the environment, reason about which action
best serves the casualties, then take it.  After each action,
re-observe before deciding what to do next -- the environment changes
as you act.

Triage guidance: assess and tag untagged casualties, treat the most
urgent injuries first, and evacuate patients when appropriate.

Continue taking actions until you are told the scenario is complete."""


def _format_available_actions(actions):
    """Describe each available action as `- tool_name: description` so
    the agent can map what the environment offers onto its tools."""
    return "\n".join(
        f"- {TOOL_NAMES_BY_ACTION_TYPE.get(a.action_type, a.action_type)}: "
        f"{a.unstructured}"
        for a in actions)


class ITMOpenWorldLangChainDriver(ITMOpenWorldDriver):
    """Open world driver where a LangChain tool-calling agent drives the
    scenario directly.

    Instead of delegating each decision to an ADM, the driver exposes the
    environment to a LangChain tool-calling agent as tools -- two
    observation tools (observe_environment / list_available_actions)
    plus one tool per action type in the pipeline (check_vitals,
    treat_patient, tag_character, move_to, move_to_evac, search,
    send_message, end_scene) -- and lets the agent run its own
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
        chat_model = self._resolve_chat_model()

        # Self-managed backends (e.g. VLLMServerChatModel) bring up
        # their server here rather than mid-scenario
        ensure_ready = getattr(chat_model, 'ensure_ready', None)
        if ensure_ready is not None:
            ensure_ready()

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
        supplies = getattr(current_state, 'supplies', None)
        if supplies:
            observation['supplies'] = [
                {'type': s.type, 'quantity': s.quantity} for s in supplies]
        if getattr(current_state, 'environment', None) is not None:
            env = current_state.environment
            observation['environment'] = (
                env.to_dict() if hasattr(env, 'to_dict') else env)

        return json.dumps(observation, indent=2, default=str)

    @staticmethod
    def _execute_agent_action(session, action_to_take, justification):
        """Execute an action with the agent's bookkeeping: decision
        time measured as the wall time since the last executed action,
        and the agent's justification recorded in choice_info."""
        choice_info = {'langchain_agent': {
            'justification': (getattr(action_to_take, 'justification', None)
                              or justification),
            'n_actions_taken_in_scenario': session.n_actions}}

        return session.execute(
            action_to_take, justification,
            choice_info=choice_info,
            decision_time_s=timer() - session.decision_start)

    def _match_candidate_for_character(self, session, tool_name,
                                       action_type, candidates,
                                       character_name):
        """Match one of `candidates` to the casualty named by the
        agent, completing a generic (untargeted) candidate if needed.
        Returns (action, None) on success, or (None, error) where
        `error` is the message to send back to the agent."""
        visible_characters = [
            c for c in session.current_state.characters
            if not getattr(c, 'unseen', False)]

        if not character_name:
            names = ", ".join(c.name for c in visible_characters)
            return None, (f"{tool_name} requires a target casualty; call "
                          "it again with character_name set to one of: "
                          f"{names}")

        matched_character = next(
            (c for c in visible_characters
             if character_name.lower() in (c.name.lower(),
                                           c.id.lower())),
            None)

        if matched_character is None:
            names = ", ".join(c.name for c in visible_characters)
            return None, (f"Unknown casualty '{character_name}'; "
                          f"valid casualties are: {names}")

        # Per-casualty guards mirroring the base driver's action
        # filtering (which can only exclude actions that already
        # name a specific casualty)
        if self.apply_action_filtering:
            if (action_type == ActionTypeEnum.TREAT_PATIENT
                    and matched_character.id in session.treated_patients):
                return None, (f"{matched_character.name} has already been "
                              "treated; choose a different casualty or "
                              "action.")
            if (action_type == ActionTypeEnum.MOVE_TO_EVAC
                    and matched_character.id in session.evac_patients):
                return None, (f"{matched_character.name} has already been "
                              "moved to evac; choose a different casualty "
                              "or action.")
            if (action_type == ActionTypeEnum.TAG_CHARACTER
                    and matched_character.tag is not None):
                return None, (f"{matched_character.name} is already tagged "
                              f"as {matched_character.tag}; choose a "
                              "different casualty or action.")

        # Prefer an action already targeting the casualty (e.g. from
        # per-character expansion), otherwise complete a generic
        # (untargeted) one
        targeted_action = next(
            (a for a in candidates
             if a.character_id == matched_character.id),
            None)

        if targeted_action is not None:
            return deepcopy(targeted_action), None

        generic_action = next(
            (a for a in candidates if a.character_id is None),
            None)

        if generic_action is None:
            character_ids_to_names = {
                c.id: c.name for c in visible_characters}
            targets = ", ".join(sorted(
                {character_ids_to_names.get(a.character_id,
                                            a.character_id)
                 for a in candidates}))
            return None, (f"{tool_name} is not currently available "
                          f"for {matched_character.name}; it is "
                          f"available for: {targets}")

        action_to_take = deepcopy(generic_action)
        action_to_take.character_id = matched_character.id
        return action_to_take, None

    @staticmethod
    def _fill_tag_parameters(action, tool_name, triage_tag):
        """Set the action's triage category from the agent's
        `triage_tag`, in place; the agent's explicit choice overrides
        any category a (tag-expanded) candidate action already
        carries.  Returns an error message for the agent, or None."""
        if action.parameters is None:
            action.parameters = {}

        if not triage_tag:
            if 'category' in action.parameters:
                return None
            return ("Tagging requires a triage category; call "
                    f"{tool_name} again with triage_tag set to "
                    f"one of: {', '.join(DEFAULT_TAGS)}")

        matched_tag = next(
            (t for t in DEFAULT_TAGS
             if t.lower() == triage_tag.lower()),
            None)

        if matched_tag is None:
            return (f"Unknown triage_tag '{triage_tag}'; "
                    f"valid tags are: {', '.join(DEFAULT_TAGS)}")

        action.parameters['category'] = matched_tag
        return None

    @staticmethod
    def _fill_treatment_parameters(session, action, tool_name,
                                   treatment_supply, injury_location):
        """Set the action's treatment supply and injury location from
        the agent's arguments, in place; explicit arguments override
        any parameters a candidate action already carries.  The (live)
        environment errors on TREAT_PATIENT without them, but they are
        only enforceable when the state reports supplies.  Returns an
        error message for the agent, or None."""
        supplies = in_stock_supplies(session.current_state)
        if not supplies:
            return None

        if action.parameters is None:
            action.parameters = {}

        supplies_listing = ", ".join(
            f"{s.type} (x{s.quantity})" if s.quantity is not None
            else str(s.type)
            for s in supplies)

        if treatment_supply:
            matched_supply = next(
                (s.type for s in supplies
                 if str(s.type).lower() == treatment_supply.lower()),
                None)

            if matched_supply is None:
                return ("Unknown or out-of-stock "
                        f"treatment_supply '{treatment_supply}'; "
                        f"available supplies are: {supplies_listing}")

            action.parameters['treatment'] = matched_supply
        elif 'treatment' not in action.parameters:
            return ("Treating requires choosing a supply; "
                    f"call {tool_name} again with "
                    "treatment_supply set to one of: "
                    f"{supplies_listing} -- and "
                    "injury_location set to the injury's "
                    "location (e.g. 'left calf', 'right "
                    "thigh', 'center chest'; 'unspecified' "
                    "if unclear)")

        if injury_location:
            matched_location = next(
                (loc for loc in VALID_INJURY_LOCATIONS
                 if loc.lower() == injury_location.lower()),
                None)

            if matched_location is None:
                return ("Unknown injury_location "
                        f"'{injury_location}'; valid "
                        "locations are: "
                        f"{', '.join(VALID_INJURY_LOCATIONS)}")

            action.parameters['location'] = matched_location
        elif 'location' not in action.parameters:
            action.parameters['location'] = 'unspecified'

        return None

    def _perform_typed_action(self, session, action_type, justification,
                              character_name="", triage_tag="",
                              treatment_supply="", injury_location=""):
        """Carry out an action of `action_type` on behalf of one of the
        per-action-type tools: re-fetch what the environment currently
        offers, match/complete an action of that type from the agent's
        arguments, and execute it.  Returns the string result for the
        agent."""
        tool_name = TOOL_NAMES_BY_ACTION_TYPE[action_type]

        session.refresh_actions()
        candidates = [a for a in session.actions_filtered
                      if a.action_type == action_type]

        if not candidates:
            if (action_type == ActionTypeEnum.END_SCENE
                    and any(a.action_type == ActionTypeEnum.END_SCENE
                            for a in session.actions_expanded)):
                # END_SCENE is held back by action filtering until no
                # other (filtered) actions remain
                return ("Cannot end the scene yet; there are still "
                        "actions to complete first:\n"
                        + _format_available_actions(
                            session.actions_filtered))

            return (f"{tool_name} is not currently available.  The "
                    "currently available actions are:\n"
                    + _format_available_actions(session.actions_filtered))

        if action_type in CHARACTER_REQUIRED_ACTIONS:
            action_to_take, error = self._match_candidate_for_character(
                session, tool_name, action_type, candidates,
                character_name)
            if error is not None:
                return error
        else:
            if len(candidates) > 1:
                log.info(f"{tool_name}: multiple candidate actions "
                         "offered by the environment; taking the first "
                         f"('{candidates[0].unstructured}')")
            action_to_take = deepcopy(candidates[0])

        if action_to_take.action_type == ActionTypeEnum.TAG_CHARACTER:
            error = self._fill_tag_parameters(
                action_to_take, tool_name, triage_tag)
            if error is not None:
                return error

        if action_to_take.action_type == ActionTypeEnum.TREAT_PATIENT:
            error = self._fill_treatment_parameters(
                session, action_to_take, tool_name,
                treatment_supply, injury_location)
            if error is not None:
                return error

        try:
            current_state = self._execute_agent_action(
                session, action_to_take, justification)
        except ActionRejectedException as e:
            return (f"The environment rejected this action: {e}  "
                    "Choose a different action (for example, you may "
                    "need to move_to a casualty before assessing or "
                    "treating them).")

        if current_state.scenario_complete:
            return "Action executed.  SCENARIO COMPLETE -- you are done."

        return ("Action executed.  Updated environment:\n"
                + self._observation_text(current_state))

    def _build_tools(self, session):
        """Build the LangChain tools through which the agent interacts
        with the per-scenario `session`: two observation tools plus one
        tool per action type in the pipeline (see
        TOOL_NAMES_BY_ACTION_TYPE)."""
        driver = self

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
            """List the actions the environment currently offers, named
            by the tool that carries each one out."""
            listing = _format_available_actions(session.refresh_actions())

            log.info("[bold]*AGENT LISTING AVAILABLE ACTIONS*[/bold]",
                     extra={"markup": True})
            log.info(listing)

            return listing

        @tool
        def check_vitals(character_name: str, justification: str) -> str:
            """Check the vitals of a casualty.

            Args:
                character_name: the casualty whose vitals to check
                justification: brief clinical reasoning for this action
            """
            return driver._perform_typed_action(
                session, ActionTypeEnum.CHECK_VITALS, justification,
                character_name=character_name)

        @tool
        def treat_patient(character_name: str, justification: str,
                          treatment_supply: str = "",
                          injury_location: str = "") -> str:
            """Treat a casualty's injuries with a supply from your
            inventory.

            Args:
                character_name: the casualty to treat
                justification: brief clinical reasoning for this action
                treatment_supply: the supply to treat with (one of the
                    supplies listed in your observation, e.g.
                    'Tourniquet', 'Pressure bandage', 'Hemostatic
                    gauze')
                injury_location: where on the body the injury being
                    treated is (e.g. 'left calf', 'right thigh',
                    'center chest'; 'unspecified' if unclear)
            """
            return driver._perform_typed_action(
                session, ActionTypeEnum.TREAT_PATIENT, justification,
                character_name=character_name,
                treatment_supply=treatment_supply,
                injury_location=injury_location)

        @tool
        def tag_character(character_name: str, triage_tag: str,
                          justification: str) -> str:
            """Apply a triage tag to a casualty.

            Args:
                character_name: the casualty to tag
                triage_tag: the triage category to apply (MINIMAL,
                    DELAYED, IMMEDIATE, or EXPECTANT)
                justification: brief clinical reasoning for this action
            """
            return driver._perform_typed_action(
                session, ActionTypeEnum.TAG_CHARACTER, justification,
                character_name=character_name, triage_tag=triage_tag)

        @tool
        def move_to(character_name: str, justification: str) -> str:
            """Move to a casualty (often required before they can be
            assessed or treated).

            Args:
                character_name: the casualty to move to
                justification: brief clinical reasoning for this action
            """
            return driver._perform_typed_action(
                session, ActionTypeEnum.MOVE_TO, justification,
                character_name=character_name)

        @tool
        def move_to_evac(character_name: str, justification: str) -> str:
            """Move a casualty to evacuation.

            Args:
                character_name: the casualty to evacuate
                justification: brief clinical reasoning for this action
            """
            return driver._perform_typed_action(
                session, ActionTypeEnum.MOVE_TO_EVAC, justification,
                character_name=character_name)

        @tool
        def search(justification: str) -> str:
            """Search the area for additional casualties.

            Args:
                justification: brief reasoning for this action
            """
            return driver._perform_typed_action(
                session, ActionTypeEnum.SEARCH, justification)

        @tool
        def send_message(justification: str) -> str:
            """Deliver the message/communication the environment
            currently offers.

            Args:
                justification: brief reasoning for this action
            """
            return driver._perform_typed_action(
                session, ActionTypeEnum.MESSAGE, justification)

        @tool
        def end_scene(justification: str) -> str:
            """End the current scene.  Only do this once all casualties
            have been assessed, tagged, and treated as appropriate.

            Args:
                justification: brief reasoning for this action
            """
            return driver._perform_typed_action(
                session, ActionTypeEnum.END_SCENE, justification)

        return [observe_environment, list_available_actions,
                check_vitals, treat_patient, tag_character, move_to,
                move_to_evac, search, send_message, end_scene]

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
            fallback_action = deepcopy(fallback_candidate)
            complete_action_parameters(
                session.current_state, fallback_action,
                character_required_actions=CHARACTER_REQUIRED_ACTIONS)
            try:
                self._execute_agent_action(
                    session, fallback_action,
                    justification=("Fallback selection: agent "
                                   "made no progress"))
                return fallback_action
            except ActionRejectedException as e:
                log.warning("Fallback action rejected by "
                            f"environment: {e}")

        raise RuntimeError("Environment rejected every fallback action")

    def _handle_tool_calls(self, session, tools_by_name, tool_calls,
                           recovered_from_text, messages):
        """Invoke each of the agent's tool calls, appending a reply
        message for every call (a ToolMessage, or a HumanMessage for
        calls recovered from plain text).  Returns whether any call
        resulted in an environment action being taken."""
        acted = False
        for tool_call in tool_calls:
            log.info("[bold]*AGENT TOOL CALL*[/bold]: "
                     "{}({})".format(
                         tool_call['name'],
                         json.dumps(tool_call['args'], default=str)),
                     extra={"markup": True})

            # Every tool call needs a reply message, even after the
            # scenario completes mid-batch
            if session.scenario_complete:
                result = "Scenario is already complete."
            elif tool_call['name'] not in tools_by_name:
                result = (f"Unknown tool: {tool_call['name']}. "
                          "Available tools: "
                          f"{', '.join(tools_by_name)}")
                log.warning("Agent called unknown tool "
                            f"'{tool_call['name']}'")
            else:
                n_actions_before = session.n_actions
                try:
                    result = tools_by_name[tool_call['name']].invoke(
                        tool_call['args'])
                except (ValidationError, TypeError, ToolException) as e:
                    # Malformed arguments are fed back to the agent;
                    # environment errors propagate (as in the base
                    # driver)
                    result = f"Tool call failed: {e}"
                    log.warning(f"Tool call failed: {e}")
                if session.n_actions > n_actions_before:
                    acted = True

            if recovered_from_text:
                # Without a structured tool call to reply to, return
                # the result as a user message
                messages.append(HumanMessage(content=(
                    f"Result of {tool_call['name']}: {result}")))
            else:
                messages.append(ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call['id']))

        return acted

    def _run_agent_loop(self, session):
        """Run the agent's observe -> decide -> act loop for a single
        scenario using plain LangChain tool calling: the chat model is
        bound to the environment tools and invoked in an explicit
        message loop until the scenario completes (or limits are
        hit)."""
        tools = self._build_tools(session)
        tools_by_name = {t.name: t for t in tools}
        llm_with_tools = self._resolve_chat_model().bind_tools(tools)

        log.info("[bold]*AGENT TOOLS*[/bold]", extra={"markup": True})
        # First paragraph of each tool's description (the rest is
        # argument documentation)
        log.info("\n".join(
            "- {}: {}".format(
                t.name, " ".join(t.description.split("\n\n")[0].split()))
            for t in tools))

        system_message = SystemMessage(content=self.system_prompt)
        messages = [HumanMessage(content=(
            "A new scenario has started.  Observe the environment and "
            "handle the casualties until the scenario is complete."))]

        log.info("[bold]*AGENT SYSTEM PROMPT*[/bold]",
                 extra={"markup": True})
        log.info(self.system_prompt)
        log.info("[bold]*AGENT INITIAL PROMPT*[/bold]",
                 extra={"markup": True})
        log.info(messages[0].content)

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

                # Reasoning models' think phase (e.g. ChatOllama with
                # reasoning: true routes it here)
                thinking = ai_message.additional_kwargs.get(
                    'reasoning_content')
                if thinking:
                    log.info("[bold]*AGENT THINKING*[/bold]: {}".format(
                        thinking), extra={"markup": True})

                if ai_message.content:
                    log.info("[bold]*AGENT*[/bold]: {}".format(
                        ai_message.content), extra={"markup": True})

                tool_calls = ai_message.tool_calls
                recovered_from_text = False
                if not tool_calls:
                    tool_calls = parse_text_tool_calls(ai_message.content)
                    recovered_from_text = bool(tool_calls)
                    if recovered_from_text:
                        log.info(f"Recovered {len(tool_calls)} tool "
                                 "call(s) from plain-text agent response")

                if not tool_calls:
                    llm_calls_since_action += 1

                    log.warning(
                        "Agent response contained no tool calls"
                        + ("" if ai_message.content
                           else " (and no content)"))

                    # Some models narrate instead of calling tools;
                    # put the concrete options in front of them
                    choices_block = _format_available_actions(
                        session.refresh_actions())

                    messages.append(HumanMessage(content=(
                        "You did not call any tool, so nothing happened "
                        "in the environment.  The scenario is not yet "
                        "complete.  The currently available actions "
                        f"are:\n{choices_block}\n\nCall the named tool "
                        "for your chosen action, with a justification "
                        "(or observe_environment to look around).")))
                else:
                    acted = self._handle_tool_calls(
                        session, tools_by_name, tool_calls,
                        recovered_from_text, messages)

                    llm_calls_since_action = (
                        0 if acted else llm_calls_since_action + 1)

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
        session = ScenarioSession(
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
