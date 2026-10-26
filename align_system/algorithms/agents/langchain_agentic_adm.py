"""Tool-calling decision ADM with optional advisory review.

``run_scenario(session, tools, tool_providers)`` is the agentic entry point: the
harness supplies native tools and environment access while this ADM owns the
conversation, proposals, reviewer calls and action-selection policy.
"""

import json
from copy import deepcopy
from dataclasses import dataclass, field
from timeit import default_timer as timer

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import ToolException, StructuredTool
from pydantic import Field, ValidationError, create_model
from pydantic.json_schema import SkipJsonSchema
from swagger_client.models import ActionTypeEnum

from align_system.algorithms.random_adm_component import OWRandomParameterCompletionADMComponent
from align_system.algorithms.agents.agent_alignment import build_alignment_context
from align_system.algorithms.agents.agent_context_budget import (
    ContextBudgetError, ContextBudgetManager, is_context_overflow,
)
from align_system.algorithms.agents.agent_history import pack_history
from align_system.algorithms.agents.agent_evidence import build_citation_guide
from align_system.algorithms.agents.agent_review import (
    ActionReviewer, validate_reviewer_contract,
)
from align_system.algorithms.agents.agent_workflow import AdvisoryReviewWorkflow, ImmediateActionWorkflow
from align_system.exceptions import ActionRejectedException
from align_system.prompt_engineering.agentic_prompts import (
    DEFAULT_LANGCHAIN_AGENT_SYSTEM_PROMPT, DECISION_ALIGNMENT_GUIDANCE,
    SCENE_END_GUIDANCE, PATIENT_HISTORY_GUIDANCE, CITATION_GUIDANCE,
    DEFAULT_AGENT_OPENING_PROMPT,
)
from align_system.utils import logging
from align_system.utils.agent_logging import (
    agent_log_call, agent_log_event, agent_log_private, agent_log_scope,
    message_payload, preview, visible_text,
)
from align_system.utils.text_tool_calls import parse_text_tool_calls

log = logging.getLogger(__name__)
MAX_SERIALIZED_REVIEW_RESPONSE_CHARS = 256 * 1024

def _decision_log_payload(messages, bound_model, tools):
    """Reuse bound schemas when available; never serialize the model itself."""
    bound_kwargs = getattr(bound_model, 'kwargs', None)
    schemas = bound_kwargs.get('tools') if isinstance(bound_kwargs, dict) else None
    if schemas is None:
        schemas = []
        for item in tools:
            schema = item.args_schema
            if schema is None:
                schema = {'type': 'object', 'properties': item.args}
            elif not isinstance(schema, dict):
                schema = schema.model_json_schema()
            schemas.append({'name': item.name, 'description': item.description,
                            'parameters': schema})
    return {'messages': message_payload(messages), 'tools': schemas}


@dataclass
class _ScenarioState:
    workflow: object = field(default_factory=ImmediateActionWorkflow)
    alignment_context: dict | None = None
    context_provider_results: dict = field(default_factory=dict)
    action_tools: dict = field(default_factory=dict)
    extra_tool_names: set = field(default_factory=set)


class LangChainAgenticADM:
    def __init__(self, chat_model=None, model=None, system_prompt=None,
                 max_llm_calls_between_actions=8, max_messages_in_context=40,
                 reviewer=None, context_budget=None, attribute_definitions=None,
                 alignment_guidance=None, scene_end_guidance=None,
                 patient_history_guidance=None, review_guidance=None,
                 expected_response_contract=None,
                 citation_guidance=None, opening_prompt=None):
        self._chat_model = chat_model
        self._model = model
        self.system_prompt = (DEFAULT_LANGCHAIN_AGENT_SYSTEM_PROMPT
                              if system_prompt is None else system_prompt)
        self.max_llm_calls_between_actions = max_llm_calls_between_actions
        self.max_messages_in_context = max_messages_in_context
        self.reviewer = reviewer
        self.context_budget = ContextBudgetManager(context_budget)
        self.attribute_definitions = dict(attribute_definitions or {})
        self.alignment_guidance = (DECISION_ALIGNMENT_GUIDANCE if alignment_guidance is None
                                   else alignment_guidance)
        self.scene_end_guidance = dict(SCENE_END_GUIDANCE if scene_end_guidance is None
                                       else scene_end_guidance)
        self.patient_history_guidance = (PATIENT_HISTORY_GUIDANCE
                                         if patient_history_guidance is None
                                         else patient_history_guidance)
        self.review_guidance = review_guidance
        self.citation_guidance = (CITATION_GUIDANCE if citation_guidance is None
                                  else citation_guidance)
        self.opening_prompt = (DEFAULT_AGENT_OPENING_PROMPT if opening_prompt is None
                               else opening_prompt)
        self.expected_response_contract = validate_reviewer_contract(
            self.reviewer, expected_response_contract)
        if reviewer is not None and max_llm_calls_between_actions < 1:
            raise ValueError("Reviewer mode requires a positive model-call budget")
        self._parameter_completer = OWRandomParameterCompletionADMComponent()
        self._state = None
        self.extra_tools = {}

    @property
    def architecture(self):
        return 'decision_agent_reviewer' if self.reviewer is not None else None

    def load_model(self):
        """Resolve configuration before the harness starts a live scenario."""
        validate_reviewer_contract(self.reviewer, self.expected_response_contract)
        return self._resolve_chat_model()

    def run_scenario(self, session, tools, tool_providers=None):
        """Start fresh conversation/review state for one harness session."""
        self._run_agent_loop(session, tools, tool_providers)
        with agent_log_scope(**session.log_context()):
            if (not session.scenario_complete
                    and session.n_actions >= session.max_actions_per_scenario):
                log.warning("Hit max_actions_per_scenario (%s) before scenario "
                            "completion; ending scene", session.max_actions_per_scenario)
                session.refresh_actions()
                self._force_end_scene(session, "Hit max_actions_per_scenario; ending scene",
                                      reason='action_cap')

    def _run_agent_loop(self, session, tools, tool_providers=None):
        decision_model, tools = self.prepare_scenario(session, tools, tool_providers)
        with agent_log_scope(reset_calls=True, **session.log_context()):
            self._run_agent_loop_impl(session, decision_model, tools)

    def prepare_scenario(self, session, tools, tool_providers=None):
        """Bind a fresh decision/review workflow to the harness's native tools."""
        self._state = _ScenarioState()
        self.extra_tools = dict(tool_providers or {})
        self._prepare_session(session)
        decision_model = self._resolve_chat_model()
        review_budget = None
        if isinstance(self.reviewer, ActionReviewer):
            review_model = (self.reviewer.chat_model if self.reviewer.chat_model is not None
                            else decision_model)
            review_budget = self.context_budget.for_model(review_model, 'reviewer')
        self._state.workflow = (
            ImmediateActionWorkflow() if self.reviewer is None else
            AdvisoryReviewWorkflow(
                self.reviewer, decision_model, self.max_llm_calls_between_actions,
                actions=session, review_request_budget=review_budget,
                expected_response_contract=self.expected_response_contract,
                system_prompt_suffix=self.review_guidance))
        return decision_model, self._configure_tools(session, tools)

    def _configure_tools(self, session, tools):
        native_schemas = session.native_action_schemas
        action_types = session.action_tool_names
        if self.reviewer is not None:
            names = [tool.name for tool in tools]
            if len(names) != len(set(names)):
                raise ValueError("Reviewer mode requires unique tool names")
        wrapped = []
        for item in tools:
            if (item.name in action_types
                    and item is session.native_tools.get(item.name)):
                item = self._action_tool(session, item, action_types[item.name])
                self._state.action_tools[item.name] = item
            wrapped.append(item)
        self._state.extra_tool_names = {
            item.name for item in tools if item is not session.native_tools.get(item.name)}
        self._state.workflow.configure_action_schemas(native_schemas)
        return wrapped

    def _action_tool(self, session, native_tool, action_type):
        schema = native_tool.args_schema
        if self.reviewer is not None:
            # qwen3_xml needs an object property rather than anyOf(object, null).
            schema = create_model(
                schema.__name__ + 'Reviewed', __base__=schema,
                review_response=(self.expected_response_contract.final_response_model
                                 | SkipJsonSchema[None], Field(
                    default=None,
                    description="Required on final choices after usable review; omit on proposals.")))

        def perform(**arguments):
            return self._perform_typed_action(session, action_type, **arguments)

        return StructuredTool.from_function(
            func=perform, name=native_tool.name, description=native_tool.description,
            args_schema=schema)

    def _resolve_chat_model(self):
        """Resolve and return the configured chat model (idempotent).
        An explicit `model` string takes precedence over a `chat_model`
        block, so `adm.instance.model=...` on the command line overrides a
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
                "ADM; set `adm.instance.model` to an init_chat_model "
                "string (e.g. 'openai:gpt-4o') or provide a "
                "`adm.instance.chat_model` block instantiating any "
                "LangChain chat model that supports tool calling")

        return self._chat_model

    def _execute_agent_action(self, session, action_to_take, justification,
                              driver_override=None):
        """Execute an action with the agent's bookkeeping: decision
        time measured as the wall time since the last executed action,
        and the agent's justification recorded in choice_info."""
        choice_info = {'langchain_agent': {
            'justification': (getattr(action_to_take, 'justification', None)
                              or justification),
            'parameter_validation': session.parameter_validation(
                action_to_take, checked=not driver_override),
            'n_actions_taken_in_scenario': session.n_actions}}

        metadata = self._state.workflow.execution_metadata(
            action_to_take, justification, driver_override)
        if metadata is not None:
            choice_info['multi_agent'] = metadata

        if driver_override or metadata is None:
            agent_log_event(
                log, 'WORKFLOW' if driver_override else 'DECISION',
                'OVERRIDE' if driver_override else 'FINAL SELECTION',
                f'action={session.enum_value(action_to_take.action_type)} '
                f'patient={getattr(action_to_take, "character_id", None)} '
                f'override={driver_override or "none"}',
                **session.log_context())

        current_state = session.execute(
            action_to_take, justification,
            choice_info=choice_info,
            decision_time_s=timer() - session.decision_start)
        self._state.context_provider_results.clear()
        self._state.workflow.action_succeeded(driver_override=driver_override)
        return current_state

    def _perform_typed_action(self, session, action_type, justification,
                              character_name="", triage_tag="",
                              treatment_supply="", injury_location="",
                              review_response=None):
        """Freshly resolve a proposal or final selection, then review/execute."""
        session.refresh_actions()
        snapshot = session.snapshot()
        action_to_take, error = session.resolve_action(
            session.action_types[action_type], {
                'justification': justification, 'character_name': character_name,
                'triage_tag': triage_tag, 'treatment_supply': treatment_supply,
                'injury_location': injury_location}, snapshot=snapshot)
        if error is not None:
            agent_log_event(
                log, 'WORKFLOW', 'CORRECTION',
                f'stage=action_resolution {preview(error)}',
                payload=error, **session.log_context())
            return error

        context = {}
        if self._state.workflow.needs_proposal_context:
            context = self._decision_review_context(
                session, state=snapshot.current_state, include_guide=False)
        review_feedback = self._state.workflow.before_action(
            snapshot, action_to_take, justification, context=context,
            review_response=review_response)
        if review_feedback is not None:
            return review_feedback

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
                + json.dumps(session.observe(current_state), indent=2))

    def _force_end_scene(self, session, justification, reason="scene_complete"):
        """Submit END_SCENE (if the environment offers it) on the
        agent's behalf: once the session reports `scene_done`, or after
        the per-scenario action cap; leaving a scene open makes the
        (live) server refuse to start the next scenario.  Expects
        `session.available_actions` to be fresh."""
        end_scene_action = next(
            (deepcopy(a) for a in session.available_actions
             if a.action_type == ActionTypeEnum.END_SCENE), None)
        if end_scene_action is None:
            agent_log_event(
                log, 'WORKFLOW', 'OVERRIDE UNAVAILABLE',
                f'reason={reason}; END_SCENE is not offered',
                level=logging.WARNING, **session.log_context())
            return

        end_scene_action.justification = justification
        try:
            self._execute_agent_action(
                session, end_scene_action, justification, driver_override=reason)
        except ActionRejectedException:
            agent_log_event(
                log, 'WORKFLOW', 'OVERRIDE REJECTED', f'reason={reason}',
                level=logging.WARNING, **session.log_context())

    def _take_fallback_action(self, session):
        """Take the first available action the environment will accept
        (with randomly completed parameters), for when the agent is
        spinning without acting."""
        justification = "Fallback selection: agent made no progress"
        agent_log_event(
            log, 'WORKFLOW', 'FALLBACK', 'reason=no_progress',
            level=logging.WARNING, **session.log_context())

        # Prefer anything over ending the scene (stable sort keeps the
        # environment's ordering otherwise); if filtering leaves nothing
        # sensible, anything the environment offers is better than
        # stalling
        fallback_candidates = sorted(
            session.refresh_actions() or session.available_actions,
            key=lambda a: a.action_type == ActionTypeEnum.END_SCENE)
        for fallback_candidate in fallback_candidates:
            fallback_action = deepcopy(fallback_candidate)
            # The shared random completer does not handle these OW action
            # types. Complete generic offers locally before submission; an
            # explicit target supplied by the environment stays authoritative.
            if (fallback_action.action_type in {
                    ActionTypeEnum.CHECK_VITALS, ActionTypeEnum.MOVE_TO}
                    and fallback_action.character_id is None):
                targets = [c for c in session.current_state.characters
                           if not getattr(c, 'unseen', False) and c.id]
                if fallback_action.action_type == ActionTypeEnum.CHECK_VITALS:
                    targets = [c for c in targets
                               if getattr(c, 'nearby', None) is not False]
                else:
                    # Prefer movement that changes proximity without making
                    # unknown proximity an eligibility failure.
                    distant = [c for c in targets
                               if getattr(c, 'nearby', None) is False]
                    targets = distant or targets
                if not targets:
                    log.warning("Skipping fallback %s: no visible target "
                                "available for this action.",
                                fallback_action.action_type)
                    continue
                fallback_action.character_id = targets[0].id
            fallback_action = self._parameter_completer.run(
                session.current_state, [], [], None,
                chosen_action=fallback_action)
            fallback_action.justification = justification

            if fallback_action.action_type == ActionTypeEnum.TREAT_PATIENT:
                # The parameter completer leaves treatment / location
                # unset, which the (live) server rejects; use the first
                # in-stock supply
                supplies = session.in_stock_supplies(session.current_state)
                if supplies and not (fallback_action.parameters or {}).get(
                        'treatment'):
                    parameters = dict(fallback_action.parameters or {})
                    parameters.update(
                        treatment=str(session.enum_value(supplies[0].type)),
                        location='unspecified')
                    fallback_action.parameters = parameters

            try:
                self._execute_agent_action(
                    session, fallback_action, justification,
                    driver_override='fallback')
                return fallback_action
            except ActionRejectedException:
                agent_log_event(
                    log, 'WORKFLOW', 'FALLBACK REJECTED',
                    f'action={session.enum_value(fallback_action.action_type)}',
                    level=logging.WARNING, **session.log_context())

        raise RuntimeError("No fallback action could be executed")

    def _decision_review_context(self, session, *, state=None, include_guide=True):
        """Visible local facts only; capture without fetching or scoring again."""
        context = {
            'scenario_id': session.scenario.id(),
            'scene_id': session.current_state.meta_info.scene_id,
            'action_index': session.n_actions,
            'observation': session.observe(state),
            'patient_history': session.patient_history.snapshot(),
            'scene_end_policy': session.scene_end_policy,
            'available_actions': session.format_available_actions(session.actions_filtered),
            'handled_patients': {
                'treated': sorted(session.treated_patients),
                'evacuated': sorted(session.evac_patients),
                'tagged': sorted(session.tagged_patients)},
            'provider_context': deepcopy(self._state.context_provider_results),
        }
        if self._state.alignment_context is not None:
            context['alignment_context'] = deepcopy(self._state.alignment_context)
        if include_guide:
            context['citation_guide'] = build_citation_guide(context)
        return context

    @staticmethod
    def _current_context_message(context):
        return HumanMessage(content=(
            'Current visible state, patient history, and cached provider context:\n'
            + json.dumps(context, ensure_ascii=False, separators=(',', ':'))))

    def _budget_decision_request(self, session, budget, system_message, messages,
                                 pending, *, force_minimal=False):
        """Pack complete exchanges against the actual request, without I/O
        to the environment or changes to the evidence/attempt state."""
        measurement = None
        measurements = {}

        def fits(window):
            nonlocal measurement
            key = tuple(id(message) for message in window)
            if key not in measurements:
                measurements[key] = budget.measure([system_message, *window])
            measurement = measurements[key]
            return measurement.fits

        packed = pack_history(
            messages, max_messages=self.max_messages_in_context, fits=fits,
            pending_messages=pending,
            force_minimal=force_minimal)
        diagnostics = {
            **measurement.diagnostics, 'role': 'decision',
            'action_index': session.n_actions,
            'removed_exchanges': packed.removed_exchanges,
            'removed_messages': packed.removed_messages,
            'recovery': 'minimal_history' if force_minimal else 'none',
        }
        agent_log_event(
            log, 'DECISION', 'CONTEXT BUDGET',
            f'input_tokens={measurement.input_tokens} '
            f'removed_exchanges={packed.removed_exchanges} '
            f'recovery={diagnostics["recovery"]}',
            payload=diagnostics,
            legacy_message='Context budget: %s', legacy_args=(json.dumps(diagnostics),),
            **session.log_context())
        if not packed.fits:
            raise ContextBudgetError(
                'Mandatory decision context cannot fit the model context budget; '
                'no environment action was submitted.', diagnostics=diagnostics)
        return [system_message, *packed.messages], measurement

    @staticmethod
    def _ordered_tool_calls(message):
        """Include malformed native calls so they consume slots and get replies."""
        invalid = [dict(call, invalid_arguments=True)
                   for call in message.invalid_tool_calls]
        if not invalid:
            return message.tool_calls
        calls = [*invalid, *message.tool_calls]
        raw_calls = message.additional_kwargs.get('tool_calls', [])
        by_id = {call['id']: call for call in calls}
        raw_ids = [call.get('id') for call in raw_calls if isinstance(call, dict)]
        if (len(by_id) == len(calls) == len(raw_ids)
                and len(set(raw_ids)) == len(raw_ids)
                and set(raw_ids) == set(by_id)):
            return [by_id[call_id] for call_id in raw_ids]
        # Providers may omit raw ordering; conservatively process invalid calls
        # first rather than let a valid action bypass a malformed proposal.
        return calls

    @staticmethod
    def _tool_call_history_message(message, tool_calls):
        """Keep rejected calls out of the next provider request's JSON parser.

        Providers can reject a whole conversation if a historical tool call has
        malformed arguments. Retain IDs for calls that will receive tool replies,
        using an empty object only as a history placeholder for rejected calls.
        Dispatch and logging must still use the original, unsanitized calls.
        Immediate mode excludes ignored invalid calls from its dispatch list, so
        those calls must also be absent here to avoid unanswered tool calls.
        """
        if not message.invalid_tool_calls:
            return message
        history = message.model_copy(deep=True)
        history.tool_calls = [
            {'name': call['name'], 'id': call['id'], 'type': 'tool_call',
             'args': ({} if call.get('invalid_arguments')
                      else deepcopy(call['args']))}
            for call in tool_calls]
        history.invalid_tool_calls = []
        # LangChain can fall back to this raw provider representation when no
        # parsed calls remain (in particular for immediate-mode invalid calls).
        history.additional_kwargs.pop('tool_calls', None)
        return history

    @staticmethod
    def _normalize_review_tool_arguments(action_tool, arguments):
        """Decode one serialized acknowledgment without repairing its contents.

        Some XML tool parsers serialize a JSON object as a string when its text
        contains apostrophes and JSON booleans/null. Only this reviewed-tool
        field may cross that transport boundary; all ordinary schema, evidence,
        suggestion and action-eligibility checks still run afterwards.
        """
        if ('review_response' not in action_tool.args_schema.model_fields
                or not isinstance(arguments, dict)):
            return arguments
        value = arguments.get('review_response')
        if (not isinstance(value, str)
                or len(value) > MAX_SERIALIZED_REVIEW_RESPONSE_CHARS):
            return arguments

        def unique_object(pairs):
            result = {}
            for name, item in pairs:
                if name in result:
                    raise ValueError('Duplicate JSON object field')
                result[name] = item
            return result

        def invalid_constant(value):
            raise ValueError('Non-JSON numeric constant')

        try:
            decoded = json.loads(value, object_pairs_hook=unique_object,
                                 parse_constant=invalid_constant)
        except (ValueError, RecursionError):
            return arguments
        if not isinstance(decoded, dict):
            return arguments
        log.info("Decoded JSON-encoded review_response from tool arguments; "
                 "normal validation still applies.")
        return {**arguments, 'review_response': decoded}

    @staticmethod
    def _action_argument_error(action_tool, arguments):
        """Give a repair template only for input failures before dispatch.

        Match LangChain's input validation, including its coercion/defaults.
        Examples are feedback for the next model turn, never executable repairs.
        Exceptions from the tool body still use the ordinary failure path.
        """
        schema = action_tool.args_schema
        try:
            schema.model_validate(arguments)
        except ValidationError as error:
            errors = error.errors(include_url=False, include_input=False)
        else:
            return None

        missing = []
        invalid = []
        received_arguments = deepcopy(arguments) if isinstance(arguments, dict) else arguments
        # Argument-repair feedback masks the final review response while
        # leaving the native tool arguments visible.
        if 'review_response' in schema.model_fields and isinstance(received_arguments, dict):
            if 'review_response' in received_arguments:
                received_arguments['review_response'] = '<supply a fresh structured review response>'
        retry_arguments = deepcopy(received_arguments) if isinstance(received_arguments, dict) else {}
        for issue in errors[:20] if 'review_response' in schema.model_fields else errors:
            if issue['loc'] and issue['loc'][0] == 'review_response':
                response_error = ('review_response: provide a valid acknowledgment, '
                                  'nonempty response, and evidence list')
                if response_error not in invalid:
                    invalid.append(response_error)
                retry_arguments['review_response'] = {
                    'acknowledged_suggestion': '<copy the displayed keep, switch, or none>',
                    'response_to_advice': '<explain agreement or disagreement using visible evidence>',
                    'evidence': [{'path': '<exact /review or /current JSON Pointer>',
                                  'value': '<copy the cited JSON value exactly>'}],
                }
                continue
            path = '.'.join(str(part) for part in issue['loc']) or 'arguments'
            if issue['type'] == 'missing':
                missing.append(path)
            else:
                invalid.append(f"{path}: {issue['msg']}")
            if issue['loc'] and issue['loc'][0] in schema.model_fields:
                field_name = issue['loc'][0]
                retry_arguments[field_name] = (
                    '<write your own brief reason for this action>'
                    if field_name == 'justification' else f'<provide {field_name}>')

        required = [name for name, field_info in schema.model_fields.items()
                    if field_info.is_required()]
        lines = [f"Tool call failed: argument validation for {action_tool.name}. "
                 "This tool was not executed."]
        if missing:
            lines.append('Missing required arguments: ' + ', '.join(missing))
        if invalid:
            lines.append('Invalid arguments: ' + '; '.join(invalid))
        lines.extend([
            'Required arguments: ' + ', '.join(required),
            'Received arguments: ' + json.dumps(received_arguments, ensure_ascii=False, default=str),
            'Retry example: ' + json.dumps(
                {'name': action_tool.name, 'arguments': retry_arguments},
                ensure_ascii=False, default=str),
        ])
        instruction = ('Replace every <...> marker with your own value before calling '
                       'the tool on a new turn. ')
        if any(issue['loc'] and issue['loc'][0] == 'justification' for issue in errors):
            instruction += ('Put justification inside the arguments; explanation '
                            'outside the call does not fill it. ')
        lines.append(instruction + 'Do not repeat the unchanged invalid call.')
        return '\n'.join(lines)

    def _handle_tool_calls(self, session, tools_by_name, tool_calls,
                           messages):
        """Invoke each of the agent's tool calls, appending a
        ToolMessage reply for every call.  Returns whether any call
        resulted in an environment action being taken."""
        acted = False
        action_attempted = False
        for tool_call in tool_calls:
            with agent_log_scope(
                    **session.log_context(), tool_call_id=tool_call['id']):
                agent_log_event(
                    log, 'TOOL', 'INPUT', f'name={tool_call["name"]}',
                    payload=lambda: {'name': tool_call['name'],
                                     'arguments': tool_call['args']})
                status = 'returned'

                # Every tool call needs a reply message, even after the
                # scenario completes mid-batch
                if session.scenario_complete:
                    status = "already_complete"
                    result = "Scenario is already complete."
                elif self._state.workflow.single_action_per_response and action_attempted:
                    status = "deferred"
                    result = ("Tool call deferred. Read the preceding action result "
                              "and choose your next tool on a new turn.")
                elif tool_call['name'] not in tools_by_name:
                    status = "unknown_tool"
                    result = (f"Unknown tool: {tool_call['name']}. "
                              "Available tools: "
                              f"{', '.join(tools_by_name)}")
                else:
                    n_actions_before = session.n_actions
                    if (self._state.workflow.single_action_per_response
                            and tool_call['name'] in session.action_types.values()):
                        # Claim the slot before validating arguments or resolving an action.
                        action_attempted = True
                    try:
                        if tool_call.get('invalid_arguments'):
                            raise ToolException("Malformed JSON tool arguments; "
                                                "use a valid JSON object.")
                        selected_tool = tools_by_name[tool_call['name']]
                        result = None
                        arguments = tool_call['args']
                        if selected_tool is self._state.action_tools.get(tool_call['name']):
                            arguments = self._normalize_review_tool_arguments(selected_tool, arguments)
                            result = self._action_argument_error(selected_tool, arguments)
                        if result is None:
                            result = selected_tool.invoke(arguments)
                        else:
                            status = 'invalid_arguments'
                    except (ValidationError, TypeError, ToolException) as e:
                        # Malformed arguments are fed back to the agent;
                        # environment errors propagate
                        agent_log_private(str(e))
                        result = f"Tool call failed: {e}"
                        status = 'failed'

                    else:
                        if (tool_call['name'] in self._state.extra_tool_names
                                and session.n_actions == n_actions_before):
                            self._state.context_provider_results[f"tool:{tool_call['name']}"] = str(result)
                    if session.n_actions > n_actions_before:
                        acted = True
                        if (tools_by_name[tool_call['name']]
                                is not self._state.action_tools.get(tool_call['name'])):
                            # Native wrappers already reset after execution. A toolkit
                            # can also execute through the supplied session directly.
                            self._state.context_provider_results.clear()
                            self._state.workflow.action_succeeded()
                            action_attempted = True

                agent_log_event(
                    log, 'TOOL', 'OUTPUT',
                    f'name={tool_call["name"]} status={status} {preview(str(result))}',
                    payload=str(result),
                    level=(logging.WARNING if status in {
                        'failed', 'invalid_arguments', 'unknown_tool'} else logging.INFO))
                messages.append(ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call['id']))

        return acted

    def _add_decision_context(self, session, messages):
        """Give every attached toolkit the chance to put information in
        front of the agent for the upcoming decision (see
        `AgentToolProvider.decision_context`); each returned text is
        appended as a human message (merged into the preceding one if
        the last message is already from the human side).  The opening
        prompt is never merged into: `pack_history` pins it at
        the front of every window, so anything merged there would stay
        in front of the agent (stale) for the whole scenario."""
        for name, provider in self.extra_tools.items():
            context_fn = getattr(provider, 'decision_context', None)
            context = context_fn(session) if context_fn else None
            if not context:
                continue

            agent_log_event(
                log, 'TOOL', 'CONTEXT',
                f'provider={name} {preview(str(context))}',
                payload=context, **session.log_context())
            self._state.context_provider_results[name] = str(context)

            if len(messages) > 1 and isinstance(messages[-1], HumanMessage):
                messages[-1] = HumanMessage(
                    content=f"{messages[-1].content}\n\n{context}")
            else:
                messages.append(HumanMessage(content=context))

    def _prepare_session(self, session):
        validate_reviewer_contract(self.reviewer, self.expected_response_contract)
        if session.align_to_target:
            self._state.alignment_context = build_alignment_context(
                session.alignment_target, self.attribute_definitions)

    def _run_agent_loop_impl(self, session, decision_model, tools):
        tools_by_name = {t.name: t for t in tools}
        llm_with_tools = decision_model.bind_tools(tools)
        decision_budget = self.context_budget.for_model(llm_with_tools, 'decision')

        system_prompt = "\n\n".join(
            [self.system_prompt,
             *(p.prompt_hint for p in self.extra_tools.values()
               if getattr(p, 'prompt_hint', None))])
        system_prompt += self._state.workflow.system_prompt_suffix
        system_prompt += '\n\n' + self.scene_end_guidance[session.scene_end_policy]
        system_prompt += '\n\n' + self.patient_history_guidance
        system_prompt += '\n' + self.citation_guidance
        if self._state.alignment_context is not None:
            system_prompt += '\n\n' + self.alignment_guidance
        system_message = SystemMessage(content=system_prompt)
        opening_prompt = self.opening_prompt
        if self._state.alignment_context is not None:
            # The opening message is pinned by both history-packing paths.
            opening_prompt += '\n\nAlignment context:\n' + json.dumps(
                {'alignment_context': self._state.alignment_context},
                ensure_ascii=False, allow_nan=False, separators=(',', ':'))
        messages = [HumanMessage(content=opening_prompt)]

        consecutive_llm_failures = 0
        # Decision point (action count) the toolkits last provided
        # context for
        context_n_actions = None
        recovery_n_actions = None
        overflow_recovery_used = False
        failed_input_tokens = None

        while (not session.scenario_complete
               and session.n_actions < session.max_actions_per_scenario):
            if session.n_actions != context_n_actions:
                context_n_actions = session.n_actions
                self._add_decision_context(session, messages)
            # Fetch offers only if no provider supplied them for this action
            # boundary, including a new scene.
            if session.actions_observed_at != session.n_actions:
                session.refresh_actions()
            if recovery_n_actions != session.n_actions:
                recovery_n_actions = session.n_actions
                overflow_recovery_used = False
                failed_input_tokens = None

            current_context = self._decision_review_context(session)
            self._state.workflow.prepare_decision_context(current_context)
            pending_feedback = self._state.workflow.pending_feedback()
            pending = [HumanMessage(content=pending_feedback) if pending_feedback is not None
                       else self._current_context_message(current_context)]
            invoke_kwargs = {}
            measurement = None
            if decision_budget is None:
                packed = pack_history(
                    messages, max_messages=self.max_messages_in_context,
                    pending_messages=pending)
                request_messages = [system_message, *packed.messages]
            else:
                request_messages, measurement = self._budget_decision_request(
                    session, decision_budget, system_message, messages, pending,
                    force_minimal=overflow_recovery_used)
                if (failed_input_tokens is not None
                        and measurement.input_tokens >= failed_input_tokens):
                    raise ContextBudgetError(
                        'Context overflow recovery cannot produce a smaller request; '
                        'no environment action was submitted.',
                        diagnostics=measurement.diagnostics)
                failed_input_tokens = None
                invoke_kwargs = measurement.invoke_kwargs
            with agent_log_call(
                    action_index=session.n_actions,
                    scene_id=session.current_state.meta_info.scene_id):
                self._state.workflow.begin_decision_attempt()
                acted = False

                phase = ('final' if pending_feedback is not None else
                         'proposal' if self.reviewer is not None else 'immediate')
                agent_log_event(
                    log, 'DECISION', 'INPUT',
                    f'messages={len(request_messages)} tools={len(tools)} '
                    f'phase={phase}',
                    payload=lambda: _decision_log_payload(request_messages, llm_with_tools, tools))
                try:
                    ai_message = llm_with_tools.invoke(request_messages, **invoke_kwargs)
                    consecutive_llm_failures = 0
                except Exception as e:
                    if decision_budget is not None and is_context_overflow(e):
                        # Count the failed generation, but never let an oversized
                        # request fall through to automatic environment fallback.
                        self._state.workflow.finish_decision_attempt(False)
                        remaining = (self.max_llm_calls_between_actions
                                     - self._state.workflow.calls_since_action)
                        diagnostics = {
                            **measurement.diagnostics, 'role': 'decision',
                            'action_index': session.n_actions,
                            'recovery': ('failed' if overflow_recovery_used or remaining < 1
                                         else 'retry_with_minimal_history'),
                        }
                        agent_log_event(
                            log, 'DECISION', 'CONTEXT REJECTED',
                            f'recovery={diagnostics["recovery"]}',
                            payload=diagnostics, level=logging.WARNING,
                            legacy_message='Context budget: %s',
                            legacy_args=(json.dumps(diagnostics),))
                        if overflow_recovery_used or remaining < 1:
                            raise ContextBudgetError(
                                'Decision model rejected the context length; bounded '
                                'recovery is exhausted. No fallback action was submitted.',
                                diagnostics=diagnostics) from e
                        overflow_recovery_used = True
                        failed_input_tokens = measurement.input_tokens
                        continue
                    agent_log_event(
                        log, 'DECISION', 'UNAVAILABLE',
                        f'error_type={type(e).__name__}', level=logging.ERROR)
                    consecutive_llm_failures += 1
                    if consecutive_llm_failures >= 3:
                        # The model is unreachable/broken, not merely
                        # indecisive; abort rather than blindly driving
                        # the scenario with fallback actions
                        raise RuntimeError(
                            "Agent LLM unreachable/failing "
                            f"({consecutive_llm_failures} consecutive "
                            f"failures; last error type: {type(e).__name__}).  Check that the "
                            "model backend is running and reachable "
                            "(e.g. `ollama serve` for ollama models, "
                            "`vllm serve ...` for the vllm_* configs).") from e
                    ai_message = None

                if ai_message is not None:
                    agent_log_event(
                        log, 'DECISION', 'OUTPUT',
                        f'tool_calls={len(ai_message.tool_calls)} '
                        f'invalid_tool_calls={len(ai_message.invalid_tool_calls)} '
                        f'preview={preview(visible_text(ai_message))}',
                        payload=lambda: {'messages': message_payload([ai_message])})

                    if (not ai_message.tool_calls
                            and not (self._state.workflow.single_action_per_response
                                     and ai_message.invalid_tool_calls)):
                        recovered = parse_text_tool_calls(ai_message.content)
                        if recovered:
                            agent_log_event(
                                log, 'DECISION', 'RECOVERED TOOL CALLS',
                                f'count={len(recovered)} source=visible_response')
                            # Rebuild the turn as a structured tool-calling
                            # message so its replies are ordinary
                            # ToolMessages and the model sees the canonical
                            # shape in its own history
                            ai_message = AIMessage(
                                content=ai_message.content,
                                tool_calls=recovered,
                                additional_kwargs=ai_message.additional_kwargs)

                    tool_calls = (self._ordered_tool_calls(ai_message)
                                  if self._state.workflow.single_action_per_response
                                  else ai_message.tool_calls)
                    messages.append(self._tool_call_history_message(ai_message, tool_calls))

                    if not tool_calls:
                        agent_log_event(
                            log, 'WORKFLOW', 'CORRECTION',
                            'reason=no_tool_calls; no action executed',
                            level=logging.WARNING)

                        # Some models narrate instead of calling tools;
                        # put the concrete options in front of them
                        choices_block = session.format_available_actions(
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
                            session, tools_by_name, tool_calls, messages)

                self._state.workflow.finish_decision_attempt(acted)
                llm_calls_since_action = self._state.workflow.calls_since_action

                if (not session.scenario_complete
                        and llm_calls_since_action
                        >= self.max_llm_calls_between_actions):
                    # The agent is spinning without acting; take the first
                    # available action so a live session can't stall
                    agent_log_event(
                        log, 'WORKFLOW', 'BUDGET EXHAUSTED',
                        f'calls_since_action={llm_calls_since_action}; using fallback',
                        level=logging.WARNING, **session.log_context())
                    fallback_action = self._take_fallback_action(session)

                    messages.append(HumanMessage(content=(
                        "You were not making progress, so the following "
                        "action was taken on your behalf: "
                        f"{fallback_action.unstructured}.  Re-observe the "
                        "environment and continue.")))

                if not session.scenario_complete:
                    session.refresh_actions()
                    if session.scene_done:
                        # Don't wait for the agent to notice it's done (it
                        # tends to keep re-checking vitals indefinitely)
                        agent_log_event(
                            log, 'WORKFLOW', 'SCENE END',
                            'reason=all_patients_handled', **session.log_context())
                        self._force_end_scene(
                            session,
                            "All patients have been tagged and treated")
