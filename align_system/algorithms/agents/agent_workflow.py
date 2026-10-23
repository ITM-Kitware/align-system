"""Per-scenario execution policies for the shared LangChain agent loop.

Policies gate resolved actions and account for model attempts. They never receive
executable tools or submit environment actions; the session owns execution.
"""

import json
from copy import deepcopy
from dataclasses import dataclass, field
from decimal import Decimal, localcontext
from types import MappingProxyType

from pydantic import BaseModel, ValidationError

from align_system.algorithms.agents.agent_review import (
    ActionReviewer, ReviewAdvice, WeightedReviewAdvice, review_diagnostics,
    validate_reviewer_contract, REVIEW_RESPONSE_CONTRACT, review_action_tool_models,
)
from align_system.algorithms.agents.agent_evidence import (
    EvidenceValidationError, FINAL_EVIDENCE_ROOTS, REVIEW_EVIDENCE_ROOTS, FinalReviewResponse,
    build_citation_guide, validate_evidence, validate_review_assessment,
)
from align_system.prompt_engineering.reviewer_prompts import ADVISORY_REVIEW_GUIDANCE
from align_system.utils import logging
from align_system.utils.agent_logging import agent_log_call, agent_log_event, preview

log = logging.getLogger(__name__)


@dataclass
class _ReviewCycle:
    """Private transaction state for one proposal, review, and final selection.

    The workflow retains this state across model turns until an action succeeds;
    frozen contexts are needed to check citations against exactly what was shown.
    It is not an interchangeable context object for other components.
    """

    # Resolved original action and the native-tool representation shown for review.
    proposal: dict | None = None
    proposal_tool: dict | None = None
    proposal_validation: dict | None = None
    # Computed advisory strength/threshold and alternative validation; never authority.
    weighting: dict | None = None
    # Review outcome, accepted advice, and bounded failures/attempt history.
    status: str | None = None
    advice: dict | None = None
    error: str | None = None
    review_diagnostics: dict | None = None
    review_attempts: list[dict] = field(default_factory=list)
    # Evidence checks and frozen contexts for the reviewer and current decision turn.
    review_assessment_validation: dict | None = None
    review_context: dict | None = None
    current_context: dict | None = None
    # Decision agent acknowledgment plus its verification context and original rationale.
    final_review_response: dict | None = None
    final_review_response_validation: dict | None = None
    final_review_context: dict | None = None
    original_justification: str | None = None
    # Model invocation attempts since the last successful environment action.
    decision_calls: int = 0
    reviewer_calls: int = 0

    @property
    def total_calls(self):
        return self.decision_calls + self.reviewer_calls


class ImmediateActionWorkflow:
    """Execute valid actions immediately, retaining legacy per-response attempt counts."""

    single_action_per_response = False
    system_prompt_suffix = ''
    needs_proposal_context = False
    cycle = None

    def __init__(self):
        self.calls_since_action = 0

    def configure_action_schemas(self, schemas):
        """Only reviewed workflows need the read-only native schema catalog."""

    def begin_decision_attempt(self):
        """Legacy accounting completes after all tools in a response run."""

    def prepare_decision_context(self, context):
        """Immediate mode does not retain reviewer evidence snapshots."""

    def finish_decision_attempt(self, acted):
        self.calls_since_action = 0 if acted else self.calls_since_action + 1

    def before_action(self, snapshot, action, justification, *, context,
                      review_response=None):
        """None permits execution; text requests a new decision-agent turn."""
        return None

    def pending_feedback(self):
        return None

    def execution_metadata(self, action, justification, driver_override):
        return None

    def action_succeeded(self, driver_override=None):
        # Legacy no-progress accounting spans automatic scene transitions.
        # Model-selected actions and explicit budget fallback reset it.
        if driver_override not in {'scene_complete', 'action_cap'}:
            self.calls_since_action = 0


class AdvisoryReviewWorkflow(ImmediateActionWorkflow):
    """Stage one proposal and review opportunity, then let the decision agent choose.

    A proposal consumes the review opportunity even when advice is unavailable.
    A malformed built-in review can be repaired once within the shared model invocation limit.
    The shared limit counts decision/reviewer invocation attempts, not tokens.
    Context-token limits are checked separately by review_request_budget.
    Reset only after successful execution, including successful driver overrides.
    """

    single_action_per_response = True
    system_prompt_suffix = ADVISORY_REVIEW_GUIDANCE

    def __init__(self, reviewer, default_chat_model, max_llm_calls_between_actions=8,
                 *, actions, review_request_budget=None,
                 expected_response_contract=REVIEW_RESPONSE_CONTRACT,
                 system_prompt_suffix=None):
        if max_llm_calls_between_actions < 1:
            raise ValueError("Reviewer mode requires a positive model-call budget")
        validate_reviewer_contract(reviewer, expected_response_contract)
        self.expected_response_contract = expected_response_contract
        self.actions = actions
        self.system_prompt_suffix = (ADVISORY_REVIEW_GUIDANCE if system_prompt_suffix is None
                                     else system_prompt_suffix)
        self.reviewer = reviewer
        self.default_chat_model = default_chat_model
        self.max_llm_calls_between_actions = max_llm_calls_between_actions
        self.review_request_budget = review_request_budget
        self.max_diagnostic_errors = getattr(reviewer, 'max_diagnostic_errors', 20)
        self.max_diagnostic_chars = getattr(reviewer, 'max_diagnostic_chars', 8192)
        self.cycle = _ReviewCycle()
        self.action_schemas = MappingProxyType({})
        self._review_action_models = {}
        self._action_json_schemas = {}
        self._offered_schema_catalogs = {}

    @property
    def needs_proposal_context(self):
        return self.cycle.proposal is None

    @property
    def calls_since_action(self):
        return self.cycle.total_calls

    def configure_action_schemas(self, schemas):
        self.action_schemas = MappingProxyType(dict(schemas))
        self._review_action_models = review_action_tool_models(
            schemas, injury_locations=self.actions.injury_locations,
            location_guidance=getattr(self.reviewer, 'location_guidance', None))
        self._action_json_schemas = {
            name: model.model_json_schema()
            for name, model in self._review_action_models.items()}
        self._offered_schema_catalogs.clear()

    def begin_decision_attempt(self):
        self.cycle.decision_calls += 1

    def prepare_decision_context(self, context):
        # This is the current evidence shown for one model invocation. A tool
        # refresh later in that invocation must not change citation meanings.
        self.cycle.current_context = deepcopy(context)
        if 'citation_guide' not in self.cycle.current_context:
            self.cycle.current_context['citation_guide'] = build_citation_guide(context)
        self._clear_final_response()

    def finish_decision_attempt(self, acted):
        # All attempts were counted before invocation, including failures.
        pass

    def pending_feedback(self):
        if self.cycle.proposal is not None:
            return self._review_reminder(self.cycle)
        return None

    def action_succeeded(self, driver_override=None):
        self.cycle = _ReviewCycle()

    def _empty_weighting(self, status):
        config = getattr(self.reviewer, 'weighting', None)
        if config is None:
            return None
        return {
            'configuration': config.model_dump(mode='json'),
            'confidence': None, 'estimated_improvement': None,
            'alternative': None, 'alternative_validation': None,
            'alternative_tool': None,
            'status': status, 'strength': None, 'suggestion': None,
            'error': None,
        }

    def _weighted_guidance(self, snapshot, advice):
        """Validate a comparison without dispatching tools or taking actions."""
        guidance = self._empty_weighting('available')
        guidance.update(confidence=advice.confidence,
                        estimated_improvement=advice.estimated_improvement)
        if advice.recommendation == 'keep':
            guidance.update(strength=0.0, suggestion='keep')
            return guidance
        if advice.alternative is None:
            guidance['status'] = 'no_alternative'
            return guidance

        alternative = advice.alternative
        schema = self.action_schemas.get(alternative.tool_name)
        if schema is None:
            guidance.update(status='invalid_alternative',
                            error='Alternative must name an available action tool.')
            return guidance
        try:
            unknown = set(alternative.arguments) - set(schema.model_fields)
            if unknown:
                raise ValueError('Unknown action arguments: ' + ', '.join(sorted(unknown)))
            arguments = schema.model_validate(
                alternative.arguments, strict=True).model_dump()
        except (ValidationError, ValueError) as error:
            guidance.update(status='invalid_alternative', error=str(error))
            return guidance
        action_type = self.actions.action_tool_names[alternative.tool_name]
        action, error = self.actions.resolve_typed_action(snapshot, action_type, **arguments)
        if error is not None:
            guidance.update(status='invalid_alternative', error=error)
            return guidance
        view = self.actions.review_action_view(action)
        if self.actions.same_action(self.cycle.proposal, view):
            guidance.update(status='not_distinct',
                            error='Alternative resolves to the original proposal.')
            return guidance

        config = self.reviewer.weighting
        # Compare the decimal ratings exactly so a threshold tie cannot switch
        # because binary floating-point rounded the product slightly upward.
        with localcontext() as context:
            # Three finite float spellings have at most 51 significant digits.
            context.prec = 64
            strength = (Decimal(str(config.influence))
                        * Decimal(str(advice.confidence))
                        * Decimal(str(advice.estimated_improvement)))
        guidance.update(
            alternative=view,
            alternative_tool=self.actions.proposal_tool_view(action),
            alternative_validation=self.actions.parameter_validation(action, snapshot=snapshot),
            strength=float(strength),
            suggestion=('switch' if strength > Decimal(str(config.swap_threshold))
                        else 'keep'))
        return guidance

    @staticmethod
    def _evidence_context_view(context):
        """Display only citeable roots, preserving their original JSON paths.

        The complete snapshot remains private workflow state and is recorded in
        metadata. In particular, native schemas and available-action catalogs
        need not be repeated in every final-choice reminder.
        """
        if context is None:
            return None
        view = {}
        for path in REVIEW_EVIDENCE_ROOTS:
            parts = path.lstrip('/').split('/')
            value = context
            for part in parts:
                if not isinstance(value, dict) or part not in value:
                    break
                value = value[part]
            else:
                target = view
                for part in parts[:-1]:
                    target = target.setdefault(part, {})
                target[parts[-1]] = deepcopy(value)
        if 'proposal_tool' in view:
            view['proposal_tool'] = deepcopy(context['proposal_tool'])
        return view

    @staticmethod
    def _review_reminder(cycle):
        weighting = cycle.weighting
        suggestion = ((weighting or {}).get('suggestion') or 'none')
        guidance = f"Computed suggestion: {suggestion}. "
        if weighting is not None:
            configuration = weighting['configuration']
            if (weighting['confidence'] is not None
                    and weighting['estimated_improvement'] is not None
                    and weighting['strength'] is not None):
                guidance += (
                    f"{configuration['influence']} * {weighting['confidence']} * "
                    f"{weighting['estimated_improvement']} = {weighting['strength']}; "
                    f"{weighting['strength']} {'>' if suggestion == 'switch' else '<='} "
                    f"swap_threshold {configuration['swap_threshold']}. ")
            elif weighting['strength'] is not None:
                guidance += (
                    f"Strength = {weighting['strength']} because the reviewer recommended "
                    "keeping the proposal; no alternative ratings were supplied. ")
            guidance += (
                "Switch is suggested only above swap_threshold; a tie suggests keep. "
                f"Guidance status: {weighting['status']}. ")
        failure_hint = (
            cycle.review_diagnostics.get('summary', 'Reviewer advice is unavailable.') + '\n'
            if cycle.review_diagnostics is not None else '')
        assessment_status = (cycle.review_assessment_validation or {}).get('status')
        acknowledgment = (
            "For the final action, review_response is required: "
            f"acknowledged_suggestion must be '{suggestion}', response_to_advice "
            "must explain your choice in light of the advice, and evidence must "
            "contain at least one {path, value} reference matching the supplied "
            "/review or /current context exactly. Cite observation, "
            "proposal_tool/arguments, proposal_validation, handled_patients, or "
            "provider_context, patient_history, or alignment_context within those namespaces. "
            "You may disagree with "
            "the suggestion; acknowledge it accurately and explain your choice.\n"
            if cycle.status == 'reviewed' else
            "No valid review is available, so review_response is not required.\n")
        return (
            guidance + '\n'
            "The proposed action has NOT executed. Reviewer advice is advisory. "
            "Your next valid action-tool call selects and executes the final "
            "action after its review_response is validated; keep or revise the "
            "proposal using the available tools. Do not wait for another review.\n"
            + failure_hint + acknowledgment
            + 'Reviewer concern (advisory; separate from computed guidance):\n'
            + json.dumps({
                'advice': ({key: cycle.advice[key] for key in
                            ('recommendation', 'reason', 'assessment')}
                           if cycle.advice is not None else None),
                'review_assessment_validation': {'status': assessment_status},
                'review_status': cycle.status, 'review_error': cycle.error,
                **({'proposal_tool': cycle.proposal_tool} if cycle.review_context is None else {}),
                'weighting': ({key: value for key, value in weighting.items()
                               if key != 'alternative'} if weighting is not None else None),
            }, ensure_ascii=False, separators=(',', ':'))
            + '\nCitation guides (path lookup only; not evidence):\n'
            + json.dumps({
                'review': AdvisoryReviewWorkflow._prefixed_guide(cycle.review_context, '/review'),
                'current': AdvisoryReviewWorkflow._prefixed_guide(cycle.current_context, '/current')},
                ensure_ascii=False, separators=(',', ':'))
            + '\nEvidence namespaces frozen for this decision-model invocation:\n'
            + json.dumps({
                'review': AdvisoryReviewWorkflow._evidence_context_view(cycle.review_context),
                'current': AdvisoryReviewWorkflow._evidence_context_view(cycle.current_context)},
                ensure_ascii=False, separators=(',', ':')))

    @staticmethod
    def _prefixed_guide(context, prefix):
        guide = (context or {}).get('citation_guide') or {}
        return {group: [{key: prefix + value if key.endswith('_path') else value
                        for key, value in entry.items()} for entry in entries]
                for group, entries in guide.items()}

    @staticmethod
    def _proposal_summary(proposal_tool, validation):
        arguments = ', '.join(
            f'{name}={json.dumps(value, ensure_ascii=False, sort_keys=True)}'
            for name, value in sorted(proposal_tool['arguments'].items()))
        summary = f"Completed {proposal_tool['tool_name']} arguments: {arguments}."
        inventory = validation.get('inventory')
        if inventory is not None:
            summary += f' Inventory status: {inventory}.'
            if inventory == 'unverified':
                summary += ' The named supply is present in the proposal; inventory is unknown.'
        return summary

    def _clear_final_response(self):
        self.cycle.final_review_response = None
        self.cycle.final_review_response_validation = None
        self.cycle.final_review_context = None
        self.cycle.original_justification = None

    @staticmethod
    def _validation_result(status, diagnostics=None):
        diagnostics = diagnostics or {}
        return {
            'status': status,
            'validation_errors': deepcopy(diagnostics.get('validation_errors', [])),
            'validation_errors_truncated': diagnostics.get('validation_errors_truncated', False),
        }

    def _validate_final_response(self, action, justification, review_response):
        self._clear_final_response()
        cycle = self.cycle
        if cycle.status != 'reviewed':
            cycle.original_justification = justification
            return None
        suggestion = ((cycle.weighting or {}).get('suggestion') or 'none')
        stage = 'schema_validation'
        try:
            if isinstance(review_response, BaseModel):
                review_response = review_response.model_dump(mode='python')
            response = FinalReviewResponse.model_validate(review_response, strict=True)
            if response.acknowledged_suggestion != suggestion:
                cycle.final_review_response_validation = self._validation_result('invalid', {
                    'validation_errors': [{
                        'loc': ['acknowledged_suggestion'], 'type': 'suggestion_mismatch',
                        'msg': f"Acknowledge the computed suggestion '{suggestion}'."}],
                })
            else:
                stage = 'evidence_validation'
                evidence_context = {
                    'review': cycle.review_context, 'current': cycle.current_context}
                validate_evidence(response.evidence, evidence_context, FINAL_EVIDENCE_ROOTS)
                cycle.final_review_response = response.model_dump(mode='json')
                cycle.final_review_response_validation = self._validation_result('verified')
                cycle.final_review_context = deepcopy(evidence_context)
                cycle.original_justification = justification
                # Always compose from this attempt's original tool argument.
                # Never append to an earlier action's composed justification.
                action.justification = (
                    f'{justification}\n\nResponse to reviewer: {response.response_to_advice}')
                return None
        except (ValidationError, EvidenceValidationError) as error:
            diagnostics = review_diagnostics(
                error, expected_schema='FinalReviewResponse', stage=stage,
                max_diagnostic_errors=self.max_diagnostic_errors,
                max_diagnostic_chars=self.max_diagnostic_chars)
            cycle.final_review_response_validation = self._validation_result('invalid', diagnostics)
        agent_log_event(
            log, 'WORKFLOW/ENVIRONMENT', 'FINAL CORRECTION',
            'Final review response rejected; a fresh action-tool call is required. '
            'The action has not executed and no new review will occur.',
            payload=lambda: cycle.final_review_response_validation)
        return (
            "The final review_response could not be verified. This action has NOT executed. "
            f"Supply review_response with acknowledged_suggestion='{suggestion}', a nonempty "
            "response_to_advice, and at least one exact {path, value} evidence reference "
            "from the supplied /review or /current context. You may still choose any "
            "eligible action and disagree with the advice. Correct the acknowledgment "
            "on a new action-tool call; no additional review will occur. Use only the "
            "documented fields, and check the cited paths and values against that context.")

    def _repair_feedback(self, diagnostics):
        """Bounded feedback goes only to the reviewer, never to decision history."""
        return {
            'failed_attempt': 1,
            'stage': diagnostics['stage'],
            'summary': diagnostics['summary'],
            'validation_errors': deepcopy(diagnostics.get('validation_errors', [])[:self.max_diagnostic_errors]),
            'rejected_visible_output': (diagnostics.get('response_excerpt') or '')[:self.max_diagnostic_chars],
        }

    def _review_proposal(self, snapshot, review_context):
        """One review opportunity with at most one built-in response repair."""
        cycle = self.cycle
        builtin = (isinstance(self.reviewer, ActionReviewer)
                   and getattr(self.reviewer.review, '__func__', None) is ActionReviewer.review)
        max_repairs = self.reviewer.max_repair_attempts if builtin else 0
        schema = ReviewAdvice if cycle.weighting is None else WeightedReviewAdvice
        repair_feedback = None
        for attempt in range(1 + max_repairs):
            with agent_log_call(review_attempt=attempt + 1, tool_call_id=None) as call_id:
                # Preflight/configuration failures consume an attempt just as before.
                cycle.reviewer_calls += 1
                generation = {}
                failure_stage = 'invocation'
                advice = None
                try:
                    review_kwargs = {}
                    if builtin:
                        review_kwargs['attempt_metadata'] = generation
                        offered = review_context.get('action_tool_schemas', {})
                        review_kwargs['action_models'] = {
                            name: model for name, model in self._review_action_models.items()
                            if name in offered}
                        if self.review_request_budget is not None:
                            review_kwargs['request_budget'] = self.review_request_budget
                    request_context = deepcopy(review_context)
                    if repair_feedback is not None:
                        request_context['repair_feedback'] = deepcopy(repair_feedback)
                        agent_log_event(
                            log, 'WORKFLOW/ENVIRONMENT', 'REVIEW REPAIR',
                            f'Repairing the same proposal after {repair_feedback["stage"]}.',
                            payload=repair_feedback)
                    if not builtin:
                        agent_log_event(
                            log, 'REVIEWER', 'ADAPTER INPUT',
                            f'Custom reviewer adapter boundary | context_fields={len(request_context)}',
                            payload=lambda: {'context': request_context})
                    advice = self.reviewer.review(
                        request_context, self.default_chat_model, **review_kwargs)
                    failure_stage = 'schema_validation'
                    if isinstance(advice, BaseModel):
                        # Revalidate injected/unchecked model instances, too.
                        advice = advice.model_dump(mode='python')
                    if not builtin:
                        agent_log_event(
                            log, 'REVIEWER', 'ADAPTER OUTPUT',
                            'Custom reviewer adapter boundary | validating returned advice',
                            payload=lambda: {'advice': advice})
                    advice = schema.model_validate(advice, strict=True)
                    failure_stage = 'evidence_validation'
                    validate_review_assessment(advice.assessment, cycle.review_context)
                    cycle.review_assessment_validation = self._validation_result('verified')
                    cycle.advice = advice.model_dump(mode='json')
                    cycle.status = 'reviewed'
                    cycle.error = None
                    cycle.review_diagnostics = None
                    if cycle.weighting is not None:
                        cycle.weighting = self._weighted_guidance(snapshot, advice)
                except Exception as error:
                    cycle.status = 'unavailable'
                    cycle.error = type(error).__name__
                    rejected_advice = (
                        json.dumps(advice.model_dump(mode='json'), ensure_ascii=False)
                        if failure_stage == 'evidence_validation'
                        and isinstance(error, EvidenceValidationError) else None)
                    cycle.advice = None
                    cycle.review_diagnostics = review_diagnostics(
                        error, expected_schema=schema.__name__,
                        response_format=getattr(self.reviewer, 'response_format', 'text'),
                        stage=failure_stage, response_text=rejected_advice,
                        max_diagnostic_errors=self.max_diagnostic_errors,
                        max_diagnostic_chars=self.max_diagnostic_chars)
                    cycle.review_assessment_validation = self._validation_result(
                        'invalid' if isinstance(error, EvidenceValidationError) else 'unavailable',
                        cycle.review_diagnostics)
                    if cycle.weighting is not None:
                        cycle.weighting['error'] = cycle.review_diagnostics.get(
                            'summary', 'Reviewer advice is unavailable.')
                    agent_log_event(
                        log, 'WORKFLOW/ENVIRONMENT', 'REVIEW UNAVAILABLE',
                        f'stage={cycle.review_diagnostics["stage"]} | '
                        f'error_type={cycle.error} | {cycle.review_diagnostics["summary"]}',
                        payload=lambda: cycle.review_diagnostics)
                entry = {
                    'attempt': attempt + 1, 'status': cycle.status,
                    'diagnostics': deepcopy(cycle.review_diagnostics),
                    'generation': deepcopy(generation),
                }
                cycle.review_attempts.append(entry)
                if cycle.status == 'reviewed' or attempt == max_repairs:
                    break
                if cycle.review_diagnostics['stage'] not in {
                        'response_shape', 'json_parsing', 'schema_validation', 'evidence_validation'}:
                    entry['retry_skipped_reason'] = 'non_retryable'
                    agent_log_event(
                        log, 'WORKFLOW/ENVIRONMENT', 'REPAIR SKIPPED',
                        'reason=non_retryable | Continue to the final decision.')
                    break
                if self.max_llm_calls_between_actions - cycle.total_calls < 2:
                    entry['retry_skipped_reason'] = 'budget'
                    agent_log_event(
                        log, 'WORKFLOW/ENVIRONMENT', 'REPAIR SKIPPED',
                        'reason=budget | Reserve one model invocation for the final decision.')
                    break
                repair_feedback = self._repair_feedback(cycle.review_diagnostics)
        return call_id

    def before_action(self, snapshot, action, justification, *, context,
                      review_response=None):
        """Stage one completed proposal; the next decision-agent turn is final."""
        cycle = self.cycle
        if cycle.proposal is not None:
            correction = self._validate_final_response(action, justification, review_response)
            if correction is None:
                final = self.actions.review_action_view(action, justification)
                agent_log_event(
                    log, 'DECISION', 'FINAL SELECTION',
                    f'tool={self.actions.proposal_tool_view(action)["tool_name"]} | '
                    f'choice={"kept" if self.actions.same_action(cycle.proposal, final) else "revised"}'
                    f' | review_status={cycle.status} | awaiting execution',
                    payload=lambda: {'action': final,
                                     'review_response': cycle.final_review_response})
            return correction

        # Consume the review opportunity even on errors or budget exhaustion.
        cycle.proposal = self.actions.review_action_view(action, justification)
        cycle.proposal_tool = self.actions.proposal_tool_view(action)
        cycle.proposal_validation = self.actions.parameter_validation(action, snapshot=snapshot)
        agent_log_event(
            log, 'DECISION', 'PROPOSAL',
            f'tool={cycle.proposal_tool["tool_name"]} | awaiting review | '
            'The proposed action has not executed.',
            payload=lambda: {'proposal_tool': cycle.proposal_tool,
                             'proposal_validation': cycle.proposal_validation})
        cycle.weighting = self._empty_weighting('unavailable')
        cycle.review_assessment_validation = self._validation_result('unavailable')
        remaining = self.max_llm_calls_between_actions - cycle.total_calls
        review_call_id = None
        if remaining < 2:
            cycle.status = 'skipped_budget'
            cycle.error = 'budget'
            if cycle.weighting is not None:
                cycle.weighting['status'] = 'skipped_budget'
            agent_log_event(
                log, 'WORKFLOW/ENVIRONMENT', 'REVIEW SKIPPED',
                f'reason=budget | remaining_calls={remaining} | '
                'Reserve one model invocation for the final decision.', call_id=None)
        else:
            # Only explicitly sanitized views enter the reviewer request. The
            # raw resolver snapshot is private to local action validation.
            review_context = deepcopy(context)
            review_context.update({
                'handled_patients': {
                    'treated': sorted(snapshot.treated_patients),
                    'evacuated': sorted(snapshot.evac_patients),
                    'tagged': sorted(snapshot.tagged_patients)},
                'proposal_tool': deepcopy(cycle.proposal_tool),
                'proposal_validation': deepcopy(cycle.proposal_validation),
                'available_actions': self.actions.format_available_actions(snapshot.actions_filtered),
            })
            if cycle.weighting is not None:
                offered_names = frozenset(self.actions.action_types.get(a.action_type)
                                          for a in snapshot.actions_filtered)
                if offered_names not in self._offered_schema_catalogs:
                    self._offered_schema_catalogs[offered_names] = {
                        name: schema for name, schema in self._action_json_schemas.items()
                        if name in offered_names}
                review_context['action_tool_schemas'] = deepcopy(
                    self._offered_schema_catalogs[offered_names])
            if 'citation_guide' not in review_context:
                review_context['citation_guide'] = build_citation_guide(review_context)
            cycle.review_context = review_context
            review_call_id = self._review_proposal(snapshot, review_context)
        diagnostic_record = {
            'scenario_id': context['scenario_id'], 'action_index': context['action_index'],
            'proposal': cycle.proposal, 'status': cycle.status,
            'proposal_tool': cycle.proposal_tool,
            'advice': cycle.advice, 'error': cycle.error,
            'review_diagnostics': cycle.review_diagnostics,
            'review_attempts': cycle.review_attempts,
            'weighting': cycle.weighting}
        advice = cycle.advice or {}
        agent_log_event(
            log, 'REVIEWER', 'RESULT',
            f'recommendation={advice.get("recommendation", "none")} | '
            f'computed_suggestion={(cycle.weighting or {}).get("suggestion") or "none"} | '
            f'status={cycle.status}\nReason preview: {preview(advice.get("reason", ""))}',
            payload=diagnostic_record,
            legacy_message='Action review: %s',
            legacy_args=(json.dumps(diagnostic_record),),
            call_id=review_call_id,
            review_attempt=cycle.reviewer_calls or None, tool_call_id=None)
        # This receipt persists in tool history; the full active reminder is
        # supplied once, ephemerally, by pending_feedback() on the next turn.
        return (
            "The proposed action has NOT executed. "
            f"Review status: {cycle.status}. Read the current advisory reminder "
            "before making a fresh final action-tool call; no further review will occur.")

    def execution_metadata(self, action, justification, driver_override):
        cycle = self.cycle
        final = self.actions.review_action_view(action, justification)
        disposition = None
        if not driver_override and cycle.proposal is not None:
            unchanged = self.actions.same_action(cycle.proposal, final)
            disposition = 'kept' if unchanged else 'revised'
        metadata = {
            'proposal': deepcopy(cycle.proposal),
            'proposal_tool': deepcopy(cycle.proposal_tool),
            'proposal_validation': deepcopy(cycle.proposal_validation),
            'proposal_summary': (self._proposal_summary(
                cycle.proposal_tool, cycle.proposal_validation) if cycle.proposal_tool else None),
            'review_response_contract': self.expected_response_contract,
            'review_status': 'bypassed' if driver_override else cycle.status,
            'advice': deepcopy(cycle.advice), 'review_error': cycle.error,
            'review_diagnostics': deepcopy(cycle.review_diagnostics),
            'review_attempts': deepcopy(cycle.review_attempts),
            'review_assessment_validation': deepcopy(cycle.review_assessment_validation),
            'review_context': deepcopy(cycle.review_context),
            'final_review_response': (
                None if driver_override else deepcopy(cycle.final_review_response)),
            'final_review_response_validation': (
                None if driver_override else deepcopy(cycle.final_review_response_validation)),
            'final_review_context': (
                None if driver_override else deepcopy(cycle.final_review_context)),
            'original_justification': None if driver_override else cycle.original_justification,
            'final_action': final, 'disposition': disposition,
            'calls': {'decision_agent': cycle.decision_calls,
                      'reviewer': cycle.reviewer_calls},
            'driver_override': driver_override,
        }
        weighting = deepcopy(cycle.weighting)
        if weighting is None:
            weighting = self._empty_weighting('bypassed' if driver_override
                                             else 'unavailable')
        if weighting is not None:
            if driver_override:
                weighting['status'] = 'bypassed'
            actual_choice = None
            if cycle.proposal is not None:
                actual_choice = 'other'
                if self.actions.same_action(cycle.proposal, final):
                    actual_choice = 'proposal'
                elif (weighting['alternative'] is not None
                      and self.actions.same_action(weighting['alternative'], final)):
                    actual_choice = 'alternative'
            followed = None
            if not driver_override and weighting['suggestion'] is not None:
                expected = ('alternative' if weighting['suggestion'] == 'switch'
                            else 'proposal')
                followed = actual_choice == expected
            weighting.update(actual_choice=actual_choice, followed_guidance=followed)
            metadata['weighting'] = weighting
        return metadata
