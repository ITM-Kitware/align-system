"""A stateless, tool-free reviewer for proposed open-world actions."""

from collections.abc import Mapping
from copy import deepcopy
import json
from typing import Literal, Union

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import (
    BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, ValidationError,
    create_model, field_validator, model_validator)
from pydantic.json_schema import GenerateJsonSchema

from align_system.algorithms.agents.agent_evidence import (
    EvidenceValidationError, ReviewAssessment,
)
from align_system.prompt_engineering.agentic_prompts import (
    ALIGNMENT_GUIDANCE, PATIENT_HISTORY_GUIDANCE, SCENE_END_GUIDANCE,
)
from align_system.prompt_engineering.reviewer_prompts import (
    ALIGNMENT_REVIEW_GUIDANCE, COMPLETED_ACTION_REVIEW_GUIDANCE,
    QUALITATIVE_REVIEW_OUTPUT_CONTRACT,
    REVIEW_REPAIR_GUIDANCE, REVIEW_LOCATION_GUIDANCE, REVIEWER_INSTRUCTIONS,
    WEIGHTED_REVIEW_OUTPUT_CONTRACT,
)
from align_system.utils import logging
from align_system.utils.agent_logging import agent_log_event, message_payload, preview


log = logging.getLogger(__name__)


REVIEW_RESPONSE_CONTRACT = 'grounded_v2'


def validate_reviewer_contract(reviewer, expected_contract=REVIEW_RESPONSE_CONTRACT):
    """Fail before generation for adapters that have not migrated their evidence."""
    if expected_contract != REVIEW_RESPONSE_CONTRACT:
        raise ValueError("The supported review response contract is 'grounded_v2'.")
    if reviewer is not None and getattr(reviewer, 'response_contract', None) != expected_contract:
        raise ValueError(
            "Reviewers require response_contract='grounded_v2'. Migrate the adapter "
            "to declare that contract and return assessment={'evidence': [...]} "
            "with exact visible-context citations; remove missing_arguments. "
            "The review(context, default_chat_model) interface is unchanged.")


class ReviewAdvice(BaseModel):
    """The reviewer's complete, advisory response."""

    model_config = ConfigDict(extra='forbid', strict=True)

    recommendation: Literal['keep', 'reconsider']
    reason: str = Field(min_length=1)
    assessment: ReviewAssessment

    @field_validator('reason', mode='before')
    @classmethod
    def _strip_reason(cls, value):
        return value.strip() if isinstance(value, str) else value


class ReviewWeighting(BaseModel):
    """Optional ADM-owned advice-strength policy; no learned calibration."""

    model_config = ConfigDict(extra='forbid', strict=True, frozen=True,
                              allow_inf_nan=False)

    influence: float = Field(default=0.5, ge=0, le=1)
    swap_threshold: float = Field(default=0.15, ge=0, le=1)


class ReviewAlternative(BaseModel):
    """A proposed action-tool request, validated semantically by the environment resolver."""

    model_config = ConfigDict(extra='forbid', strict=True, allow_inf_nan=False)

    tool_name: str = Field(min_length=1)
    arguments: dict[str, JsonValue]

    @field_validator('tool_name', mode='before')
    @classmethod
    def _strip_tool_name(cls, value):
        return value.strip() if isinstance(value, str) else value


class WeightedReviewAdvice(ReviewAdvice):
    """Qualitative advice plus an optional concrete, rated alternative."""

    alternative: ReviewAlternative | None
    confidence: float | None = Field(ge=0, le=1, allow_inf_nan=False)
    estimated_improvement: float | None = Field(ge=0, le=1, allow_inf_nan=False)

    @model_validator(mode='after')
    def _validate_comparison(self):
        if self.recommendation == 'keep' or self.alternative is None:
            if (self.alternative is not None or self.confidence is not None
                    or self.estimated_improvement is not None):
                raise ValueError(
                    'Keeping the proposal or giving qualitative critique '
                    'requires null alternative, confidence, and estimated_improvement.')
        elif self.confidence is None or self.estimated_improvement is None:
            raise ValueError('A concrete alternative requires both numerical ratings.')
        return self


class ReviewUnavailableError(ValueError):
    """A reviewer response cannot be used as advice."""

    def __init__(self, message, diagnostics=None):
        super().__init__(message)
        self.diagnostics = deepcopy(diagnostics)


_DIAGNOSTIC_SUMMARIES = {
    'invocation': 'Reviewer model invocation failed.',
    'context_budget': 'Reviewer request could not fit or verify its context budget.',
    'response_shape': 'Reviewer did not return usable visible text without tool calls.',
    'json_parsing': 'Reviewer response is not valid JSON.',
    'schema_validation': 'Reviewer response does not match its required contract.',
    'evidence_validation': 'Reviewer evidence does not match the supplied visible facts.',
}
_CONFIGURATION_ERRORS = {
    'ollama_json_schema': 'ollama_json_schema requires an unbound ChatOllama model.',
    'vllm_json_schema': (
        'vllm_json_schema requires an unbound ChatOpenAI using '
        'Chat Completions with no preconfigured tools or output constraints.'),
}


def _validate_diagnostic_limits(max_diagnostic_errors, max_diagnostic_chars):
    for name, value in (('max_diagnostic_errors', max_diagnostic_errors),
                        ('max_diagnostic_chars', max_diagnostic_chars)):
        if type(value) is not int or value < 1:
            raise ValueError(f'{name} must be a positive integer.')


def review_diagnostics(error, *, expected_schema, response_format='text',
                       stage='invocation', response_text=None,
                       max_diagnostic_errors=20, max_diagnostic_chars=8192):
    """Build bounded, JSON-safe diagnostics without provider/private contents.

    Visible response text is supplied explicitly; exception strings, Pydantic
    inputs/context, and model reasoning are intentionally never serialized.
    Adapter diagnostics keep their more specific response-stage classification.
    """
    _validate_diagnostic_limits(max_diagnostic_errors, max_diagnostic_chars)
    if isinstance(error, ReviewUnavailableError) and error.diagnostics is not None:
        return deepcopy(error.diagnostics)
    errors = (error.errors(include_input=False, include_context=False,
                           include_url=False)
              if isinstance(error, ValidationError)
              and stage in ('json_parsing', 'schema_validation') else [])
    if isinstance(error, EvidenceValidationError) and stage == 'evidence_validation':
        errors = error.validation_errors
    return {
        'stage': stage,
        'expected_schema': expected_schema,
        'response_format': response_format,
        'exception_type': type(error).__name__,
        'summary': (_CONFIGURATION_ERRORS.get(
            response_format, 'Reviewer response-format configuration is unsupported.')
            if stage == 'configuration' else _DIAGNOSTIC_SUMMARIES.get(
                stage, 'Reviewer advice is unavailable.')),
        'response_excerpt': (response_text[:max_diagnostic_chars]
                             if isinstance(response_text, str) else None),
        'response_truncated': (isinstance(response_text, str)
                               and len(response_text) > max_diagnostic_chars),
        'validation_errors': [
            {'loc': list(item['loc']), 'type': item['type'],
             'msg': item['msg'][:512]}
            for item in errors[:max_diagnostic_errors]],
        'validation_errors_truncated': (len(errors) > max_diagnostic_errors or getattr(
            error, 'validation_errors_truncated', False)),
    }


def _generation_metadata(response):
    """Allowlisted provider facts, without guessing why generation stopped."""
    if not isinstance(response, AIMessage):
        return {}
    metadata = response.response_metadata
    metadata = metadata if isinstance(metadata, Mapping) else {}
    generation = {}
    finish_reason = metadata.get('finish_reason')
    if isinstance(finish_reason, str) and finish_reason:
        generation['finish_reason'] = finish_reason[:64]
    normalized = response.usage_metadata
    normalized = normalized if isinstance(normalized, Mapping) else {}
    provider = metadata.get('token_usage')
    provider = provider if isinstance(provider, Mapping) else {}

    def count(name, *values):
        for value in values:
            if type(value) is int and 0 <= value < 2 ** 63:
                generation[name] = value
                return

    for name, provider_name in (
            ('input_tokens', 'prompt_tokens'),
            ('output_tokens', 'completion_tokens'),
            ('total_tokens', 'total_tokens')):
        count(name, normalized.get(name), provider.get(provider_name))
    details = normalized.get('output_token_details')
    details = details if isinstance(details, Mapping) else {}
    provider_details = provider.get('completion_tokens_details')
    provider_details = provider_details if isinstance(provider_details, Mapping) else {}
    count('reasoning_tokens', details.get('reasoning'),
          provider_details.get('reasoning_tokens'))
    return generation


def _response_text(response):
    """Extract public text blocks, including from otherwise invalid responses."""
    if not isinstance(response, AIMessage):
        return None
    content = response.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return ''.join(
            block if isinstance(block, str) else block['text']
            for block in content
            if isinstance(block, str)
            or (isinstance(block, dict) and block.get('type') == 'text'
                and isinstance(block.get('text'), str)))
    return ''


def _visible_text(response):
    """Read text content only; private reasoning never supplies advice."""
    if not isinstance(response, AIMessage):
        raise ReviewUnavailableError('Reviewer did not return an AIMessage.')
    if (response.tool_calls or response.invalid_tool_calls
            or response.additional_kwargs.get('tool_calls')
            or response.additional_kwargs.get('function_call')):
        raise ReviewUnavailableError('Reviewer returned a tool call.')

    return _response_text(response)


def compact_action_catalog(schemas):
    """Remove schema titles without touching literal defaults or evidence values."""
    def compact(schema):
        result = deepcopy(schema)
        if not isinstance(result, dict):
            return result
        result.pop('title', None)
        for key in ('properties', 'patternProperties', '$defs', 'definitions', 'dependentSchemas'):
            if isinstance(result.get(key), dict):
                result[key] = {name: compact(value) for name, value in result[key].items()}
        for key in ('allOf', 'anyOf', 'oneOf', 'prefixItems'):
            if isinstance(result.get(key), list):
                result[key] = [compact(value) for value in result[key]]
        for key in ('items', 'additionalProperties', 'additionalItems', 'contains',
                    'propertyNames', 'not', 'if', 'then', 'else'):
            if isinstance(result.get(key), dict):
                result[key] = compact(result[key])
        return result
    return {name: compact(schema) for name, schema in schemas.items()}


def review_action_tool_models(action_models, *, injury_locations=(), location_guidance=None):
    """Create strict, nonexecutable reviewer argument models from native models.

    The harness supplies only native action models. Treatment's unconstrained
    location gets the harness's canonical values; custom annotations retain
    their original constraints. Native models and their defaults stay unchanged.
    """
    location_guidance = (REVIEW_LOCATION_GUIDANCE if location_guidance is None
                         else location_guidance)
    result = {}
    for name, model in action_models.items():
        fields = {}
        location = model.model_fields.get('injury_location')
        if (name == 'treat_patient' and injury_locations and location is not None
                and location.annotation is str and not location.metadata
                and not location.json_schema_extra):
            location = deepcopy(location)
            location.annotation = Literal[('', *injury_locations)]
            location.description = ' '.join(filter(None, (
                location.description, location_guidance)))
            fields['injury_location'] = (location.annotation, location)
        result[name] = create_model(
            f'{name}_ReviewArguments', __base__=model,
            __config__=ConfigDict(extra='forbid', strict=True), **fields)
    return result


class _ReviewJsonSchema(GenerateJsonSchema):
    """Use singleton enums for Literal values, as required by Ollama grammar."""

    def literal_schema(self, schema):
        result = super().literal_schema(schema)
        if 'const' in result:
            result['enum'] = [result.pop('const')]
        return result


class _UnratedReview(ReviewAdvice):
    alternative: None
    confidence: None
    estimated_improvement: None


class _KeepReview(_UnratedReview):
    recommendation: Literal['keep']


class _QualitativeReconsiderReview(_UnratedReview):
    recommendation: Literal['reconsider']


class _RatedReview(ReviewAdvice):
    recommendation: Literal['reconsider']
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    estimated_improvement: float = Field(ge=0, le=1, allow_inf_nan=False)


def _review_json_schema(weighted, action_models):
    """Compose correlated action/argument and rated/unrated response types.

    Pydantic owns nested definitions, reference names, and required fields.
    Only the harness's offered native action models enter this mapping.
    """
    if not weighted:
        return ReviewAdvice.model_json_schema(schema_generator=_ReviewJsonSchema)
    branches = [_KeepReview, _QualitativeReconsiderReview]
    alternatives = [create_model(
        f'{name}_ReviewAlternative',
        __config__=ConfigDict(extra='forbid', strict=True),
        tool_name=(Literal[name], ...), arguments=(model, ...))
        for name, model in sorted(action_models.items())]
    if alternatives:
        branches.append(create_model(
            '_ConcreteReconsiderReview', __base__=_RatedReview,
            alternative=(Union[tuple(alternatives)], ...)))
    return TypeAdapter(Union[tuple(branches)]).json_schema(
        schema_generator=_ReviewJsonSchema)


def _vllm_review_kwargs(chat_model, schema):
    """Build per-call vLLM constraints without changing the shared model.

    Use the installed ChatOpenAI's effective parameters/routing so settings in
    model_kwargs cannot bypass the guard. The vLLM extension keeps ordinary
    visible-text parsing (including truncated output diagnostics) intact.
    """
    try:
        from langchain_openai import ChatOpenAI
        supported = isinstance(chat_model, ChatOpenAI)
    except ImportError:
        supported = False
    if not supported:
        raise ReviewUnavailableError(
            'vllm_json_schema requires an unbound ChatOpenAI model.')

    params = chat_model._default_params
    if chat_model._use_responses_api(params):
        raise ReviewUnavailableError(
            'vllm_json_schema requires Chat Completions, not the Responses API.')
    body = params.get('extra_body')
    if body is None:
        body = {}
    if not isinstance(body, Mapping):
        raise ReviewUnavailableError(
            'vllm_json_schema requires extra_body to be a mapping.')
    for settings in (params, body):
        if any(key in ('response_format', 'structured_outputs', 'tools',
                       'tool_choice', 'functions', 'function_call')
               or key.startswith('guided_') for key in settings):
            raise ReviewUnavailableError(
                'vllm_json_schema cannot combine preconfigured tools or output constraints.')
    return {'extra_body': {
        **deepcopy(dict(body)), 'structured_outputs': {'json': schema}}}


class ActionReviewer:
    """Invoke a configured unbound model once with fresh review messages.

    The caller owns budgets, weighting arithmetic, and failure handling. Model
    invocation exceptions propagate to the caller, while unusable responses
    raise :class:`ReviewUnavailableError`. No tools are bound or dispatched here.
    """

    def __init__(self, chat_model=None, system_prompt=None, weighting=None,
                 response_format='text', response_contract=REVIEW_RESPONSE_CONTRACT,
                 max_repair_attempts=1, max_diagnostic_errors=20,
                 max_diagnostic_chars=8192, qualitative_output_contract=None,
                 weighted_output_contract=None, alignment_guidance=None,
                 alignment_review_guidance=None, patient_history_guidance=None,
                 scene_end_guidance=None, repair_guidance=None, location_guidance=None):
        if type(max_repair_attempts) is not int or max_repair_attempts not in (0, 1):
            raise ValueError('max_repair_attempts must be 0 or 1.')
        self.max_repair_attempts = max_repair_attempts
        _validate_diagnostic_limits(max_diagnostic_errors, max_diagnostic_chars)
        self.max_diagnostic_errors = max_diagnostic_errors
        self.max_diagnostic_chars = max_diagnostic_chars
        if response_format not in ('text', 'ollama_json_schema', 'vllm_json_schema'):
            raise ValueError(
                'response_format must be text, ollama_json_schema, or vllm_json_schema.')
        self.response_contract = response_contract
        validate_reviewer_contract(self, response_contract)
        self.response_format = response_format
        self.chat_model = chat_model
        self.weighting = (
            None if weighting is None else ReviewWeighting.model_validate(
                dict(weighting) if isinstance(weighting, Mapping) else weighting))
        # None selects a default; an empty string intentionally disables a section.
        self.qualitative_output_contract = (
            COMPLETED_ACTION_REVIEW_GUIDANCE + QUALITATIVE_REVIEW_OUTPUT_CONTRACT
            if qualitative_output_contract is None else qualitative_output_contract)
        self.weighted_output_contract = (
            COMPLETED_ACTION_REVIEW_GUIDANCE + WEIGHTED_REVIEW_OUTPUT_CONTRACT
            if weighted_output_contract is None else weighted_output_contract)
        instructions = REVIEWER_INSTRUCTIONS if system_prompt is None else system_prompt
        contract = (self.qualitative_output_contract if self.weighting is None
                    else self.weighted_output_contract)
        self.system_prompt = instructions + '\n\n' + contract
        self.alignment_guidance = (ALIGNMENT_GUIDANCE if alignment_guidance is None
                                   else alignment_guidance)
        self.alignment_review_guidance = (
            ALIGNMENT_REVIEW_GUIDANCE if alignment_review_guidance is None
            else alignment_review_guidance)
        self.patient_history_guidance = (
            PATIENT_HISTORY_GUIDANCE if patient_history_guidance is None
            else patient_history_guidance)
        self.scene_end_guidance = dict(
            SCENE_END_GUIDANCE if scene_end_guidance is None else scene_end_guidance)
        self.repair_guidance = (REVIEW_REPAIR_GUIDANCE if repair_guidance is None
                                else repair_guidance)
        self.location_guidance = (REVIEW_LOCATION_GUIDANCE if location_guidance is None
                                  else location_guidance)
        self._transport_schemas = {}

    def _diagnostics(self, error, **kwargs):
        return review_diagnostics(
            error, max_diagnostic_errors=self.max_diagnostic_errors,
            max_diagnostic_chars=self.max_diagnostic_chars, **kwargs)

    def _response_schema(self, action_models):
        # Cache only types/schema values, never facts or provider responses.
        key = (self.weighting is not None, tuple(sorted(action_models.items())))
        if key not in self._transport_schemas:
            if len(self._transport_schemas) >= 32:
                self._transport_schemas.clear()
            self._transport_schemas[key] = _review_json_schema(
                self.weighting is not None, action_models)
        return deepcopy(self._transport_schemas[key])

    def review(self, context: dict, default_chat_model, *, request_budget=None,
               attempt_metadata=None, action_models=None) -> ReviewAdvice:
        """Review once; optionally report provider metadata to the caller's dict.

        The workflow owns bounded repair attempts. ``action_models`` contains
        only the offered native argument models, supplied separately from JSON
        context. Custom reviewer adapters keep their two-argument interface.
        The metadata hook never changes model or reviewer state.
        """
        if attempt_metadata is not None:
            attempt_metadata.clear()
        chat_model = (self.chat_model if self.chat_model is not None
                      else default_chat_model)
        advice_type = (ReviewAdvice if self.weighting is None
                       else WeightedReviewAdvice)
        invoke_kwargs = {}
        if self.response_format == 'ollama_json_schema':
            # Import lazily so text mode keeps its existing optional-dependency
            # requirements. A bound wrapper is not an unbound ChatOllama model.
            try:
                from langchain_ollama import ChatOllama
                supported = isinstance(chat_model, ChatOllama)
            except ImportError:
                supported = False
            if not supported:
                error = ReviewUnavailableError(
                    'ollama_json_schema requires an unbound ChatOllama model.')
                error.diagnostics = self._diagnostics(
                    error, expected_schema=advice_type.__name__,
                    response_format=self.response_format, stage='configuration')
                raise error
            invoke_kwargs['format'] = self._response_schema(action_models or {})
        elif self.response_format == 'vllm_json_schema':
            try:
                invoke_kwargs = _vllm_review_kwargs(chat_model, self._response_schema(action_models or {}))
            except ReviewUnavailableError as exc:
                raise ReviewUnavailableError(str(exc), diagnostics=self._diagnostics(
                    exc, expected_schema=advice_type.__name__,
                    response_format=self.response_format, stage='configuration')) from exc
        system_prompt = self.system_prompt
        if context.get('alignment_context') is not None:
            system_prompt += ('\n\n' + self.alignment_guidance
                              + '\n\n' + self.alignment_review_guidance)
        if context.get('patient_history') is not None:
            system_prompt += '\n\n' + self.patient_history_guidance
        if context.get('scene_end_policy') in self.scene_end_guidance:
            system_prompt += '\n\n' + self.scene_end_guidance[context['scene_end_policy']]
        if context.get('repair_feedback') is not None:
            system_prompt += '\n\n' + self.repair_guidance
        model_context = dict(context)
        schemas = model_context.pop('action_tool_schemas', None)
        if schemas is not None:
            model_context['action_tools'] = compact_action_catalog(schemas)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=json.dumps(model_context, ensure_ascii=False, separators=(',', ':'))),
        ]
        if request_budget is not None:
            from align_system.algorithms.agents.agent_context_budget import ContextBudgetError

            try:
                prepared = request_budget.prepare(messages, invoke_kwargs)
            except ContextBudgetError as exc:
                diagnostics = self._diagnostics(
                    exc, expected_schema=advice_type.__name__,
                    response_format=self.response_format, stage='context_budget')
                diagnostics['context_budget'] = deepcopy(exc.diagnostics)
                raise ReviewUnavailableError(
                    diagnostics['summary'], diagnostics=diagnostics) from exc
            messages, invoke_kwargs = prepared.messages, prepared.invoke_kwargs
            diagnostics = {
                **prepared.diagnostics, 'role': 'reviewer',
                'removed_exchanges': 0, 'recovery': 'none'}
            agent_log_event(
                log, 'REVIEWER', 'CONTEXT BUDGET',
                'Prepared reviewer request within the context budget.',
                payload=diagnostics, legacy_message='Context budget: %s',
                legacy_args=(json.dumps(diagnostics),))
        # Capture only model-visible messages and the prepared output schema.
        # Other provider kwargs may contain credentials or opaque objects.
        output_schema = invoke_kwargs.get('format')
        if self.response_format == 'vllm_json_schema':
            output_schema = invoke_kwargs.get('extra_body', {}).get(
                'structured_outputs', {}).get('json')
        agent_log_event(
            log, 'REVIEWER', 'INPUT',
            f'messages={len(messages)} | response_format={self.response_format}'
            f' | repair={context.get("repair_feedback") is not None}',
            payload=lambda: {'messages': message_payload(messages),
                             'output_schema': output_schema})
        response = chat_model.invoke(messages, **invoke_kwargs)
        response_text = _response_text(response)
        agent_log_event(
            log, 'REVIEWER', 'OUTPUT',
            f'Visible response preview: {preview(response_text or "")}',
            payload=lambda: {'response_text': response_text})
        if attempt_metadata is not None:
            attempt_metadata.update(_generation_metadata(response))
        try:
            text = _visible_text(response).strip()
            if not text:
                raise ReviewUnavailableError('Reviewer returned no visible text.')
        except ReviewUnavailableError as exc:
            raise ReviewUnavailableError(str(exc), diagnostics=self._diagnostics(
                exc, expected_schema=advice_type.__name__,
                response_format=self.response_format, stage='response_shape',
                response_text=_response_text(response))) from exc
        try:
            return advice_type.model_validate_json(text)
        except ValidationError as exc:
            raise ReviewUnavailableError(
                'Reviewer response does not match the required '
                f'{advice_type.__name__} JSON contract.',
                diagnostics=self._diagnostics(
                    exc, expected_schema=advice_type.__name__,
                    response_format=self.response_format,
                    stage=('json_parsing' if any(
                        item['type'] == 'json_invalid' for item in exc.errors())
                        else 'schema_validation'),
                    response_text=_response_text(response))) from exc
