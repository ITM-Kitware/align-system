"""A stateless, tool-free reviewer for proposed open-world actions."""

from collections.abc import Mapping
from copy import deepcopy
import json
from typing import Literal, Union

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import (
    BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, ValidationError, WithJsonSchema,
    create_model, field_validator, model_validator)
from pydantic.json_schema import GenerateJsonSchema
from pydantic_core import PydanticUndefined

from align_system.algorithms.agents.agent_evidence import (
    EvidenceValidationError, REVIEW_EVIDENCE_ROOTS,
    MAX_EVIDENCE_REFERENCES, MAX_EVIDENCE_VALUE_LENGTH, MAX_POINTER_LENGTH,
    FinalReviewResponse, ReviewAssessment, validate_evidence,
    validate_review_assessment,
)
from align_system.algorithms.agents.chat_transport import (
    ChatCapabilities, generation_metadata as _generation_metadata,
    invoke_chat, response_failure as _openai_response_failure,
    response_text as _response_text, structured_review_kwargs,
)
from align_system.prompt_engineering.agentic_prompts import (
    ALIGNMENT_GUIDANCE, PATIENT_HISTORY_GUIDANCE, SCENE_END_GUIDANCE,
)
from align_system.prompt_engineering.reviewer_prompts import (
    ALIGNMENT_REVIEW_GUIDANCE, COMPLETED_ACTION_REVIEW_GUIDANCE,
    QUALITATIVE_REVIEW_OUTPUT_CONTRACT,
    OPENAI_QUALITATIVE_REVIEW_OUTPUT_CONTRACT,
    OPENAI_WEIGHTED_REVIEW_OUTPUT_CONTRACT,
    REVIEW_REPAIR_GUIDANCE, REVIEW_LOCATION_GUIDANCE, REVIEWER_INSTRUCTIONS,
    WEIGHTED_REVIEW_OUTPUT_CONTRACT,
)
from align_system.utils import logging
from align_system.utils.agent_logging import agent_log_event, message_payload, preview


log = logging.getLogger(__name__)


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
    'provider_refusal': 'Reviewer provider refused the request.',
}
_CONFIGURATION_ERRORS = {
    'json_schema': (
        'json_schema requires an explicit provider and API, matching an unbound '
        'ChatOpenAI with no preconfigured tools or output constraints.'),
    'ollama_json_schema': 'ollama_json_schema requires an unbound ChatOllama model.',
    'vllm_json_schema': (
        'vllm_json_schema requires an unbound ChatOpenAI using '
        'Chat Completions with no preconfigured tools or output constraints.'),
    'openai_json_schema': (
        'openai_json_schema requires an unbound ChatOpenAI with no '
        'preconfigured tools or output constraints.'),
}


class _ReviewJsonError(ValueError):
    """A safe JSON parsing diagnostic, without echoing arbitrary input keys."""

    def __init__(self, message, loc=()):
        super().__init__(message)
        self.validation_errors = [{'loc': loc, 'type': 'json_invalid', 'msg': message}]


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
    if isinstance(error, _ReviewJsonError) and stage == 'json_parsing':
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
            {'loc': list(item['loc'][:-1] if item['type'] == 'extra_forbidden'
                         else item['loc']), 'type': item['type'],
             'msg': item['msg'][:512]}
            for item in errors[:max_diagnostic_errors]],
        'validation_errors_truncated': (len(errors) > max_diagnostic_errors or getattr(
            error, 'validation_errors_truncated', False)),
    }


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
        native_location_schema = (location is not None and len(location.metadata) == 1
                                  and isinstance(location.metadata[0], WithJsonSchema)
                                  and location.metadata[0].mode is None
                                  and location.metadata[0].json_schema == {
                                      'type': 'string', 'enum': ['', *injury_locations]})
        if (name == 'treat_patient' and injury_locations and location is not None
                and location.annotation is str
                and (not location.metadata or native_location_schema)
                and not location.json_schema_extra):
            location = deepcopy(location)
            location.annotation = Literal[('', *injury_locations)]
            if native_location_schema:
                location.metadata = []
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


class _OpenAIReviewEvidence(BaseModel):
    """Encode arbitrary JSON values without an unconstrained strict-schema field."""

    model_config = ConfigDict(extra='forbid', strict=True)

    path: str = Field(min_length=1, max_length=MAX_POINTER_LENGTH)
    value_json: str = Field(min_length=1, max_length=MAX_EVIDENCE_VALUE_LENGTH)


class _OpenAIReviewAssessment(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)

    evidence: list[_OpenAIReviewEvidence] = Field(
        min_length=1, max_length=MAX_EVIDENCE_REFERENCES)


def _openai_review_model(weighted, action_models):
    """Build a closed root and correlated variants for OpenAI strict outputs.

    Evidence remains arbitrary JSON in the canonical contract. Its wire string
    avoids unrestricted objects, while explicit argument defaults preserve the
    native action's semantics under OpenAI's all-fields-required constraint.
    """
    assessment_field = (_OpenAIReviewAssessment, ...)
    if not weighted:
        advice = create_model(
            '_OpenAIQualitativeReview', __base__=ReviewAdvice,
            assessment=assessment_field)
    else:
        branches = [create_model(
            f'_OpenAI{base.__name__}', __base__=base,
            assessment=assessment_field)
            for base in (_KeepReview, _QualitativeReconsiderReview)]
        alternatives = []
        for name, model in sorted(action_models.items()):
            fields = {}
            for field_name, original in model.model_fields.items():
                field = deepcopy(original)
                if not field.is_required() and field.default is not PydanticUndefined:
                    default = json.dumps(field.default, ensure_ascii=False, allow_nan=False)
                    field.description = ' '.join(filter(None, (
                        field.description, f'Use {default} for the native default.')))
                field.default = PydanticUndefined
                field.default_factory = None
                fields[field_name] = (field.annotation, field)
            arguments = create_model(
                f'{name}_OpenAIReviewArguments', __base__=model,
                __config__=ConfigDict(extra='forbid', strict=True), **fields)
            alternatives.append(create_model(
                f'{name}_OpenAIReviewAlternative',
                __config__=ConfigDict(extra='forbid', strict=True),
                tool_name=(Literal[name], ...), arguments=(arguments, ...)))
        if alternatives:
            branches.append(create_model(
                '_OpenAIConcreteReconsiderReview', __base__=_RatedReview,
                assessment=assessment_field,
                alternative=(Union[tuple(alternatives)], ...)))
        advice = Union[tuple(branches)]
    return create_model(
        'OpenAIReviewEnvelope', __config__=ConfigDict(extra='forbid', strict=True),
        advice=(advice, ...))


def _strict_review_json(text, loc=()):
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise _ReviewJsonError('JSON object contains a duplicate field.', loc)
            result[key] = value
        return result

    def invalid_constant(value):
        raise _ReviewJsonError('JSON numbers must be finite.', loc)

    try:
        return json.loads(text, object_pairs_hook=object_pairs,
                          parse_constant=invalid_constant)
    except json.JSONDecodeError as exc:
        raise _ReviewJsonError(
            f'Invalid JSON at line {exc.lineno}, column {exc.colno}.', loc) from exc
    except RecursionError as exc:
        raise _ReviewJsonError('JSON nesting exceeds the parser limit.', loc) from exc
    except _ReviewJsonError:
        raise
    except ValueError as exc:
        raise _ReviewJsonError('JSON value exceeds the parser limits.', loc) from exc


def _parse_openai_review(text, transport_model, advice_type):
    envelope = transport_model.model_validate(_strict_review_json(text), strict=True)
    advice = envelope.advice.model_dump()
    for index, evidence in enumerate(advice['assessment']['evidence']):
        value_json = evidence.pop('value_json')
        evidence['value'] = _strict_review_json(
            value_json, ('advice', 'assessment', 'evidence', index, 'value_json'))
    return advice_type.model_validate(advice, strict=True)


class GroundedReviewContract:
    """Schemas and evidence validation for advisory action review.

    Defines keep/reconsider advice, optional ratings and alternative actions,
    and the decision agent's response to that advice. Evidence references pair
    a JSON pointer with an exact value from the visible context; validation
    checks those references without evaluating clinical reasoning.

    Instances are stateless and configurable through Hydra. The contract name
    identifies the format in reviewer declarations and recorded review metadata.
    """

    name = 'visible_fact_review'
    review_evidence_roots = REVIEW_EVIDENCE_ROOTS
    final_response_model = FinalReviewResponse

    def advice_model(self, weighted=False):
        return WeightedReviewAdvice if weighted else ReviewAdvice

    def response_schema(self, weighted, action_models):
        return _review_json_schema(weighted, action_models)

    def validate_assessment(self, assessment, context):
        validate_review_assessment(
            assessment, context, allowed_roots=self.review_evidence_roots)

    def validate_final_evidence(self, evidence, context):
        final_roots = tuple(
            prefix + root for prefix in ('/review', '/current')
            for root in self.review_evidence_roots)
        validate_evidence(evidence, context, final_roots)


def resolve_review_contract(value=None):
    """Use the default contract for None, or accept a configured instance."""
    if value is None:
        return GroundedReviewContract()
    if isinstance(value, GroundedReviewContract):
        return value
    raise TypeError(
        'Review response contract must be a GroundedReviewContract instance '
        'or None. Use Hydra _target_ to configure a contract instance.')


def validate_reviewer_contract(reviewer, expected_contract=None):
    """Resolve the contract and reject mismatched adapters before generation."""
    contract = resolve_review_contract(expected_contract)
    if reviewer is not None:
        declared = getattr(reviewer, 'response_contract', None)
        if isinstance(declared, GroundedReviewContract):
            declared = declared.name
        if not isinstance(declared, str) or declared != contract.name:
            raise ValueError(
                f'Reviewers must declare response_contract={contract.name!r}. '
                'review(context, default_chat_model) must return advice containing '
                "assessment={'evidence': [...]} with exact visible-context citations.")
    return contract


def _vllm_review_kwargs(chat_model, schema):
    """Compatibility wrapper for the legacy vLLM response-format alias."""
    try:
        return structured_review_kwargs(chat_model, schema, provider='vllm')
    except ValueError as exc:
        raise ReviewUnavailableError(_CONFIGURATION_ERRORS['vllm_json_schema']) from exc


def _openai_review_kwargs(chat_model, schema):
    """Compatibility wrapper for the legacy OpenAI response-format alias."""
    try:
        return structured_review_kwargs(chat_model, schema, provider='openai')
    except ValueError as exc:
        raise ReviewUnavailableError(_CONFIGURATION_ERRORS['openai_json_schema']) from exc


def _invoke_openai_review(chat_model, messages, invoke_kwargs):
    return invoke_chat(chat_model, messages, **invoke_kwargs)


class ActionReviewer:
    """Invoke a configured unbound model once with fresh review messages.

    The caller owns budgets, weighting arithmetic, and failure handling. Model
    invocation exceptions propagate to the caller, while unusable responses
    raise :class:`ReviewUnavailableError`. No tools are bound or dispatched here.
    """

    def __init__(self, chat_model=None, system_prompt=None, weighting=None,
                 response_format='text', response_contract=None,
                 max_repair_attempts=1, max_diagnostic_errors=20,
                 max_diagnostic_chars=8192, qualitative_output_contract=None,
                 weighted_output_contract=None, alignment_guidance=None,
                 alignment_review_guidance=None, patient_history_guidance=None,
                 scene_end_guidance=None, repair_guidance=None, location_guidance=None,
                 provider=None, api=None):
        if type(max_repair_attempts) is not int or max_repair_attempts not in (0, 1):
            raise ValueError('max_repair_attempts must be 0 or 1.')
        self.max_repair_attempts = max_repair_attempts
        _validate_diagnostic_limits(max_diagnostic_errors, max_diagnostic_chars)
        self.max_diagnostic_errors = max_diagnostic_errors
        self.max_diagnostic_chars = max_diagnostic_chars
        if response_format not in ('text', 'ollama_json_schema', 'vllm_json_schema',
                                   'openai_json_schema', 'json_schema'):
            raise ValueError(
                'response_format must be text, ollama_json_schema, vllm_json_schema, '
                'openai_json_schema, or json_schema.')
        if response_format == 'json_schema':
            self.capabilities = ChatCapabilities(provider, api)
        else:
            if provider is not None or api is not None:
                raise ValueError('provider and api require response_format=json_schema.')
            self.capabilities = None
        self._transport_format = (
            f'{provider}_json_schema' if response_format == 'json_schema' else response_format)
        self.contract = resolve_review_contract(response_contract)
        self.response_contract = self.contract.name
        self.response_format = response_format
        self.chat_model = chat_model
        self.weighting = (
            None if weighting is None else ReviewWeighting.model_validate(
                dict(weighting) if isinstance(weighting, Mapping) else weighting))
        # None selects a default; an empty string intentionally disables a section.
        qualitative_default = (OPENAI_QUALITATIVE_REVIEW_OUTPUT_CONTRACT
                               if self._transport_format == 'openai_json_schema'
                               else QUALITATIVE_REVIEW_OUTPUT_CONTRACT)
        weighted_default = (OPENAI_WEIGHTED_REVIEW_OUTPUT_CONTRACT
                            if self._transport_format == 'openai_json_schema'
                            else WEIGHTED_REVIEW_OUTPUT_CONTRACT)
        self.qualitative_output_contract = (
            COMPLETED_ACTION_REVIEW_GUIDANCE + qualitative_default
            if qualitative_output_contract is None else qualitative_output_contract)
        self.weighted_output_contract = (
            COMPLETED_ACTION_REVIEW_GUIDANCE + weighted_default
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
        self._openai_transport_models = {}

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
            if self._transport_format == 'openai_json_schema':
                self._transport_schemas[key] = self._openai_transport_model(
                    action_models).model_json_schema(schema_generator=_ReviewJsonSchema)
            else:
                self._transport_schemas[key] = self.contract.response_schema(
                    self.weighting is not None, action_models)
        return deepcopy(self._transport_schemas[key])

    def _openai_transport_model(self, action_models):
        key = (self.weighting is not None, tuple(sorted(action_models.items())))
        if key not in self._openai_transport_models:
            if len(self._openai_transport_models) >= 32:
                self._openai_transport_models.clear()
            self._openai_transport_models[key] = _openai_review_model(
                self.weighting is not None, action_models)
        return self._openai_transport_models[key]

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
        advice_type = self.contract.advice_model(self.weighting is not None)
        invoke_kwargs = {}
        if self._transport_format == 'ollama_json_schema':
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
        elif self._transport_format in ('vllm_json_schema', 'openai_json_schema'):
            try:
                schema = self._response_schema(action_models or {})
                if self.capabilities is not None:
                    invoke_kwargs = structured_review_kwargs(
                        chat_model, schema, provider=self.capabilities.provider,
                        api=self.capabilities.api)
                else:
                    builder = (_openai_review_kwargs
                               if self._transport_format == 'openai_json_schema'
                               else _vllm_review_kwargs)
                    invoke_kwargs = builder(chat_model, schema)
            except ValueError as exc:
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
        tokenization_elapsed_s = None
        if request_budget is not None:
            from align_system.algorithms.agents.agent_context_budget import ContextBudgetError

            tokenization_before = getattr(request_budget, 'tokenization_elapsed_s', 0.0)
            try:
                prepared = request_budget.prepare(messages, invoke_kwargs)
            except ContextBudgetError as exc:
                diagnostics = self._diagnostics(
                    exc, expected_schema=advice_type.__name__,
                    response_format=self.response_format, stage='context_budget')
                diagnostics['context_budget'] = deepcopy(exc.diagnostics)
                raise ReviewUnavailableError(
                    diagnostics['summary'], diagnostics=diagnostics) from exc
            tokenization_elapsed_s = (
                getattr(request_budget, 'tokenization_elapsed_s', 0.0) - tokenization_before)
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
        if self._transport_format == 'vllm_json_schema':
            output_schema = invoke_kwargs.get('extra_body', {}).get(
                'structured_outputs', {}).get('json')
        elif self._transport_format == 'openai_json_schema':
            output_schema = invoke_kwargs.get('response_format', {}).get(
                'json_schema', {}).get('schema')
        agent_log_event(
            log, 'REVIEWER', 'INPUT',
            f'messages={len(messages)} | response_format={self.response_format}'
            f' | repair={context.get("repair_feedback") is not None}',
            payload=lambda: {'messages': message_payload(messages),
                             'output_schema': output_schema})
        from align_system.utils.model_telemetry import invoke_with_telemetry

        def invoke(messages, **kwargs):
            if self._transport_format in ('openai_json_schema', 'vllm_json_schema'):
                return _invoke_openai_review(chat_model, messages, kwargs)
            return chat_model.invoke(messages, **kwargs)

        response = invoke_with_telemetry(
            chat_model, messages, invoke_kwargs,
            phase='review_repair' if context.get('repair_feedback') is not None else 'review',
            invoke=invoke, tokenization_elapsed_s=tokenization_elapsed_s)
        response_text = _response_text(response)
        # Preserve strict OpenAI output semantics. Other transports validate
        # the visible response itself and retain their existing repair feedback.
        failure_stage = (_openai_response_failure(response)
                         if self._transport_format == 'openai_json_schema' else None)
        if failure_stage == 'provider_refusal':
            response_text = None
        agent_log_event(
            log, 'REVIEWER', 'OUTPUT',
            f'Visible response preview: {preview(response_text or "")}',
            payload=lambda: {'response_text': response_text})
        if attempt_metadata is not None:
            attempt_metadata.update(_generation_metadata(response))
        if failure_stage is not None:
            error = ReviewUnavailableError(_DIAGNOSTIC_SUMMARIES[failure_stage])
            raise ReviewUnavailableError(str(error), diagnostics=self._diagnostics(
                error, expected_schema=advice_type.__name__,
                response_format=self.response_format, stage=failure_stage,
                response_text=(None if failure_stage == 'provider_refusal'
                               else response_text)))
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
            if self._transport_format == 'openai_json_schema':
                return _parse_openai_review(
                    text, self._openai_transport_model(action_models or {}), advice_type)
            return advice_type.model_validate_json(text)
        except _ReviewJsonError as exc:
            raise ReviewUnavailableError(
                _DIAGNOSTIC_SUMMARIES['json_parsing'], diagnostics=self._diagnostics(
                    exc, expected_schema=advice_type.__name__,
                    response_format=self.response_format, stage='json_parsing',
                    response_text=response_text)) from exc
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
