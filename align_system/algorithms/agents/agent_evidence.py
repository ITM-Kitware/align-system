"""Bounded structured evidence checks against explicitly visible JSON data.

Evidence means an exact JSON Pointer and value copied from visible scenario,
proposal, history, or alignment data. These checks establish that cited facts
exist and match the recorded snapshot; they are not retrieved medical knowledge.
They do not judge whether a fact supports the model's clinical reasoning.
"""

from copy import deepcopy
import json
import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator


MAX_EVIDENCE_REFERENCES = 20
MAX_POINTER_LENGTH = 512
MAX_POINTER_DEPTH = 16
MAX_EVIDENCE_VALUE_LENGTH = 8192
MAX_EVIDENCE_VALUE_DEPTH = 16
REVIEW_EVIDENCE_ROOTS = (
    '/observation', '/proposal_tool/arguments', '/proposal_validation',
    '/handled_patients', '/provider_context', '/alignment_context', '/patient_history',
)
FINAL_EVIDENCE_ROOTS = tuple(
    prefix + root for prefix in ('/review', '/current')
    for root in REVIEW_EVIDENCE_ROOTS)


def build_citation_guide(context, prefix=''):
    """Map identifiers to actual snapshot paths without adding evidence facts.

    Array positions are never inferred from casualty names. Guides themselves
    are not evidence roots, and callers still validate every cited value.
    """
    context = context or {}
    history = context.get('patient_history') or {}
    patients = []
    seen = set()
    for index, casualty in enumerate((context.get('observation') or {}).get('casualties', [])):
        if not isinstance(casualty, dict):
            continue
        identifier = casualty.get('id')
        if not isinstance(identifier, str):
            continue
        entry = {'id': identifier, 'observation_path': f'{prefix}/observation/casualties/{index}'}
        if identifier in history:
            escaped = identifier.replace('~', '~0').replace('/', '~1')
            entry['history_path'] = f'{prefix}/patient_history/{escaped}'
        patients.append(entry)
        seen.add(identifier)
    for identifier in history:
        if isinstance(identifier, str) and identifier not in seen:
            escaped = identifier.replace('~', '~0').replace('/', '~1')
            patients.append({'id': identifier, 'history_path': f'{prefix}/patient_history/{escaped}'})
    kdmas = []
    for index, entry in enumerate((context.get('alignment_context') or {}).get('kdmas', [])):
        if not isinstance(entry, dict) or not isinstance(entry.get('kdma'), str):
            continue
        base = f'{prefix}/alignment_context/kdmas/{index}'
        kdmas.append({'kdma': entry['kdma'], 'identifier_path': base + '/kdma',
                      'value_path': base + '/value', 'name_path': base + '/definition/name',
                      'examples_path': base + '/definition/score_examples'})
    return {'casualties': patients, 'kdmas': kdmas}


def _bounded_value(value, depth=0):
    if depth > MAX_EVIDENCE_VALUE_DEPTH:
        raise ValueError('Evidence value exceeds the nesting limit.')
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Evidence values must be finite JSON values.')
    if isinstance(value, dict):
        for item in value.values():
            _bounded_value(item, depth + 1)
    elif isinstance(value, list):
        for item in value:
            _bounded_value(item, depth + 1)
    return value


class EvidenceReference(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, allow_inf_nan=False)

    path: str = Field(min_length=1, max_length=MAX_POINTER_LENGTH)
    value: JsonValue

    @field_validator('value')
    @classmethod
    def _limit_value(cls, value):
        _bounded_value(value)
        if len(json.dumps(value, ensure_ascii=False, allow_nan=False)) > MAX_EVIDENCE_VALUE_LENGTH:
            raise ValueError('Evidence value exceeds the size limit.')
        return value


class ReviewAssessment(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)

    evidence: list[EvidenceReference] = Field(
        min_length=1, max_length=MAX_EVIDENCE_REFERENCES)


class FinalReviewResponse(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)

    acknowledged_suggestion: Literal['keep', 'switch', 'none']
    response_to_advice: str = Field(min_length=1, max_length=8192)
    evidence: list[EvidenceReference] = Field(
        min_length=1, max_length=MAX_EVIDENCE_REFERENCES)

    @field_validator('response_to_advice', mode='before')
    @classmethod
    def _strip_response(cls, value):
        return value.strip() if isinstance(value, str) else value


class EvidenceValidationError(ValueError):
    """Safe diagnostics; never contains rejected values or provider contents."""

    def __init__(self, errors):
        super().__init__('Structured evidence does not match the visible context.')
        self.validation_errors = deepcopy(errors[:20])
        self.validation_errors_truncated = len(errors) > 20


def _error(loc, kind, message):
    return {'loc': list(loc), 'type': kind, 'msg': message[:512]}


def _tokens(path):
    if (not isinstance(path, str) or not path.startswith('/')
            or len(path) > MAX_POINTER_LENGTH):
        raise ValueError('Evidence path must be a bounded absolute JSON Pointer.')
    encoded = path[1:].split('/')
    if len(encoded) > MAX_POINTER_DEPTH:
        raise ValueError('Evidence path exceeds the depth limit.')
    decoded = []
    for token in encoded:
        index = 0
        while index < len(token):
            if token[index] == '~':
                if index + 1 == len(token) or token[index + 1] not in '01':
                    raise ValueError('Evidence path contains an invalid JSON Pointer escape.')
                index += 1
            index += 1
        decoded.append(token.replace('~1', '/').replace('~0', '~'))
    return decoded


def _lookup(context, tokens):
    current = context
    for token in tokens:
        if isinstance(current, dict):
            if token not in current:
                raise LookupError
            current = current[token]
        elif isinstance(current, list):
            if (not token.isascii() or not token.isdigit()
                    or (len(token) > 1 and token.startswith('0'))):
                raise LookupError
            index = int(token)
            if index >= len(current):
                raise LookupError
            current = current[index]
        else:
            raise LookupError
    return current


def _same_json_value(left, right):
    """JSON numeric equality never makes true equal to 1 (including nested data)."""
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is bool and type(right) is bool and left == right
    if left is None or right is None:
        return left is None and right is None
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return ((not isinstance(left, float) or math.isfinite(left))
                and (not isinstance(right, float) or math.isfinite(right))
                and left == right)
    if isinstance(left, dict) and isinstance(right, dict):
        return (left.keys() == right.keys()
                and all(_same_json_value(left[key], right[key]) for key in left))
    if isinstance(left, list) and isinstance(right, list):
        return (len(left) == len(right)
                and all(_same_json_value(a, b) for a, b in zip(left, right)))
    return type(left) is type(right) and left == right


def validate_evidence(evidence, context, allowed_roots=REVIEW_EVIDENCE_ROOTS):
    """Check bounded pointers and exact values without exposing rejected data."""
    from pydantic import ValidationError

    if not isinstance(evidence, (list, tuple)) or not 1 <= len(evidence) <= MAX_EVIDENCE_REFERENCES:
        raise EvidenceValidationError([_error(
            ('evidence',), 'evidence_count', 'Supply between 1 and 20 evidence references.')])
    root_tokens = [tuple(_tokens(root)) for root in allowed_roots]
    errors = []
    for index, item in enumerate(evidence):
        loc = ('evidence', index)
        try:
            if isinstance(item, EvidenceReference):
                item = item.model_dump(mode='python')
            reference = EvidenceReference.model_validate(item)
        except ValidationError:
            errors.append(_error(loc, 'evidence_shape', 'Evidence reference has an invalid shape or exceeds a limit.'))
            continue
        try:
            tokens = tuple(_tokens(reference.path))
        except ValueError as error:
            errors.append(_error((*loc, 'path'), 'evidence_path', str(error)))
            continue
        if not any(tokens[:len(root)] == root for root in root_tokens):
            errors.append(_error((*loc, 'path'), 'evidence_root', 'Evidence path is outside the permitted visible context.'))
            continue
        try:
            actual = _lookup(context, tokens)
        except LookupError:
            errors.append(_error((*loc, 'path'), 'evidence_missing', 'Evidence path does not exist in the visible context.'))
            continue
        if not _same_json_value(actual, reference.value):
            errors.append(_error((*loc, 'value'), 'evidence_mismatch', 'Evidence value does not match the cited visible fact.'))
    if errors:
        raise EvidenceValidationError(errors)


def validate_review_assessment(assessment, context):
    """Validate the grounded evidence shape and exact cited visible facts."""
    from pydantic import ValidationError

    try:
        if isinstance(assessment, ReviewAssessment):
            assessment = assessment.model_dump(mode='python')
        assessment = ReviewAssessment.model_validate(assessment)
    except ValidationError:
        raise EvidenceValidationError([_error(
            ('assessment',), 'assessment_shape', 'Assessment has an invalid shape or exceeds a limit.')]) from None
    validate_evidence(assessment.evidence, context, REVIEW_EVIDENCE_ROOTS)
