"""Shared, scalar Phase2 alignment context for decision and review agents."""

from collections.abc import Mapping, Sequence
from copy import deepcopy
import math
from numbers import Real

from jinja2 import StrictUndefined, TemplateError
from jinja2.sandbox import SandboxedEnvironment
from omegaconf import OmegaConf


def _as_mapping(value, label):
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    elif hasattr(value, 'model_dump'):
        value = value.model_dump()
    elif hasattr(value, 'to_dict'):
        value = value.to_dict()
    if not isinstance(value, Mapping):
        raise ValueError(f'{label} must be a mapping or model object')
    return deepcopy(dict(value))


def _finite_number(value, label):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f'{label} must be a finite number')
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f'{label} must be a finite number') from exc
    if not math.isfinite(result):
        raise ValueError(f'{label} must be a finite number')
    return result


def _definition_view(kdma, definitions):
    if kdma not in definitions or definitions[kdma] is None:
        raise ValueError(f'Missing alignment attribute definition for {kdma!r}')
    definition = _as_mapping(definitions[kdma], f'Definition for {kdma!r}')
    if definition.get('kdma', kdma) != kdma:
        raise ValueError(f'Alignment attribute definition does not match {kdma!r}')
    for field in ('name', 'description'):
        if not isinstance(definition.get(field), str) or not definition[field].strip():
            raise ValueError(f'Definition for {kdma!r} requires a nonempty {field}')
    factor = _finite_number(definition.get('factor'), f'Score factor for {kdma!r}')
    if factor <= 0:
        raise ValueError(f'Score factor for {kdma!r} must be positive')
    examples = definition.get('score_examples')
    if examples is not None:
        if not isinstance(examples, str):
            raise ValueError(f'Score examples for {kdma!r} must be text')
        # Configured attribute definitions can contain Jinja expressions using
        # this factor. Keep rendering confined to the configured numerical scale.
        environment = SandboxedEnvironment(undefined=StrictUndefined, autoescape=False)
        try:
            examples = environment.from_string(examples).render(kdma_scale_factor=factor)
        except TemplateError as exc:
            raise ValueError(f'Invalid score examples for {kdma!r}: {exc}') from exc
    return {
        'name': definition['name'],
        'description': definition['description'],
        'score_examples': examples,
        'score_scale': {'min': 0, 'max': factor},
    }


def build_alignment_context(alignment_target, attribute_definitions):
    """Copy a scalar target and its configured definitions into prompt context.

    Called only for an aligned session. Unsupported target forms fail explicitly
    rather than being silently treated as scalar preferences.
    """
    if alignment_target is None:
        raise ValueError('Alignment is enabled but no alignment target was supplied')
    target = _as_mapping(alignment_target, 'Alignment target')
    entries = target.get('kdma_values')
    if OmegaConf.is_config(entries):
        entries = OmegaConf.to_container(entries, resolve=True)
    if (not isinstance(entries, Sequence)
            or isinstance(entries, (str, bytes)) or not entries):
        raise ValueError('Alignment target requires nonempty kdma_values')
    definitions = _as_mapping(attribute_definitions, 'Alignment attribute definitions')
    target_id = target.get('id')
    if target_id is not None and not isinstance(target_id, str):
        raise ValueError('Alignment target id must be text or null')
    kdmas = []
    seen = set()
    for entry in entries:
        entry = _as_mapping(entry, 'Alignment KDMA value')
        kdma = entry.get('kdma')
        if not isinstance(kdma, str) or not kdma.strip():
            raise ValueError('Alignment KDMA value requires a nonempty kdma identifier')
        if kdma in seen:
            raise ValueError(f'Duplicate alignment target KDMA {kdma!r}')
        seen.add(kdma)
        for field in ('parameters', 'kdes', 'scores'):
            if entry.get(field):
                raise ValueError(
                    f'Alignment context supports scalar targets only; '
                    f'{kdma!r} has unsupported {field}')
        value = _finite_number(entry.get('value'), f'Scalar target for {kdma!r}')
        if not 0 <= value <= 1:
            raise ValueError(f'Scalar target for {kdma!r} must be between 0 and 1')
        kdmas.append({
            'kdma': kdma,
            'value': value,
            'definition': _definition_view(kdma, definitions),
        })
    return {
        'target_id': target_id,
        'target_scale': {'min': 0, 'max': 1},
        'target_semantics': 'medical_kdma_tradeoff',
        'kdmas': kdmas,
        'medical_context': _definition_view('medical', definitions),
    }
