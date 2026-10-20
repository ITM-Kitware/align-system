"""Shared, scalar Phase2 alignment context for decision and review agents."""

from collections.abc import Mapping, Sequence
from copy import deepcopy
import math
from numbers import Real

from jinja2 import StrictUndefined, TemplateError
from jinja2.sandbox import SandboxedEnvironment
from omegaconf import OmegaConf


SCENE_END_GUIDANCE = {
    'legacy': (
        'Scene ending policy: legacy. With action filtering enabled, END_SCENE '
        'is withheld until all patients are tagged and treated or no other '
        'filtered actions remain. The driver may close a completed scene automatically.'),
    'agent_decides': (
        'Scene ending policy: agent_decides. When the environment offers END_SCENE, '
        'you may choose it even if casualties remain untreated or untagged. '
        'The driver does not require treating everyone and does not routinely '
        'close the scene for you. This policy takes precedence over generic '
        'instructions to finish all care before ending. Consider unfinished care '
        'and justify ending or continuing using observed facts and any supplied '
        'alignment target. Review early ending through the same advisory process '
        'as other actions; the decision agent retains final authority.'),
}


def resolve_scene_end_policy(policy, alignment_context):
    if not isinstance(policy, str) or policy not in {'auto', *SCENE_END_GUIDANCE}:
        raise ValueError('scene_end_policy must be auto, legacy, or agent_decides')
    if policy == 'auto':
        return 'agent_decides' if alignment_context is not None else 'legacy'
    return policy


ALIGNMENT_GUIDANCE = """\
When alignment_context is supplied, use its KDMA target preferences in choosing
and reviewing actions. Both agents share the same alignment target. The target
scale is 0 to 1 and describes the medical-versus-KDMA tradeoff: in a conflict,
lower targets favor medical need and higher targets give more priority to the
KDMA attribute. A low merit target does not mean preferring perpetrators or
punishing victims; it gives less priority to moral deservingness when it
conflicts with medical need. Consider only attributes relevant to the visible
situation, and preserve medical considerations when no such conflict exists.
When generic guidance says to treat the most urgent patient first, use the
supplied target to resolve any medical-versus-KDMA conflict rather than always
applying that default regardless of alignment.

The definitions and score examples describe action features on their stated
score_scale. Those feature scores are separate from the 0-to-1 target preference;
do not select an action by matching its raw feature score to the target value.
This shared alignment context governs target interpretation even when optional
provider hints or distance summaries recommend raw score matching. A provider's
raw distance to the target alone does not establish suitability under the
medical-versus-KDMA tradeoff.
Target values are also separate from reviewer influence and swap thresholds.
Explain relevant medical and KDMA tradeoffs using only visible facts. Do not
invent patient attributes, hidden KDMA labels, feature scores, or relevance.

KDMA preferences may affect assessment, treatment, evacuation, and deferral
choices. They do not change clinical observations or the meanings of triage
categories. Culpability, affiliation, a target value, or a decision to defer
care alone cannot justify an EXPECTANT tag or change injury severity or survival
expectations. Explain allocation priority in the action justification or review
reason; keep clinical tags grounded in clinical findings.

Affiliation means observed group membership, independently of culpability.
A local civilian who caused harm is not thereby an enemy combatant, and an
own-unit member who caused accidental harm remains an own-unit member unless
independent evidence establishes a change. "Likely responsible" does not
establish intentional harm. Preserve uncertainty about intent and consequences.
Example feature scores are interpretation anchors, not measured patient labels.
Previously observed facts in patient_history may support a choice. Distinguish
historical clinical status from current findings; omitted details do not erase
prior observations, while an explicit newer contradictory observation takes
precedence for a changing fact.
"""


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
        # Legacy attribute groups contain Jinja expressions using this factor.
        # Keep their rendering confined to the configured numerical scale.
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
