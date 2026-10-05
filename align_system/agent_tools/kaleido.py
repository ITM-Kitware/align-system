"""Kaleido value-assessment toolkit for the LangChain agent drivers.

Exposes a single `assess_available_actions` tool that scores each
currently available environment action against the attributes in the
scenario's alignment target using the Kaleido value model
(``align_system.algorithms.lib.kaleido.KaleidoSys``): how relevant the
attribute is to the action, whether the action supports or opposes it,
and an estimated attribute value to compare against the target.

Deliberately self-contained: this only depends on the low-level
Kaleido model wrapper, not on the ADM pipeline components or prompt
templates, so it can be attached to (or dropped from) an agent driver
without touching either.
"""
import json
from collections.abc import Mapping

from langchain_core.tools import tool

from align_system.agent_tools.base import AgentToolProvider
from align_system.drivers.itm_open_world_langchain import (
    TOOL_NAMES_BY_ACTION_TYPE)
from align_system.prompt_engineering.ow_prompts import (
    KaleidoAutoAssessHintPrompt, KaleidoToolHintPrompt)
from align_system.utils import logging


log = logging.getLogger(__name__)


def _target_values(alignment_target):
    """{attribute: target value} from an alignment target (swagger
    model, dict, or OmegaConf DictConfig); attributes without a scalar
    value are skipped."""
    if alignment_target is None:
        return {}

    if isinstance(alignment_target, Mapping):
        kdma_values = alignment_target.get('kdma_values') or []
    else:
        kdma_values = getattr(alignment_target, 'kdma_values', None) or []

    targets = {}
    for entry in kdma_values:
        if hasattr(entry, 'to_dict'):
            entry = entry.to_dict()
        if not isinstance(entry, Mapping):
            entry = vars(entry)
        value = entry.get('value')
        if entry.get('kdma') is not None and value is not None:
            targets[entry['kdma']] = float(value)

    return targets


def _choice_text(scenario_state, choice, other_choices):
    """The text Kaleido scores: the chosen action framed against the
    alternatives, in the context of the scene description."""
    if other_choices:
        alternatives = ' instead of "{}"'.format('", "'.join(other_choices))
    else:
        alternatives = ''
    return (f'Choosing to "{choice}"{alternatives}.  In the context of '
            f'the following scenario: {scenario_state.unstructured}')


class KaleidoToolkit(AgentToolProvider):
    """Agent toolkit wrapping a Kaleido model (see module docstring).

    `attribute_descriptions` maps attribute names as they appear in
    alignment targets to a dict with the `vrd` (Value / Right / Duty)
    and `description` Kaleido scores the attribute as; attributes in
    the target without a description are reported as unscored.

    `prompt_hint` is the text appended to the agent's system prompt (a
    string, or a prompt template object rendering to one when called
    with no arguments); when None it defaults to
    `KaleidoAutoAssessHintPrompt` or `KaleidoToolHintPrompt` (see
    `align_system.prompt_engineering.ow_prompts`) depending on
    `auto_assess`.
    """

    def __init__(self,
                 kaleido_instance,
                 attribute_descriptions,
                 auto_assess=True,
                 explain=False,
                 batch_size=None,
                 prompt_hint=None):
        # Model loading is deferred by KaleidoSys itself until the
        # first scoring call
        self.kaleido = kaleido_instance
        self.attribute_descriptions = {
            k: dict(v) for k, v in attribute_descriptions.items()}
        # Hand the agent an assessment before every decision (via the
        # driver's decision_context hook) rather than relying on it
        # to call the tool
        self.auto_assess = auto_assess
        self.explain = explain
        self.batch_size = batch_size

        if prompt_hint is None:
            prompt_hint = (KaleidoAutoAssessHintPrompt() if auto_assess
                           else KaleidoToolHintPrompt())
        self.prompt_hint = (prompt_hint() if callable(prompt_hint)
                            else prompt_hint)

        # Re-running the model for the same decision point would only
        # repeat the answer, so assessments are cached on the number of
        # actions taken so far; shared by the automatic assessment and
        # the explicit tool, and reset per scenario (see build_tools)
        self._cache = {}

    # -- Scoring --------------------------------------------------------

    def assess(self, scenario_state, actions, alignment_target):
        """Score `actions` against `alignment_target`.  Returns
        (targets, rows, unscored) where `targets` is {attribute:
        target value}, `rows` is a list of per-action dicts with a
        `scores` mapping of attribute -> {relevant, supports, opposes,
        either, estimate, distance[, explanation]}, and `unscored` is
        the list of target attributes with no description."""
        targets = _target_values(alignment_target)
        scored_attributes = [a for a in targets
                             if a in self.attribute_descriptions]
        unscored = [a for a in targets if a not in scored_attributes]

        choices = [a.unstructured for a in actions]

        # One Kaleido query per (action, attribute) pair, batched
        queries = []
        for action_idx, choice in enumerate(choices):
            text = _choice_text(
                scenario_state, choice,
                [c for i, c in enumerate(choices) if i != action_idx])
            for attribute in scored_attributes:
                desc = self.attribute_descriptions[attribute]
                queries.append((action_idx, attribute, text,
                                desc.get('vrd', 'Value'),
                                desc['description']))

        rows = [{'action': a,
                 'tool_name': TOOL_NAMES_BY_ACTION_TYPE.get(
                     a.action_type, a.action_type),
                 'description': a.unstructured,
                 'scores': {}}
                for a in actions]

        if queries:
            texts = [q[2] for q in queries]
            vrds = [q[3] for q in queries]
            descriptions = [q[4] for q in queries]

            relevance = self.kaleido.get_relevance(
                texts, vrds, descriptions, batch_size=self.batch_size)
            valence = self.kaleido.get_valence(
                texts, vrds, descriptions, batch_size=self.batch_size)
            explanations = (
                self.kaleido.get_explanation(
                    texts, vrds, descriptions, batch_size=self.batch_size)
                if self.explain else [None] * len(queries))

            for (action_idx, attribute, *_), rel, val, expl in zip(
                    queries, relevance, valence, explanations):
                relevant = float(rel[0])
                supports, opposes, either = (float(x) for x in val)
                # Estimated attribute value on the target's [0, 1]
                # scale: full support -> 1, full opposition -> 0,
                # "either" splits the difference
                estimate = supports + 0.5 * either
                score = {'relevant': relevant,
                         'supports': supports,
                         'opposes': opposes,
                         'either': either,
                         'estimate': estimate,
                         'distance': abs(estimate - targets[attribute])}
                if expl is not None:
                    score['explanation'] = expl
                rows[action_idx]['scores'][attribute] = score

        for row in rows:
            distances = [s['distance'] for s in row['scores'].values()]
            row['avg_distance'] = (sum(distances) / len(distances)
                                   if distances else None)

        return targets, rows, unscored

    @staticmethod
    def format_assessment(targets, rows, unscored):
        lines = ["Alignment target: " + ", ".join(
            f"{k}={v:.2f}" for k, v in targets.items())]
        if unscored:
            lines.append("(no value-model description for: "
                         f"{', '.join(unscored)}; not scored)")
        lines.append(
            f"Assessment of the {len(rows)} currently available "
            "action(s).  Estimates are on the target's [0, 1] scale; "
            "smaller distance to the target is better.")

        for row in rows:
            lines.append(f"- {row['tool_name']}: {row['description']}")
            for attribute, s in row['scores'].items():
                lines.append(
                    f"    {attribute}: relevance {s['relevant']:.2f}; "
                    f"supports {s['supports']:.2f} / opposes "
                    f"{s['opposes']:.2f} / either {s['either']:.2f}; "
                    f"estimate {s['estimate']:.2f} vs target "
                    f"{targets[attribute]:.2f} "
                    f"(distance {s['distance']:.2f})")
                if s.get('explanation'):
                    lines.append(f"      {s['explanation']}")

        ranked = [r for r in rows if r['avg_distance'] is not None]
        if ranked:
            best = min(ranked, key=lambda r: r['avg_distance'])
            lines.append(
                f"Closest to target overall: {best['tool_name']}: "
                f"{best['description']} (average distance "
                f"{best['avg_distance']:.2f})")

        return "\n".join(lines)

    # -- Per-decision assessment ----------------------------------------

    def assessment_text(self, session, actions=None):
        """The formatted assessment for the session's current decision
        point (cached per decision point), or an explanation of why
        there is nothing to assess.  `actions` are the currently
        available actions when the caller has already fetched them;
        otherwise they are (re-)fetched from the environment."""
        if session.alignment_target is None:
            return ("No alignment target is configured for this "
                    "scenario, so there is nothing to assess against; "
                    "decide on clinical grounds instead.")

        if not _target_values(session.alignment_target):
            return ("The alignment target has no attribute values to "
                    "assess against; decide on clinical grounds instead.")

        if actions is None:
            actions = session.refresh_actions()
        if not actions:
            return "No actions are currently available to assess."

        cache_key = session.n_actions
        if cache_key in self._cache:
            log.debug("Returning cached Kaleido assessment")
            return self._cache[cache_key]

        log.info("[bold]*ASSESSING ACTIONS (KALEIDO)*[/bold]",
                 extra={"markup": True})
        targets, rows, unscored = self.assess(
            session.current_state, actions, session.alignment_target)

        text = self.format_assessment(targets, rows, unscored)
        log.debug(json.dumps(
            [{k: v for k, v in r.items() if k != 'action'}
             for r in rows],
            indent=2, default=str))

        self._cache[cache_key] = text
        return text

    def decision_context(self, session):
        """Driver hook: the assessment handed to the agent before each
        decision when `auto_assess` is on.  Skipped when there is no
        choice to make (fewer than two available actions)."""
        if not self.auto_assess:
            return None

        # One fetch from the environment per decision point, shared
        # with the assessment below
        actions = session.refresh_actions()
        if len(actions) < 2:
            log.info(f"Kaleido assessment skipped: {len(actions)} "
                     "available action(s), nothing to choose between")
            return None

        return ("Value assessment of the currently available actions "
                "for your next decision:\n"
                + self.assessment_text(session, actions))

    # -- Tools ----------------------------------------------------------

    def build_tools(self, session):
        # Called once per scenario; cached assessments belong to the
        # previous scenario's decision points
        self._cache = {}
        toolkit = self

        @tool
        def assess_available_actions(justification: str) -> str:
            """Score every currently available action against the
            attributes of the alignment target using a value model:
            per action and attribute, the attribute's relevance,
            whether the action supports or opposes it, and an
            estimated attribute value next to the target value.  Slow;
            call at most once per decision, after
            list_available_actions.

            Args:
                justification: brief reason for requesting an assessment
            """
            log.info("[bold]*AGENT REQUESTED ACTION ASSESSMENT "
                     "(KALEIDO)*[/bold]", extra={"markup": True})
            text = toolkit.assessment_text(session)
            log.info(text)
            return text

        return [assess_available_actions]
