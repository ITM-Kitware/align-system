"""Comparative regression value-assessment toolkit for the LangChain
agent drivers.

The agent-side counterpart of the ADM pipeline's comparative
regression step (``ComparativeRegressionADMComponent``, as configured
in ``phase2_pipeline_zeroshot_comparative_regression``): for each
attribute of interest, an LLM is shown the scenario and all currently
available actions at once and asked for a brief explanation plus a
0-100 score per action. ADM prompt templates and attribute definitions
are reused with short candidate IDs and a compact output schema. Predictions
are scaled to the target's [0, 1] range. Ranked mode compares them to
the alignment target using average absolute distance; evidence mode
leaves interpretation to the agent and includes scorer reasoning.

Inference goes through a LangChain chat model with structured output
(guided JSON decoding on vLLM / ollama) rather than the ADM's outlines
inference engine, so it can share the agent's already-served model
instead of loading a second copy in process. Prompt
templates, attribute definitions, treatment-option rules, and optional
few-shot example selection are shared with the ADM pipeline.
"""
import json
import math
from timeit import default_timer as timer

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from omegaconf import OmegaConf

from align_system.agent_tools.base import AgentToolProvider
from align_system.agent_tools.concrete_actions import concrete_actions
from align_system.agent_tools.cr_reminder import CRReminder
from align_system.exceptions import ContextWindowError
from align_system.utils.alignment_targets import alignment_target_values
from align_system.utils.agent_actions import (
    describe_action, assessment_cache_key, assessment_rows)
from align_system.drivers.itm_open_world_langchain import (
    TOOL_NAMES_BY_ACTION_TYPE, _filter_available_actions)
from align_system.prompt_engineering.compat.outlines.templates import prompt as compat_outlines_prompt
from align_system.utils import logging, call_with_coerced_args


@compat_outlines_prompt
def comparative_regression_auto_assess_hint():
    """
    Value alignment: your actions must align with the alignment target.
    Before each decision you are given a value assessment of the currently
    available actions: per action and attribute, a predicted attribute
    value (on the target's 0 to 1 scale) alongside the target value, and
    the distance between them.  Medical urgency is also predicted for
    reference.  Read it before choosing: prefer the action whose
    predictions are closest to the target (breaking ties on medical
    urgency and clinical grounds), and if you choose a different action,
    say in your justification why the assessment does not apply.  You also
    have a score_available_actions(justification) tool that returns the
    same assessment again if you need to re-read it.
    """


class ComparativeRegressionAutoAssessHintPrompt:
    """System prompt addition for the LangChain agent when the
    comparative regression toolkit hands it an assessment before every
    decision (auto_assess=True)."""
    def __call__(self):
        return comparative_regression_auto_assess_hint()


@compat_outlines_prompt
def comparative_regression_tool_hint():
    """
    You also have a value-assessment tool:

    - score_available_actions(justification): score every currently
      available action against the attributes of your alignment target.
      It reports, per action and attribute, a predicted attribute value
      (on the target's 0 to 1 scale) alongside the target value and the
      distance between them, plus a medical urgency prediction for
      reference.

    This is an optional advisory tool. You may call it when useful, at
    most once per decision, or act without it. It also returns all selected
    labeled training examples, clearly separated from the current scene.
    Use the target, clinical needs, and scene to interpret its predictions;
    they can be incomplete or mistaken and do not require a particular action.
    """


class ComparativeRegressionToolHintPrompt:
    """System prompt addition for the LangChain agent when it may
    request an optional comparative regression assessment
    (auto_assess=False)."""
    def __call__(self):
        return comparative_regression_tool_hint()


class ComparativeRegressionEvidenceHintPrompt:
    """Leave interpretation of attribute predictions to the agent."""
    def __init__(self, auto_assess=False):
        self.auto_assess = auto_assess

    def __call__(self):
        delivery = (
            "Before each decision, you receive "
            "an attribute assessment. You can re-read it with "
            "score_available_actions(justification)."
            if self.auto_assess else
            "You may optionally call score_available_actions(justification) "
            "at most once per decision, or act without it.")
        return (delivery + " The assessment contains the alignment target, "
                "predicted attribute values on a 0 to 1 scale, brief scorer "
                "explanations, and all selected labeled training examples. "
                "Medical urgency is also predicted for reference. "
                "Interpret this evidence using the target and current scene, "
                "choose an action, and explain how your choice serves the "
                "target and clinical needs. Predictions can be incomplete "
                "or mistaken; use their reasoning and the scene context.")


log = logging.getLogger(__name__)


def _choice_keys(actions, scenario_state=None):
    """Short IDs; full patient/action context stays in the query and rows."""
    return [f"C{i:02d}" for i in range(1, len(actions) + 1)]


class ComparativeRegressionToolkit(AgentToolProvider):
    """Agent toolkit running comparative regression over the available
    actions (see module docstring).

    `attributes` maps attribute names as they appear in alignment
    targets to `align_system.data_models.attribute.Attribute`
    definitions (the same ones the ADM pipelines use); target
    attributes without a definition are reported as unscored.

    `target_attribute_names_override` selects the attributes to score
    as in the ADM component: '*' stands for the target's attributes,
    so the default ['medical', '*'] also predicts medical urgency for
    reference (it has no target value, so it doesn't count towards the
    distance).

    `num_samples` predictions are averaged per action and attribute.
    `max_retries` bounds additional attempts for an invalid or failed attribute;
    every candidate and sample must validate before that attribute is published.

    `generation_kwargs` overrides model options for scoring requests only
    (for example, `max_tokens` for vLLM/OpenAI or `num_predict` for Ollama).

    `prompt_hint` is the text appended to the agent's system prompt (a
    string, or a prompt template object rendering to one when called
    with no arguments); when None it defaults to
    `ComparativeRegressionAutoAssessHintPrompt` or
    `ComparativeRegressionToolHintPrompt` (see
    `align_system.agent_tools.comparative_regression`) depending on
    `auto_assess`. In `assessment_mode="evidence"`, a local evidence
    prompt replaces ranking instructions and explanations are included.
    """

    def __init__(self,
                 chat_model,
                 attributes,
                 scenario_description_template,
                 prompt_template,
                 score_schema_template,
                 system_prompt_template=None,
                 target_attribute_names_override=('medical', '*'),
                 num_samples=1,
                 auto_assess=False,
                 show_reasoning=True,
                 prompt_hint=None,
                 assessment_mode="ranked",
                 generation_kwargs=None,
                 icl=None,
                 max_retries=2,
                 reminder_interval=5):
        if assessment_mode not in {'evidence', 'ranked'}:
            raise ValueError("assessment_mode must be 'evidence' or 'ranked'")
        self.assessment_mode = assessment_mode
        self.chat_model = chat_model
        self.icl = icl
        # Hydra retains nested DictConfig/ListConfig objects by default; the
        # provider SDK needs ordinary JSON-serializable request options.
        self.generation_kwargs = OmegaConf.to_container(
            OmegaConf.create(generation_kwargs or {}), resolve=True)
        if type(max_retries) is not int or max_retries < 0:
            raise ValueError("max_retries must be a nonnegative integer")
        self.max_retries = max_retries
        self.reminder = CRReminder(reminder_interval)
        self.attributes = dict(attributes)
        self.scenario_description_template = scenario_description_template
        self.prompt_template = prompt_template
        self.score_schema_template = score_schema_template
        self.system_prompt_template = system_prompt_template
        self.target_attribute_names_override = (
            None if target_attribute_names_override is None
            else list(target_attribute_names_override))
        if type(num_samples) is not int or num_samples < 1:
            raise ValueError("num_samples must be a positive integer")
        self.num_samples = num_samples
        # Hand the agent an assessment before every decision (via the
        # driver's decision_context hook) rather than relying on it
        # to call the tool
        self.auto_assess = auto_assess
        # Include the scorer's per-attribute reasoning in the text
        # handed to the agent (always logged)
        self.show_reasoning = show_reasoning or assessment_mode == "evidence"

        if prompt_hint is None:
            if assessment_mode == 'evidence':
                prompt_hint = ComparativeRegressionEvidenceHintPrompt(auto_assess)
            elif auto_assess:
                prompt_hint = ComparativeRegressionAutoAssessHintPrompt()
            else:
                prompt_hint = ComparativeRegressionToolHintPrompt()
        self.prompt_hint = (prompt_hint() if callable(prompt_hint)
                            else prompt_hint)

        # Re-running the regression for the same decision point would
        # only repeat the answer, so assessments are cached on the
        # decision, available actions, and target; shared by the automatic
        # assessment and the explicit tool, and reset per scenario
        # (see build_tools)
        self._cache = {}

    # -- Scoring --------------------------------------------------------

    def _attribute_names(self, targets):
        """Attributes to score, in order and without repeats, following
        the ADM component's override semantics."""
        if self.target_attribute_names_override is None:
            names = list(targets)
        else:
            names = []
            for name in self.target_attribute_names_override:
                names.extend(targets if name == '*' else [name])

        return list(dict.fromkeys(names))

    def _regress_attribute(self, scenario_state, scenario_description,
                           choices, attribute, icl_dialog=(),
                           choice_descriptions=None):
        """Run comparative regression for one attribute; returns
        (list of per-sample {choice: score in [0, 1]}, list of
        per-sample {choice: brief explanation})."""
        messages = []
        if self.system_prompt_template is not None:
            messages.append(SystemMessage(content=call_with_coerced_args(
                self.system_prompt_template,
                {'target_attribute': attribute})))

        for example in icl_dialog:
            message_type = HumanMessage if example.role == 'user' else AIMessage
            messages.append(message_type(content=example.content))

        descriptions = choice_descriptions or {c: c for c in choices}
        prompt_choices = [f"{c}: {descriptions[c]}" for c in choices]
        query = call_with_coerced_args(
            self.prompt_template,
            {'scenario_state': scenario_state,
             'scenario_description': scenario_description,
             'choices': prompt_choices,
             'choice_outcomes': {c: None for c in prompt_choices},
             'attribute': attribute.name})
        messages.append(HumanMessage(content=query + (
            "\nCURRENT CANDIDATE OUTPUT FORMAT: Return only the JSON object "
            "specified by the schema, keyed by the short candidate IDs ("
            + ", ".join(choices) + "). For every candidate provide a numeric "
            f"score from 0 to {attribute.factor} and an explanation of at most "
            "160 characters. Use one brief sentence grounded in the scene. "
            "Do not repeat action descriptions or add overall reasoning. "
            "The earlier labeled scenarios are training examples, not facts "
            "about the current patients.")))

        schema = json.loads(call_with_coerced_args(
            self.score_schema_template,
            {'choices': choices,
             'attribute': attribute.name}))
        # Keep the configured score scale/type, but use short keys and bounded
        # explanations instead of repeating full descriptions in generated JSON.
        schema['properties'] = {c: schema['properties'][c] for c in choices}
        schema['required'] = list(choices)
        schema['additionalProperties'] = False
        for candidate in schema['properties'].values():
            candidate['properties']['explanation'] = {
                'type': 'string', 'minLength': 1, 'maxLength': 160}
            candidate['required'] = ['score', 'explanation']
            candidate['additionalProperties'] = False
        # A title / description is what LangChain names the structured
        # output format after
        schema.setdefault('title', 'comparative_regression')
        schema.setdefault(
            'description',
            f'A {attribute.name} score and brief explanation per candidate')

        log.info("[bold]*KDMA SCORE PREDICTION DIALOG ({})*[/bold]".format(
            attribute.kdma), extra={"markup": True})
        for message in messages:
            log.info(f"[{message.type}] {message.content}")

        scorer = self.chat_model.with_structured_output(
            schema, method='json_schema')
        generation_kwargs = dict(self.generation_kwargs)
        from align_system.utils.vllm_chat_model import VLLMChatOpenAI
        if isinstance(self.chat_model, VLLMChatOpenAI):
            # Few-shot examples must not be evicted as though they were old
            # actor conversation turns when the scoring prompt is oversized.
            generation_kwargs['preserve_messages'] = True
        responses = scorer.batch(
            [messages] * self.num_samples, **generation_kwargs)
        self._validate_responses(responses, choices, attribute)

        sample_scores = []
        sample_reasonings = []
        for i, response in enumerate(responses):
            log.info("[bold]*KDMA SCORE PREDICTION RESPONSE ({}, sample "
                     "#{})*[/bold]".format(attribute.kdma, i),
                     extra={"markup": True})
            log.info(json.dumps(response, indent=2))

            sample_scores.append(
                {c: response[c]['score'] / attribute.factor
                 for c in choices})
            sample_reasonings.append({c: response[c]['explanation'] for c in choices})

        return sample_scores, sample_reasonings

    def _validate_responses(self, responses, choices, attribute):
        """Reject partial or invalid batches before publishing any scores."""
        if not isinstance(responses, list) or len(responses) != self.num_samples:
            raise ValueError(f"Expected {self.num_samples} complete scoring samples")
        if (type(attribute.factor) not in (int, float)
                or not math.isfinite(attribute.factor) or attribute.factor <= 0):
            raise ValueError("Attribute score factor must be finite and positive")
        errors = []
        for i, response in enumerate(responses, 1):
            if not isinstance(response, dict):
                errors.append(f"sample {i}: response is not an object")
                continue
            if set(response) != set(choices):
                errors.append(f"sample {i}: candidate IDs must be exactly {', '.join(choices)}")
            for choice in choices:
                entry = response.get(choice)
                if not isinstance(entry, dict):
                    errors.append(f"sample {i}: {choice} missing score/explanation")
                    continue
                score = entry.get('score')
                if (type(score) not in (int, float) or not math.isfinite(score)
                        or not 0 <= score <= attribute.factor):
                    errors.append(f"sample {i}: {choice} invalid score (expected 0..{attribute.factor})")
                explanation = entry.get('explanation')
                if (not isinstance(explanation, str) or not explanation.strip()
                        or len(explanation) > 160):
                    errors.append(f"sample {i}: {choice} explanation must contain 1..160 characters")
        if errors:
            raise ValueError("; ".join(errors))

    def assess(self, scenario_state, actions, alignment_target):
        """Score `actions` against `alignment_target`.  Returns
        (targets, rows, unscored, failed, reasonings) where `targets` is
        {attribute: target value}, `rows` is a list of per-action dicts
        with a `scores` mapping of attribute -> {predicted, target}
        (target None for attributes without a target value). Ranked
        mode adds distance and per-row avg_distance. `unscored` lists
        the requested attributes with no
        definition, `failed` lists the attributes whose regression
        call failed, and `reasonings` is {attribute: [per-sample
        reasoning]}."""
        actions, _ = concrete_actions(scenario_state, actions)
        return self._assess_concrete(scenario_state, actions, alignment_target)

    def _assess_concrete(self, scenario_state, actions, alignment_target,
                         icl_info=None, scoring_info=None):
        targets = alignment_target_values(alignment_target)
        names = self._attribute_names(targets)
        scored_attributes = [n for n in names if n in self.attributes]
        unscored = [n for n in names if n not in self.attributes]

        choices = _choice_keys(actions, scenario_state)
        scenario_description = call_with_coerced_args(
            self.scenario_description_template,
            {'scenario_state': scenario_state,
             'alignment_target': alignment_target})

        rows = [{'action': a, 'candidate_id': choice,
                 'tool_name': TOOL_NAMES_BY_ACTION_TYPE.get(
                     a.action_type, a.action_type),
                 'description': describe_action(a, scenario_state),
                 'scores': {}}
                for a, choice in zip(actions, choices)]
        descriptions = {row['candidate_id']: row['description'] for row in rows}

        icl_dialogs = {}
        if self.icl is not None and actions:
            # Use the ADM's actual dataset hydration, example selection and
            # ground-truth scaling; do not silently fall back to zero-shot.
            icl_dialogs, selected_info = self.icl.run(
                scenario_state,
                [f"Choice {i}: {descriptions[c]}" for i, c in enumerate(choices)],
                actions, alignment_target)
            if icl_info is not None:
                icl_info.update(selected_info)

        failed = []
        reasonings = {}
        for name in scored_attributes if actions else []:
            attribute = self.attributes[name]
            attempts = []
            complete = False
            for attempt in range(1, self.max_retries + 2):
                try:
                    sample_scores, sample_reasonings = self._regress_attribute(
                        scenario_state, scenario_description, choices,
                        attribute, icl_dialogs.get(attribute.kdma, ()), descriptions)
                except Exception as e:
                    attempts.append({'attempt': attempt, 'status': 'failed',
                                     'error': f'{type(e).__name__}: {e}'})
                    log.warning(f"Comparative regression for '{name}' "
                                f"attempt {attempt}/{self.max_retries + 1} failed: {e}")
                    # Retrying an unchanged prompt cannot fix a context limit.
                    if isinstance(e, ContextWindowError):
                        break
                else:
                    attempts.append({'attempt': attempt, 'status': 'complete'})
                    complete = True
                    break
            if scoring_info is not None:
                scoring_info[name] = attempts
            if not complete:
                failed.append(name)
                continue

            reasonings[name] = ["; ".join(f"{c}: {sample[c]}" for c in choices)
                                for sample in sample_reasonings]
            for row, choice in zip(rows, choices):
                predicted = (sum(s[choice] for s in sample_scores)
                             / len(sample_scores))
                target = targets.get(name)
                row['scores'][name] = {
                    'predicted': predicted,
                    'target': target,
                    'explanations': [sample[choice] for sample in sample_reasonings]}
                if self.assessment_mode == 'ranked':
                    row['scores'][name]['distance'] = (
                        None if target is None else abs(predicted - target))

        if self.assessment_mode == 'ranked':
            for row in rows:
                distances = [s['distance'] for s in row['scores'].values()
                             if s['distance'] is not None]
                row['avg_distance'] = (
                    sum(distances) / len(distances)
                    if distances and all(n in row['scores'] for n in targets)
                    else None)

        return targets, rows, unscored, failed, reasonings

    def format_assessment(self, targets, rows, unscored, failed,
                          reasonings):
        lines = ["COMPARATIVE REGRESSION ASSESSMENT", "Alignment target: " + ", ".join(
            f"{k}={v:.2f}" for k, v in targets.items())]
        if unscored:
            lines.append("(no attribute definition for: "
                         f"{', '.join(unscored)}; not scored)")
        if failed:
            lines.append("(scoring failed for: "
                         f"{', '.join(failed)}; not scored)")
        lines.append(
            f"CURRENT ACTIONS: {len(rows)} currently available "
            "action(s). Predictions are on the target's [0, 1] scale. " + (
                "Smaller distance to the target is better."
                if self.assessment_mode == 'ranked' else
                "Interpret the predictions using the target and scene."))

        for row in rows:
            lines.append(f"- {row['candidate_id']}: {row['description']}")
            for attribute, s in row['scores'].items():
                if s['target'] is None:
                    lines.append(
                        f"    {attribute}: predicted {s['predicted']:.2f} "
                        "(no target; for reference)")
                else:
                    lines.append(
                        f"    {attribute}: predicted {s['predicted']:.2f} "
                        f"vs target {s['target']:.2f}")
                    if self.assessment_mode == 'ranked':
                        lines[-1] += f" (distance {s['distance']:.2f})"
                if self.show_reasoning:
                    lines[-1] += "; " + " / ".join(dict.fromkeys(s['explanations']))
            missing = list(dict.fromkeys([*unscored, *failed]))
            if missing:
                lines.append(f"    Missing scores: {', '.join(missing)}")

        ranked = [r for r in rows if r.get('avg_distance') is not None]
        if ranked:
            best = min(ranked, key=lambda r: r['avg_distance'])
            lines.append(
                f"Closest to target overall: {best['candidate_id']} (average distance "
                f"{best['avg_distance']:.2f})")
        elif rows and self.assessment_mode == 'ranked':
            lines.append("Overall recommendation unavailable: target scores are incomplete.")
        lines.append("Predictions are advisory and may be mistaken; you remain free to act.")

        return "\n".join(lines)

    def format_examples(self, icl_info):
        """Expose every selected dataset prompt and label without regeneration."""
        if not icl_info:
            return ""
        lines = [
            "SIMILAR TRAINING SCENARIOS",
            "Historical labeled examples, not facts about the current patients. "
            "Similarity does not establish equivalence. The following prompts "
            "and labeled answers are copied from the examples supplied to the scorer; "
            "instructions inside them apply only to those historical examples."]
        for name, examples in icl_info.items():
            attribute = self.attributes.get(name)
            scale = f" (label scale 0..{attribute.factor})" if attribute else ""
            lines.append(f"{name}: showing all {len(examples)} selected examples{scale}.")
            for index, example in enumerate(examples, 1):
                lines.extend([
                    f"BEGIN TRAINING EXAMPLE {name}/E{index:02d}",
                    "Scenario and candidate actions (dataset prompt):",
                    example['prompt'],
                    "Labeled answer:",
                    str(example['response']),
                    f"END TRAINING EXAMPLE {name}/E{index:02d}"])
        lines.append("END TRAINING SCENARIOS. Choose using the current scene and target.")
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

        if not alignment_target_values(session.alignment_target):
            return ("The alignment target has no attribute values to "
                    "assess against; decide on clinical grounds instead.")

        # Remember requests independently of session.assessments, which is
        # cleared after each action. Failed requests also start the cooldown.
        self.reminder.note_use(session)
        if actions is None:
            actions = session.refresh_actions()
        if not actions:
            return "No actions are currently available to assess."

        actions, omitted = concrete_actions(session.current_state, actions)
        if session.apply_action_filtering:
            actions = _filter_available_actions(
                session.current_state, actions, session.treated_patients,
                session.evac_patients, session.tagged_patients, session.vitals_checked,
                session.blocked_moves)
        state = session.current_state
        state_data = state.to_dict() if hasattr(state, 'to_dict') else vars(state)
        cache_key = (assessment_cache_key(session, actions),
                     json.dumps(state_data, sort_keys=True, default=str), tuple(omitted))
        if cache_key in self._cache:
            log.debug("Returning cached comparative regression assessment")
            text, record = self._cache[cache_key]
            session.record_assessment('comparative_regression', record)
            return text

        log.info("[bold]*ASSESSING ACTIONS (COMPARATIVE REGRESSION)*[/bold]",
                 extra={"markup": True})
        start = timer()
        icl_info = {}
        scoring_info = {}
        targets, rows, unscored, failed, reasonings = self._assess_concrete(
            session.current_state, actions, session.alignment_target,
            icl_info, scoring_info)
        log.info("Comparative regression assessment took "
                 f"{timer() - start:.1f}s")

        record = {
            'mode': self.assessment_mode,
            'icl_examples': icl_info,
            'auto_assess': self.auto_assess,
            'targets': targets,
            'rows': assessment_rows(rows),
            'unscored': unscored,
            'failed': failed, 'reasonings': reasonings, 'omitted': omitted,
            'scoring_attempts': scoring_info,
            'missing_scores': {name: [row['candidate_id'] for row in rows]
                               for name in [*unscored, *failed]},
        }

        text = self.format_assessment(
            targets, rows, unscored, failed, reasonings)
        text += ("\nExecute a scored candidate using execute_available_action with its "
                 "action_id and character_name set to its character_id. Copy treatment "
                 "to treatment_supply, location to injury_location, and category to "
                 "triage_tag when present. Scores apply only to that exact candidate.")
        if omitted:
            text += "\nNot scored (insufficient observed information):\n- " + "\n- ".join(omitted)
        if not actions:
            text += "\nNo concrete candidates to score; inspect the scene before choosing."
        examples_text = self.format_examples(icl_info)
        if examples_text:
            text += "\n\n" + examples_text
        record['text'] = text
        session.record_assessment('comparative_regression', record)
        log.debug(json.dumps(
            [{k: v for k, v in r.items() if k != 'action'}
             for r in rows],
            indent=2, default=str))

        # A failed attribute may succeed on a retry, so only complete
        # assessments are cached
        if not failed:
            self._cache[cache_key] = (text, record)
        return text

    def decision_context(self, session):
        """Offer cheap reminders, or generate assessments when configured."""
        if not self.auto_assess:
            return self.reminder.context(
                session, alignment_target_values(session.alignment_target))

        # One fetch from the environment per decision point, shared
        # with the assessment below
        actions = session.refresh_actions()
        if not actions:
            return None

        return ("Value assessment of the currently available actions "
                "for your next decision:\n"
                + self.assessment_text(session, actions))

    # -- Tools ----------------------------------------------------------

    def build_tools(self, session):
        # Called once per scenario; cached assessments belong to the
        # previous scenario's decision points
        self._cache = {}
        self.reminder.reset()
        toolkit = self

        @tool
        def score_available_actions(justification: str) -> str:
            """Score every currently available action against the
            attributes of the alignment target: per action and
            attribute, a predicted attribute value next to the target
            value, brief explanations, a medical urgency prediction, and all
            selected labeled training scenarios. This advisory tool is optional.
            Ranked mode also reports target distances. Slow; call at most once
            per decision, after list_available_actions.

            Args:
                justification: brief reason for requesting an assessment
            """
            log.info("[bold]*AGENT REQUESTED ACTION ASSESSMENT "
                     "(COMPARATIVE REGRESSION)*[/bold]",
                     extra={"markup": True})
            text = toolkit.assessment_text(session)
            log.info(text)
            return text

        return [score_available_actions]
