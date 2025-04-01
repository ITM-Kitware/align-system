from rich.highlighter import JSONHighlighter
import pandas as pd

from align_system.utils import logging
from align_system.algorithms.abstracts import ADMComponent
from align_system.algorithms.lib.util import format_template
from align_system.utils.alignment_utils import attributes_in_alignment_target

log = logging.getLogger(__name__)
JSON_HIGHLIGHTER = JSONHighlighter()


class KaleidoADMComponent(ADMComponent):
    def __init__(self,
                 kaleido_instance,
                 estimator_fn,
                 kdma_descriptions_map,
                 prompt_template='What action do you take? "{{choice}}" instead of {{other_choices}}.  In the context of the following scenario: {{scene_state}} {{mission}} {{decision_environment}} With the following casualties:\n{{characters_str}}'):
        self.kaleido_instance = kaleido_instance

        self.estimator_fn = estimator_fn

        self.kdma_descriptions_map = kdma_descriptions_map

        self.prompt_template = prompt_template

    # TODO: Move to separate class (alongside the other ADM component
    # prompt templates)
    def _build_partial_template(self, scenario_state):
        if scenario_state.mission is None:
            mission_str = ''
        else:
            mission_str = scenario_state.mission.unstructured

        characters_str = '\n'.join(
            ['{} ({}): {}'.format(c.name, c.id, c.unstructured)
             for c in scenario_state.characters])

        partial_template = format_template(
            self.prompt_template,
            partial=True,
            scene_state=scenario_state.unstructured,
            mission=mission_str,
            decision_environment=scenario_state.environment.decision_environment.unstructured.strip(),
            characters_str=characters_str)

        return partial_template

    def run_returns(self):
        return ('kdma_prediction_reasonings',
                'relevance_prediction_scores',
                'kdma_prediction_scores')

    def run(self,
            scenario_state,
            choices,
            alignment_target=None):
        """
        Returns:
            kdma_prediction_reasonings
            relevance_prediction_scores
            kdma_prediction_scores
        """
        if alignment_target is None:
            target_attribute_names = []
        else:
            target_attribute_names = attributes_in_alignment_target(alignment_target)

        partial_template = self._build_partial_template(scenario_state)

        rows = []
        for choice in choices:
            other_choices_str = ', '.join(
                ['"{}"'.format(c) for c in choices if c != choice])
            choice_prompt = format_template(
                partial_template,
                allow_extraneous=True,
                choice=choice, other_choices=other_choices_str)

            log.info("[bold] ** Kaleido Prompt ** [/bold]",
                     extra={"markup": True})
            log.info(choice_prompt)

            for attribute in target_attribute_names:
                mapped_kdma = self.kdma_descriptions_map[attribute]

                vrd = mapped_kdma.get('vrd', 'Value')
                description = mapped_kdma['description']

                relevance = self.kaleido_instance.get_relevance(choice_prompt, vrd, description)
                valence = self.kaleido_instance.get_valence(choice_prompt, vrd, description)

                # relevant, not_relevant = relevance
                # supports, opposes, either = valence

                explanation = self.kaleido_instance.get_explanation(
                    choice_prompt, vrd, description)

                rows.append((choice,
                             vrd,
                             attribute,
                             description,
                             *relevance,
                             *valence,
                             explanation))

        results = pd.DataFrame(
            rows, columns=["choice", "VRD", "KDMA", "kdma_description", "relevant", "not_relevant", "supports", "opposes", "either", "explanation"])

        results['estimated_kdma_value'] = self.estimator_fn(results)

        log.explain("[bold] ** Kaleido Relevance / Valence and Estimated "
                    "KDMA Values ** [/bold]",
                    extra={"markup": True})
        log.debug(results)

        display_results = results.copy()
        display_results[['relevant', 'supports', 'opposes', 'either', 'estimated_kdma_value']] =\
            display_results[['relevant', 'supports', 'opposes', 'either', 'estimated_kdma_value']].map(lambda x: f"{float(x):.2f}")
        log.explain(display_results[['choice', 'VRD', 'KDMA', 'kdma_description', 'relevant', 'supports', 'opposes', 'either', 'estimated_kdma_value']])

        reasonings = {}
        relevances = {}
        scores = {}
        for group_key, group_records in results.groupby(['choice', 'KDMA']):
            # group_key is a single element tuple in this case
            choice, kdma = group_key

            reasonings.setdefault(choice, {})
            relevances.setdefault(choice, {})
            scores.setdefault(choice, {})

            reasonings[choice].setdefault(kdma, []).append(
                str(group_records['explanation'].iloc[0]))

            # alignment functions expecting values to be 0-1 (rather than 0-10)
            scores[choice].setdefault(kdma, []).extend(
                [float(v) for v in (group_records['estimated_kdma_value'] / 10)])

            relevances[choice].setdefault(kdma, []).append(
                float(group_records['relevant'].iloc[0]))

        return reasonings, relevances, scores
