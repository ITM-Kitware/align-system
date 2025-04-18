from rich.highlighter import JSONHighlighter
import pandas as pd

from align_system.utils import logging, call_with_coerced_args
from align_system.algorithms.abstracts import ADMComponent
from align_system.utils.alignment_utils import attributes_in_alignment_target

log = logging.getLogger(__name__)
JSON_HIGHLIGHTER = JSONHighlighter()


class KaleidoADMComponent(ADMComponent):
    def __init__(self,
                 kaleido_instance,
                 estimator_fn,
                 kdma_descriptions_map,
                 prompt_template):
        self.kaleido_instance = kaleido_instance
        self.estimator_fn = estimator_fn
        self.kdma_descriptions_map = kdma_descriptions_map
        self.prompt_template = prompt_template

    def run_returns(self):
        return ('attribute_prediction_reasonings',
                'relevance_prediction_scores',
                'attribute_prediction_scores')

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

        partial_template = call_with_coerced_args(
            self.prompt_template,
            {'scenario_state': scenario_state},
            partial=True)

        rows = []
        for choice in choices:
            choice_prompt = call_with_coerced_args(
                partial_template,
                {'choice': choice,
                 'other_choices': [c for c in choices if c != choice]})

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
