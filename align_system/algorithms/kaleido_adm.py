from abc import ABC, abstractmethod

import pandas as pd

from align_system.algorithms.kaleido import KaleidoSys
from align_system.algorithms.lib.aligned_decision_maker import AlignedDecisionMaker
from align_system.algorithms.lib.util import format_template
from align_system.utils import logging


log = logging.getLogger(__name__)

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)


class EstimateKDMAFunction(ABC):
    def __call__(self, results_df: pd.DataFrame) -> pd.DataFrame:
        ...


class SimpleKDMAEstimator(EstimateKDMAFunction):
    def __call__(self, results_df: pd.DataFrame) -> pd.DataFrame:
        return results_df['either'] * 5 + results_df['supports'] * 10


class ChoiceDistanceFunction(ABC):
    def __call__(self, group_records: pd.DataFrame) -> pd.DataFrame:
        ...


class RelevanceWeightedDistance(ChoiceDistanceFunction):
    def __call__(self, group_records: pd.DataFrame) -> pd.DataFrame:
        # (1.0 / relevant) as a weight could be too punitive?
        return sum((1.0 / group_records['relevant'])
                   * abs(group_records['weight'] - group_records['target']))


class MeanDistance(ChoiceDistanceFunction):
    def __call__(self, group_records: pd.DataFrame) -> pd.DataFrame:
        return (sum(group_records['relevant']
                    * abs(group_records['weight'] - group_records['target']))
                / len(group_records))


class MeanDistance2(ChoiceDistanceFunction):
    # Probably want to divide by the sum(group_records['relevant'])
    # instead of len(group_records) (as with the MeanDistance
    # function) so that we don't reward having only a few relevant
    # KDMAs (vs. having more relevant but stronger supporting KDMAs)
    def __call__(self, group_records: pd.DataFrame) -> pd.DataFrame:
        return (sum(group_records['relevant']
                    * abs(group_records['weight'] - group_records['target']))
                / sum(group_records['relevant']))


DefaultKDMAEstimatorFunction = SimpleKDMAEstimator
DefaultDistanceFunction = RelevanceWeightedDistance


class KaleidoADM(AlignedDecisionMaker):
    def __init__(self, **kwargs):
        self.kaleido = KaleidoSys(**kwargs)


    # def estimate_kdma_values(self,
    #                          prompt_template,
    #                          choices,
    #                          target_kdmas,
    #                          estimator_fn=DefaultKDMAEstimatorFunction,
    #                          kdma_descriptions_map=None):


    def predict_kdma_weights(self,
                             prompt_template,
                             choices,
                             target_kdmas,
                             compute_weight_fn=default_kdma_weights_fn,
                             distance_fn=default_distance_fn,
                             kdma_descriptions_map=None):

        if kdma_descriptions_map is None:
            kdma_descriptions_map = {k: {'description': k} for k in target_kdmas.keys()}

        rows = []
        choice_prompts = {}
        for choice in choices:
            other_choices_str = ', '.join(['"{}"'.format(c) for c in (set(choices) - {choice})])
            choice_prompt = format_template(
                prompt_template,
                allow_extraneous=True,
                choice=choice, other_choices=other_choices_str)
            choice_prompts[choice] = choice_prompt

            log.debug("[bold] ** Kaleido Prompt ** [/bold]",
                      extra={"markup": True})
            log.debug(choice_prompt)

            for kdma, target in target_kdmas.items():
                mapped_kdma = kdma_descriptions_map[kdma]

                vrd = mapped_kdma.get('vrd', 'Value')
                description = mapped_kdma['description']

                relevance = self.get_relevance(choice_prompt, vrd, description)
                valence = self.get_valence(choice_prompt, vrd, description)

                # relevant, not_relevant = relevance
                # supports, opposes, either = valence

                rows.append((choice, vrd, description, *relevance, *valence, target))

        results = pd.DataFrame(
            rows, columns=["choice", "VRD", "KDMA", "relevant", "not_relevant", "supports", "opposes", "either", "target"])

        results['weight'] = compute_weight_fn(results)

        log.explain("[bold] ** Kaleido Computed Weights ** [/bold]",
                    extra={"markup": True})
        log.explain(results)

        choice_rows = []
        for group_key, group_records in results.groupby(['choice']):
            # group_key is a single element tuple in this case
            choice, = group_key

            sum_distance = distance_fn(group_records)

            choice_rows.append((choice, sum_distance))

        choice_results = pd.DataFrame(
            choice_rows, columns=["choice", "distance"])

        log.explain("[bold] ** Kaleido Choice Distances from Alignment Target (sorted) ** [/bold]",
                    extra={"markup": True})
        log.explain(choice_results.sort_values(by=['distance']))

        most_aligned_choice_idx = choice_results.idxmin()['distance']
        most_aligned_choice = choice_results.iloc[most_aligned_choice_idx]['choice']

        per_kdma_explanations_rows = []
        for _, r in results[results['choice'] == most_aligned_choice].iterrows():
            explanation = self.get_explanation(r['choice'], r['VRD'], r['KDMA'])
            per_kdma_explanations_rows.append(
                (choice_prompts[r['choice']], r['VRD'], r['KDMA'], explanation))

        per_kdma_explanations = pd.DataFrame(
            per_kdma_explanations_rows, columns=["choice", "VRD", "KDMA", "explanation"])

        log.explain("[bold] ** Kaleido KDMA Explanations for Choice ** [/bold]",
                    extra={"markup": True})
        log.explain(per_kdma_explanations)

        output_choice_idx = choices.index(most_aligned_choice)

        return output_choice_idx, results

    def __call__(self, sample, target_kdma_values, labels, **kwargs):
        import yaml

        from align_system.algorithms.lib.util import format_template

        kdma_descriptions_map = None
        if 'kdma_descriptions_map' in kwargs:
            if isinstance(kwargs['kdma_descriptions_map'], str):
                with open(kwargs['kdma_descriptions_map']) as f:
                    kdma_descriptions_map = yaml.safe_load(f)
            elif isinstance(kwargs['kdma_descriptions_map'], dict):
                kdma_descriptions_map = kwargs['kdma_descriptions_map']

        if kwargs.get('sample_scenario_is_yaml', False):
            scenario_data = yaml.safe_load(sample['scenario'])
        else:
            scenario_data = sample['scenario']

        distance_fn = relevance_weighted_distance_fn
        if 'distance_fn' in kwargs:
            if kwargs['distance_fn'] == 'mean':
                distance_fn = mean_distance_fn
            elif kwargs['distance_fn'] == 'mean2':
                distance_fn = mean_distance_2_fn
            elif kwargs['distance_fn'] == 'relevance':
                distance_fn = relevance_weighted_distance_fn
            else:
                raise NotImplementedError("Unsupported distance_fn: '{}'".format(kwargs['distance_fn']))

        if isinstance(scenario_data, str):
            template = '{{probe}} "{{choice}}" instead of {{other_choices}}.  In the context of the following scenario: {{scenario_data}}'
            partial_template = format_template(
                template,
                partial=True,
                probe=sample['probe'],
                scenario_data=scenario_data)
        elif isinstance(scenario_data, dict):
            template = '{{probe}} "{{choice}}" instead of {{other_choices}}.  In the context of the following scenario: {{initial_setting}} {{casualty_setting}} With the following casualties: {{casualties_str}}'
            casualties_str = '\n'.join([f'{k}: {v}' for k, v in scenario_data['Casualties'].items()])
            partial_template = format_template(
                template,
                partial=True,
                probe=sample['probe'],
                initial_setting=scenario_data['Initial setting'],
                casualty_setting=scenario_data['Casualty setting'],
                casualties_str=casualties_str)
        else:
            raise RuntimeError('Unexpected scenario_data type: {}'.format(type(scenario_data)))

        selected_choice_idx, results = self.predict_kdma_weights(
            partial_template,
            sample['choices'],
            target_kdma_values,
            distance_fn=distance_fn,
            kdma_descriptions_map=kdma_descriptions_map)

        predicted_kdma_values = []
        for choice in sample['choices']:
            predicted_kdma_values.append(
                {r['KDMA']: float(r['weight']) for _, r
                 in results[results['choice'] == choice].iterrows()})

        return {'choice': selected_choice_idx,
                'predicted_kdma_values': predicted_kdma_values}
