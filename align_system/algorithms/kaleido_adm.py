from abc import ABC, abstractmethod
from functools import reduce
import inspect
import yaml

import pandas as pd

from align_system.algorithms.abstracts import ActionBasedADM
from align_system.algorithms.lib.kaleido import KaleidoSys
from align_system.algorithms.lib.aligned_decision_maker import AlignedDecisionMaker
from align_system.algorithms.lib.util import format_template
from align_system.utils import logging


log = logging.getLogger(__name__)

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


class EstimateKDMAFunction(ABC):
    @abstractmethod
    def __call__(self, results_df: pd.DataFrame) -> pd.DataFrame:
        ...


class SimpleKDMAEstimator(EstimateKDMAFunction):
    def __call__(self, results_df: pd.DataFrame) -> pd.DataFrame:
        return results_df['either'] * 5 + results_df['supports'] * 10


class ChoiceDistanceFunction(ABC):
    @abstractmethod
    def __call__(self, group_records: pd.DataFrame) -> pd.DataFrame:
        ...


class RelevanceWeightedDistance(ChoiceDistanceFunction):
    def __call__(self, group_records: pd.DataFrame) -> pd.DataFrame:
        # (1.0 / relevant) as a weight could be too punitive?
        return sum((1.0 / group_records['relevant'])
                   * abs(group_records['estimated_kdma_value'] - group_records['target']))


class MeanDistance(ChoiceDistanceFunction):
    def __call__(self, group_records: pd.DataFrame) -> pd.DataFrame:
        return (sum(group_records['relevant']
                    * abs(group_records['estimated_kdma_value'] - group_records['target']))
                / len(group_records))


class MeanDistance2(ChoiceDistanceFunction):
    # Probably want to divide by the sum(group_records['relevant'])
    # instead of len(group_records) (as with the MeanDistance
    # function) so that we don't reward having only a few relevant
    # KDMAs (vs. having more relevant but stronger supporting KDMAs)
    def __call__(self, group_records: pd.DataFrame) -> pd.DataFrame:
        return (sum(group_records['relevant']
                    * abs(group_records['estimated_kdma_value'] - group_records['target']))
                / sum(group_records['relevant']))


DefaultKDMAEstimatorFunction = SimpleKDMAEstimator
KnownKDMAEstimatorFunctions = [SimpleKDMAEstimator]

DefaultDistanceFunction = RelevanceWeightedDistance
KnownDistanceFunctions = [RelevanceWeightedDistance,
                          MeanDistance,
                          MeanDistance2]


class KaleidoADM(AlignedDecisionMaker, ActionBasedADM):
    def __init__(self, **kwargs):
        log.info('Initializing Kaleido..')
        self.kaleido = KaleidoSys(**kwargs)
        log.info('..done initializing Kaleido')

    def estimate_kdma_values(self,
                             prompt_template,
                             choices,
                             target_kdmas,
                             estimator_fn=DefaultKDMAEstimatorFunction,
                             kdma_descriptions_map=None):
        if isinstance(estimator_fn, str):
            estimator_fn = {fn.__name__: fn for fn
                            in KnownKDMAEstimatorFunctions}.get(
                                estimator_fn, estimator_fn)

        if issubclass(estimator_fn, EstimateKDMAFunction):
            estimator_fn = estimator_fn()
        elif isinstance(estimator_fn, EstimateKDMAFunction):
            # Already initialized
            pass
        else:
            raise RuntimeError(
                f"Estimator function '{estimator_fn}' not "
                "found, or does not implement EstimateKDMAFunction")

        if kdma_descriptions_map is None:
            kdma_descriptions_map = {k: {'description': k.replace('_', ' ')} for k in target_kdmas.keys()}

        rows = []
        for choice in choices:
            other_choices_str = ', '.join(['"{}"'.format(c) for c in (set(choices) - {choice})])
            choice_prompt = format_template(
                prompt_template,
                allow_extraneous=True,
                choice=choice, other_choices=other_choices_str)

            log.debug("[bold] ** Kaleido Prompt ** [/bold]",
                      extra={"markup": True})
            log.debug(choice_prompt)

            for kdma, target in target_kdmas.items():
                mapped_kdma = kdma_descriptions_map[kdma]

                vrd = mapped_kdma.get('vrd', 'Value')
                description = mapped_kdma['description']

                relevance = self.kaleido.get_relevance(choice_prompt, vrd, description)
                valence = self.kaleido.get_valence(choice_prompt, vrd, description)

                # relevant, not_relevant = relevance
                # supports, opposes, either = valence

                explanation = self.kaleido.get_explanation(choice_prompt, vrd, description)

                rows.append((choice, vrd, description, *relevance, *valence, target, explanation))

        results = pd.DataFrame(
            rows, columns=["choice", "VRD", "KDMA", "relevant", "not_relevant", "supports", "opposes", "either", "target", "explanation"])

        results['estimated_kdma_value'] = estimator_fn(results)

        log.explain("[bold] ** Kaleido Relevance / Valence and Estimated "
                    "KDMA Values ** [/bold]",
                    extra={"markup": True})
        log.debug(results)

        display_results = results.copy()
        display_results[['relevant', 'supports', 'opposes', 'either', 'target', 'estimated_kdma_value']] =\
            display_results[['relevant', 'supports', 'opposes', 'either', 'target', 'estimated_kdma_value']].map(lambda x: f"{float(x):.2f}")
        log.explain(display_results[['choice', 'VRD', 'KDMA', 'relevant', 'supports', 'opposes', 'either', 'target', 'estimated_kdma_value']])

        return results

    def force_choice(self, kaleido_results, choices, distance_fn=DefaultDistanceFunction):
        if isinstance(distance_fn, str):
            distance_fn = {fn.__name__: fn for fn
                           in KnownDistanceFunctions}.get(
                               distance_fn, distance_fn)

        if inspect.isclass(distance_fn) and issubclass(distance_fn, ChoiceDistanceFunction):
            distance_fn = distance_fn()
        elif isinstance(distance_fn, ChoiceDistanceFunction):
            # Already initialized
            pass
        else:
            raise RuntimeError(
                f"Distance function '{distance_fn}' not "
                "found, or does not implement ChoiceDistanceFunction")

        choice_rows = []
        for group_key, group_records in kaleido_results.groupby(['choice']):
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

        log.explain("[bold] ** Kaleido KDMA Explanations for Choice ** [/bold]",
                    extra={"markup": True})
        per_kdma_explanations = kaleido_results[kaleido_results['choice'] == most_aligned_choice][["choice", "VRD", "KDMA", "explanation"]]
        log.explain(per_kdma_explanations)

        output_choice_idx = choices.index(most_aligned_choice)

        return output_choice_idx

    def __call__(self, sample, target_kdma_values, labels, **kwargs):
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

        label_kdmas = reduce(set.union, (choice_kdmas.keys() for choice_kdmas in labels), set())
        target_kdma_values_in_labels = {k: target_kdma_values[k] for k in label_kdmas}

        kaleido_results = self.estimate_kdma_values(
            partial_template,
            sample['choices'],
            target_kdma_values_in_labels,
            kdma_descriptions_map=kdma_descriptions_map)

        selected_choice_idx = self.force_choice(
            kaleido_results,
            sample['choices'],
            distance_fn=kwargs.get('distance_fn', DefaultDistanceFunction))

        predicted_kdma_values = []
        for choice in sample['choices']:
            predicted_kdma_values.append(
                {r['KDMA']: float(r['estimated_kdma_value']) for _, r
                 in kaleido_results[kaleido_results['choice'] == choice].iterrows()})

        return {'choice': selected_choice_idx,
                'predicted_kdma_values': predicted_kdma_values}

    def choose_action(self, scenario_state, available_actions, alignment_target, **kwargs):
        kdma_descriptions_map = None
        if 'kdma_descriptions_map' in kwargs:
            if isinstance(kwargs['kdma_descriptions_map'], str):
                with open(kwargs['kdma_descriptions_map']) as f:
                    kdma_descriptions_map = yaml.safe_load(f)
            elif isinstance(kwargs['kdma_descriptions_map'], dict):
                kdma_descriptions_map = kwargs['kdma_descriptions_map']

        template = 'What action do you take? "{{choice}}" instead of {{other_choices}}.  In the context of the following scenario: {{scene_state}} {{mission}} {{decision_environment}} With the following casualties:\n{{characters_str}}'
        characters_str = '\n'.join(['{} ({}): {}'.format(c.name, c.id, c.unstructured) for c in scenario_state.characters])
        partial_template = format_template(
            template,
            partial=True,
            scene_state=scenario_state.unstructured,
            mission=scenario_state.mission.unstructured,
            decision_environment=scenario_state.environment.decision_environment.unstructured.strip(),
            characters_str=characters_str)

        # Scaling KDMA values by 10 (range should be 0-10)
        target_kdma_values = {t.kdma: t.value * 10 for t in alignment_target.kdma_values}

        choices_unstructured = [a.unstructured for a in available_actions]

        kaleido_results = self.estimate_kdma_values(
            partial_template,
            choices_unstructured,
            target_kdma_values,
            kdma_descriptions_map=kdma_descriptions_map)

        selected_choice_idx = self.force_choice(
            kaleido_results,
            choices_unstructured,
            distance_fn=kwargs.get('distance_fn', DefaultDistanceFunction))

        return available_actions[selected_choice_idx]
