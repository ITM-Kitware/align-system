from abc import ABC, abstractmethod
from functools import reduce, partial
import inspect
import yaml

import pandas as pd
from swagger_client.models import (
    kdma_value
)

from align_system.algorithms.abstracts import ActionBasedADM
from align_system.algorithms.lib.kaleido import KaleidoSys
from align_system.algorithms.abstracts import AlignedDecisionMaker
from align_system.algorithms.lib.util import format_template
from align_system.algorithms.outlines_adm import OutlinesTransformersADM
from align_system.utils import logging
from align_system.utils import adm_utils
from align_system.utils import alignment_utils


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
                             kdmas,
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
            kdma_descriptions_map = {k: {'description': k.replace('_', ' ')} for k in kdmas}

        rows = []
        for choice in choices:
            other_choices_str = ', '.join(['"{}"'.format(c) for c in choices if c != choice])
            choice_prompt = format_template(
                prompt_template,
                allow_extraneous=True,
                choice=choice, other_choices=other_choices_str)

            log.debug("[bold] ** Kaleido Prompt ** [/bold]",
                      extra={"markup": True})
            log.debug(choice_prompt)

            for kdma in kdmas:
                mapped_kdma = kdma_descriptions_map[kdma]

                vrd = mapped_kdma.get('vrd', 'Value')
                description = mapped_kdma['description']

                relevance = self.kaleido.get_relevance(choice_prompt, vrd, description)
                valence = self.kaleido.get_valence(choice_prompt, vrd, description)

                # relevant, not_relevant = relevance
                # supports, opposes, either = valence

                explanation = self.kaleido.get_explanation(choice_prompt, vrd, description)

                rows.append((choice, vrd, kdma, description, *relevance, *valence, explanation))

        results = pd.DataFrame(
            rows, columns=["choice", "VRD", "KDMA", "kdma_description", "relevant", "not_relevant", "supports", "opposes", "either", "explanation"])

        results['estimated_kdma_value'] = estimator_fn(results)

        log.explain("[bold] ** Kaleido Relevance / Valence and Estimated "
                    "KDMA Values ** [/bold]",
                    extra={"markup": True})
        log.debug(results)

        display_results = results.copy()
        display_results[['relevant', 'supports', 'opposes', 'either', 'estimated_kdma_value']] =\
            display_results[['relevant', 'supports', 'opposes', 'either', 'estimated_kdma_value']].map(lambda x: f"{float(x):.2f}")
        log.explain(display_results[['choice', 'VRD', 'KDMA', 'kdma_description', 'relevant', 'supports', 'opposes', 'either', 'estimated_kdma_value']])

        return results

    def force_choice(self, kaleido_results, choices, target_kdma_values, distance_fn=DefaultDistanceFunction):
        # Make sure we don't modify the original when we add in the targets
        kaleido_results = kaleido_results.copy(deep=True)

        kaleido_results['target'] = kaleido_results['KDMA'].apply(
            lambda k: target_kdma_values[k])

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

    def force_choice_with_alignment_fn(self, kaleido_results, choices, target_kdmas, alignment_fn):
        kdma_values = {}
        relevance_values = {}
        for group_key, group_records in kaleido_results.groupby(['choice', 'KDMA']):
            # group_key is a single element tuple in this case
            choice, kdma = group_key

            # alignment functions expecting values to be 0-1 (rather than 0-10)
            kdma_values.setdefault(choice, {})[kdma] =\
                [float(v) for v in (group_records['estimated_kdma_value'] / 10)]

            relevance_values.setdefault(choice, {})[kdma] =\
                [float(v) for v in (group_records['estimated_kdma_value'] / 10)]

        selected_choice, probs = alignment_fn(kdma_values, target_kdmas)

        # NOTE ** Not making use of relevance at all here, could
        # consider adding a weighting argument to the alignment
        # function

        return choices.index(selected_choice)

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
            label_kdmas,
            kdma_descriptions_map=kdma_descriptions_map)

        selected_choice_idx = self.force_choice(
            kaleido_results,
            sample['choices'],
            target_kdma_values_in_labels,
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

        if 'action_template' in kwargs:
            # Example alternate template without `other_choices`:
            # template = 'Choosing to "{{choice}}".  In the context of the following scenario: {{scene_state}} {{mission}} {{decision_environment}} With the following casualties:\n{{characters_str}}'
            template = kwargs['action_template']
        else:
            template = 'What action do you take? "{{choice}}" instead of {{other_choices}}.  In the context of the following scenario: {{scene_state}} {{mission}} {{decision_environment}} With the following casualties:\n{{characters_str}}'

        characters_str = '\n'.join(['{} ({}): {}'.format(c.name, c.id, c.unstructured) for c in scenario_state.characters])
        if scenario_state.mission is None:
            mission_str = ''
        else:
            mission_str = scenario_state.mission.unstructured

        partial_template = format_template(
            template,
            partial=True,
            scene_state=scenario_state.unstructured,
            mission=mission_str,
            decision_environment=scenario_state.environment.decision_environment.unstructured.strip(),
            characters_str=characters_str)

        # Re-using the OutlinesTransformersADM format choices option
        # to ensure unstructured choice text is unique.  TODO: move
        # this function out to utilities somewhere as it's generally
        # useful
        choices_unstructured = [a.unstructured for a in available_actions]
        choices_unstructured = adm_utils.format_choices(
            choices_unstructured,
            available_actions,
            scenario_state,
            log)

        target_kdmas = alignment_target.kdma_values

        kdmas = set()
        for kdma_idx in range(len(target_kdmas)):
            if not isinstance(target_kdmas[kdma_idx], dict):
                if isinstance(target_kdmas[kdma_idx], kdma_value.KDMAValue):
                    target_kdmas[kdma_idx] = target_kdmas[kdma_idx].to_dict()
                else:
                    target_kdmas[kdma_idx] = dict(target_kdmas[kdma_idx])
            kdma = target_kdmas[kdma_idx]['kdma']
            kdmas.add(kdma)

        kaleido_results = self.estimate_kdma_values(
            partial_template,
            choices_unstructured,
            kdmas,
            kdma_descriptions_map=kdma_descriptions_map)

        # Get type of targets
        all_scalar_targets = True
        all_kde_targets = True
        for target_kdma in target_kdmas:
            if not hasattr(target_kdma, 'value') or target_kdma.value is None:
                all_scalar_targets = False
            if not hasattr(target_kdma, 'kdes') or target_kdma.kdes is None:
                all_kde_targets = False

        # Select aligned choice
        if all_scalar_targets:
            alignment_function = alignment_utils.AvgDistScalarAlignment()
        elif all_kde_targets:
            if not kwargs.get('use_alignment_utils', True):
                raise RuntimeError("Can't handle KDE alignment targets "
                                   "without `use_alignment_utils` "
                                   "(set to True and retry)")

            distribution_matching = kwargs.get('distribution_matching', 'sample')

            if distribution_matching == 'sample':
                alignment_function = alignment_utils.MinDistToRandomSampleKdeAlignment()
            elif distribution_matching == 'max_likelihood':
                alignment_function = alignment_utils.MaxLikelihoodKdeAlignment()
            elif distribution_matching == 'js_divergence':
                alignment_function = alignment_utils.JsDivergenceKdeAlignment()
            else:
                raise RuntimeError(distribution_matching, "distribution matching function unrecognized.")

            alignment_function = partial(alignment_function, kde_norm=kwargs.get('kde_norm', 'globalnorm'))
        else:
            # TODO: Currently we assume all targets either have scalar values or KDES,
            #       Down the line, we should extend to handling multiple targets of mixed types
            raise ValueError("ADM does not currently support a mix of scalar and KDE targets.")

        if kwargs.get('use_alignment_utils', True):
            selected_choice_idx = self.force_choice_with_alignment_fn(
                    kaleido_results,
                    choices_unstructured,
                    target_kdmas,
                    alignment_function)
        else:
            target_kdmas_to_values = {t['kdma']: t['value'] for t in target_kdmas}

            selected_choice_idx = self.force_choice(
                kaleido_results,
                choices_unstructured,
                target_kdmas_to_values,
                distance_fn=kwargs.get('distance_fn', DefaultDistanceFunction))

        action_to_take = available_actions[selected_choice_idx]
        action_to_take.justification = kaleido_results.loc[selected_choice_idx, :].explanation

        return action_to_take
