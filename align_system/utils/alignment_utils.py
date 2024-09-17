from abc import ABC, abstractmethod
import numpy as np
import math
import random

from align_system.utils import kde_utils
from align_system.utils import logging
from swagger_client.models import KDMAValue

log = logging.getLogger(__name__)


'''
Abstract class for alignment function
Inputs:
    - kdma_values: Dictionary of choices and KDMA score(s) (between 0 and 1)
        For example: {'Treat Patient A':{'Moral judgement': [0.1,0.1,0.2], ...}, 'Treat Patient B':{...}, ... }
    - target_kdmas: A list of KDMA alignment targets (typically alignment_target.kdma_values)
        For example: [{'kdma':'Moral judgement', 'value':0}, ...] or [{'kdma':'Moral judgement', 'kdes':{...}}, ...]
    - misaligned (optional): If true will pick the least aligned option (default is false)
    - kde_norm (optional): Normalization to use if target is KDE
        Options: 'rawscores', 'localnorm', 'globalnorm', 'globalnormx_localnormy'
    - probabilisitic (optional): If true, will select action probabilistically, weighted by distances
        to alignment target (default is false)
Returns:
    - The selected choice from kdma_values.keys()
        For example: 'Treat Patient A'
    - The probability associated with each choice
'''
class AlignmentFunction(ABC):
    @abstractmethod
    def __call__(self, kdma_values, target_kdmas, misaligned=False, kde_norm=None, probabilistic=False):
        '''
        1. Make sure the data is in the right format (scalar vs KDE target)
        2. Compute the distance of each choice to the targets
        3. Call _select_min_dist_choice() to get selected choice and probs
        4. Optionally find the index of the best sample for justification
        '''
        pass

    def _select_min_dist_choice(self, choices, dists, misaligned=False, probabilistic=False):
        if len(dists) != len(choices):
            raise RuntimeError("A distance must be provided for each choice during alignment")
        if len(choices) == 0:
            raise RuntimeError("No choices provided")

        eps = 1e-16
        if not misaligned:
            # For aligned, want to minimize to distance to target
            # Invert distances so minimal distances have higher probability

            # Small epsilon for a perfect (0 distance) match
            inv_dists = [1/(distance+eps) for distance in dists]

            # Convert inverse distances to probabilities
            probs = [inv_dist/sum(inv_dists) for inv_dist in inv_dists]
        else:
            # For misaligned, want to maximize distance to target, so
            # maximize over non-inverted distances

            # Convert distances to probabilities
            # Avoid divide by zero if all distances are 0
            probs = [dist/(sum(dists)+eps) for dist in dists]

        if probabilistic:
            selected_choice = np.random.choice(choices, p=probs)
        else:
            max_prob = max(probs)
            max_actions = [idx for idx, p in enumerate(probs) if p == max_prob]
            # Randomly chose one of the max probability actions
            # TODO could add some tie breaking logic here
            selected_choice = choices[random.choice(max_actions)]


        probs_dict = {c: p for c, p in zip(choices, probs)}

        return selected_choice, probs_dict

    # Given the selected choice, get the index of the sample closest to the target
    def get_best_sample_index(self, kdma_values, target_kdmas, selected_choice, misaligned=False, kde_norm=None):
        pass


class AvgDistScalarAlignment(AlignmentFunction):
    def __call__(self, kdma_values, target_kdmas, misaligned=False, probabilistic=False):
        '''
        Selects a choice by first averaging score across samples,
        then selecting the one with minimal MSE to the scalar target.
        Returns the selected choice.
        '''
        kdma_values = _handle_single_value(kdma_values, target_kdmas)
        _check_if_targets_are_scalar(target_kdmas)

        # Get distance from average of predicted scores to targets
        distances = []
        choices = list(kdma_values.keys())
        for choice in choices:
            distance = 0.
            for target_kdma in target_kdmas:
                if isinstance(target_kdma, KDMAValue):
                    target_kdma = target_kdma.to_dict()

                kdma = target_kdma['kdma']
                samples = kdma_values[choice][kdma]
                average_score = (sum(samples) / len(samples))
                distance += _euclidean_distance(target_kdma['value'], average_score)
            distances.append(distance)

        selected_choice, probs = self._select_min_dist_choice(choices, distances, misaligned, probabilistic=probabilistic)
        return selected_choice, probs

    def get_best_sample_index(self, kdma_values, target_kdmas, selected_choice, misaligned=False):
        sample_distances = []
        sample_indices = range(len(kdma_values[selected_choice][target_kdmas[0]['kdma']]))
        if len(sample_indices) == 1:
            best_sample_index = 0
        else:
            # For the selected choice, find the sample closest to the target
            for sample_idx in sample_indices:
                sample_dist = 0
                for target_kdma in target_kdmas:
                    if isinstance(target_kdma, KDMAValue):
                        target_kdma = target_kdma.to_dict()

                    sample = kdma_values[selected_choice][target_kdma['kdma']][sample_idx]
                    sample_dist += _euclidean_distance(target_kdma['value'], sample)
                sample_distances.append(sample_dist)
            best_sample_index, _ = self._select_min_dist_choice(sample_indices, sample_distances, misaligned)
        return best_sample_index


class MinDistToRandomSampleKdeAlignment(AlignmentFunction):
    def __call__(self, kdma_values, target_kdmas, misaligned=False, kde_norm='globalnorm', probabilistic=False):
        '''
        Returns the choice with min average distance to random sample from the target KDEs
        '''
        _check_if_targets_are_kde(target_kdmas)

        # Sample KDEs to get scalar targets
        self.sampled_target_kdmas = []
        for target_kdma in target_kdmas:

            if isinstance(target_kdma, KDMAValue):
                target_kdma = target_kdma.to_dict()

            sampled_target_kdma = {'kdma':target_kdma["kdma"]}
            target_kde = kde_utils.load_kde(target_kdma, kde_norm)
            kde_sample = float(target_kde.sample(1)) # sample returns array
            kde_sample = max(0.0, min(kde_sample, 1.0)) # clamp to be between 0 and 1
            sampled_target_kdma['value'] = kde_sample
            log.info("Sampled Target KDMA Value(s): {}".format(sampled_target_kdma['value']))

            # Log average KDMA values for each KDMA/choice; this
            # should probably done outside of this method
            for choice, kv in kdma_values.items():
                for k, v in kv.items():
                    if isinstance(v, float) or isinstance(v, int):
                        avg = v
                    else:
                        avg = sum(v) / len(v)

                    log.info('KDMA "{}" Values for "{}": {} (average: {:0.3f})'.format(
                        k, choice, v, avg))

            self.sampled_target_kdmas.append(sampled_target_kdma)

        # Use avergae distance to sampled scalar targets
        avg_alignment_function = AvgDistScalarAlignment()
        return avg_alignment_function(kdma_values, self.sampled_target_kdmas, misaligned=misaligned, probabilistic=probabilistic)

    def get_best_sample_index(self, kdma_values, target_kdmas, selected_choice, misaligned=False, kde_norm=None):
        # Use avergae distance to sampled scalar targets
        avg_alignment_function = AvgDistScalarAlignment()
        return avg_alignment_function.get_best_sample_index(kdma_values, self.sampled_target_kdmas, selected_choice, misaligned=misaligned)


class MaxLikelihoodKdeAlignment(AlignmentFunction):
    def __call__(self, kdma_values, target_kdmas, misaligned=False, kde_norm='globalnorm', probabilistic=False):
        '''
        Gets the likelihood of sampled score predictions under the target KDE for each choice
        Returns the selected choice with maximum average likelihood
        '''
        kdma_values = _handle_single_value(kdma_values, target_kdmas)
        _check_if_targets_are_kde(target_kdmas)

        # Get likelihood of each choice under KDE
        likelihoods = []
        choices = list(kdma_values.keys())
        for choice in choices:
            total_likelihood = 0.
            for target_kdma in target_kdmas:
                if isinstance(target_kdma, KDMAValue):
                    target_kdma = target_kdma.to_dict()

                target_kde = kde_utils.load_kde(target_kdma, kde_norm)
                predicted_samples = kdma_values[choice][target_kdma.kdma]
                log_likelihoods = target_kde.score_samples(np.array(predicted_samples).reshape(-1, 1))
                total_likelihood += np.sum(np.exp(log_likelihoods))
            likelihoods.append(total_likelihood)

        # distances are inverse to likelihood
        distances = [1/(likelihood) for likelihood in likelihoods]

        selected_choice, probs = self._select_min_dist_choice(choices, distances, misaligned, probabilistic=probabilistic)
        return selected_choice, probs

    def get_best_sample_index(self, kdma_values, target_kdmas, selected_choice, misaligned=False, kde_norm=None):
        sample_distances = []
        sample_indices = range(len(kdma_values[selected_choice][target_kdmas[0]['kdma']]))
        if len(sample_indices) == 1:
            best_sample_index = 0
        else:
            # For the selected choice, find the sample closest to the target
            for sample_idx in sample_indices:
                sample_dist = 0
                for target_kdma in target_kdmas:
                    if isinstance(target_kdma, KDMAValue):
                        target_kdma = target_kdma.to_dict()

                    target_kde = kde_utils.load_kde(target_kdma, kde_norm)
                    sample = kdma_values[selected_choice][target_kdma['kdma']][sample_idx]
                    likelihood = np.exp(target_kde.score_samples(np.array([sample]).reshape(-1, 1))[0])
                    sample_dist += 1/likelihood
                sample_distances.append(sample_dist)
            best_sample_index, _ = self._select_min_dist_choice(sample_indices, sample_distances, misaligned)
        return best_sample_index


class JsDivergenceKdeAlignment(AlignmentFunction):
    def __call__(self, kdma_values, target_kdmas, misaligned=False, kde_norm='globalnorm', probabilistic=False):
        '''
        Creates predicted KDEs for each choice using sampled score predictions
        Returns the selected choice with minimum JS divergence to target KDE
        '''
        kdma_values = _handle_single_value(kdma_values, target_kdmas)
        _check_if_targets_are_kde(target_kdmas)

        # Get predicted KDE for each choice and get distance to target
        distances = []
        choices = list(kdma_values.keys())
        for choice in choices:
            distance = 0.
            for target_kdma in target_kdmas:
                if isinstance(target_kdma, KDMAValue):
                    target_kdma = target_kdma.to_dict()

                target_kde = kde_utils.load_kde(target_kdma, kde_norm)
                predicted_samples = kdma_values[choice][target_kdma.kdma]
                predicted_kde = kde_utils.get_kde_from_samples(predicted_samples)
                distance += kde_utils.js_distance(target_kde, predicted_kde, 100)
            distances.append(distance)

        selected_choice, probs = self._select_min_dist_choice(choices, distances, misaligned, probabilistic=probabilistic)
        return selected_choice, probs

    def get_best_sample_index(self, kdma_values, target_kdmas, selected_choice, misaligned=False, kde_norm=None):
        # Use max likelihood as distance from a sample to the distribution because JS is disitribution to distribution
        ml_alignment_function = MaxLikelihoodKdeAlignment()
        return ml_alignment_function.get_best_sample_index(kdma_values, target_kdmas, selected_choice, misaligned=misaligned, kde_norm=kde_norm)


class CumulativeJsDivergenceKdeAlignment(AlignmentFunction):
    def __call__(self, kdma_values, target_kdmas, choice_history, misaligned=False, kde_norm='globalnorm', probabilistic=False):
        '''
        Creates potential cumulative KDEs (with history) for each choice by adding mean of sampled score predictions
        Returns the selected choice resulting in cumulative KDE with minimum JS divergence to target KDE
        '''
        kdma_values = _handle_single_value(kdma_values, target_kdmas)
        _check_if_targets_are_kde(target_kdmas)

        # Get predicted KDE for each choice and get distance to target
        distances = []
        choices = list(kdma_values.keys())
        for choice in choices:
            distance = 0.
            for target_kdma in target_kdmas:
                if isinstance(target_kdma, KDMAValue):
                    target_kdma = target_kdma.to_dict()
                if target_kdma.kdma not in choice_history:
                    choice_history[target_kdma.kdma] = []
                target_kde = kde_utils.load_kde(target_kdma, kde_norm)
                predicted_samples = kdma_values[choice][target_kdma.kdma]
                history_and_predicted_samples = choice_history[target_kdma.kdma] + [np.mean(predicted_samples)]
                predicted_kde = kde_utils.get_kde_from_samples(history_and_predicted_samples)
                distance += kde_utils.js_distance(target_kde, predicted_kde, 100)
            distances.append(distance)

        selected_choice, probs = self._select_min_dist_choice(choices, distances, misaligned, probabilistic=probabilistic)

        return selected_choice, probs

    def get_best_sample_index(self, kdma_values, target_kdmas, selected_choice, misaligned=False, kde_norm=None):
        # Use max likelihood as distance from a sample to the distribution because JS is disitribution to distribution
        ml_alignment_function = MaxLikelihoodKdeAlignment()
        return ml_alignment_function.get_best_sample_index(kdma_values, target_kdmas, selected_choice, misaligned=misaligned, kde_norm=kde_norm)

# If score is a single value, then set it to a list containing that value
def _handle_single_value(kdma_values, target_kdmas):
    for choice in kdma_values.keys():
        for target_kdma in target_kdmas:
            if isinstance(target_kdma, KDMAValue):
                target_kdma = target_kdma.to_dict()

            kdma = target_kdma['kdma']
            # Check that we have a value for the KDMA
            if kdma not in kdma_values[choice]:
                raise RuntimeError(f"Missing value for {kdma} in alignment function.")
            # If there is only a single value, set it to a list
            elif not isinstance(kdma_values[choice][kdma], list):
                kdma_values[choice][kdma] = [(kdma_values[choice][kdma])]
    return kdma_values

# Raises error if all targets aren't scalar
def _check_if_targets_are_scalar(target_kdmas):
    if len(target_kdmas) == 0:
        raise RuntimeError("Alignment function requires at least one KDMA target.")
    for target_kdma in target_kdmas:
        if isinstance(target_kdma, KDMAValue):
            target_kdma = target_kdma.to_dict()

        if 'value' not in target_kdma or not isinstance(target_kdma['value'], float):
            raise RuntimeError("Alignment function requires scalar KDMA targets.")

# Raises error if all targets aren't KDE
def _check_if_targets_are_kde(target_kdmas):
    if len(target_kdmas) == 0:
        raise RuntimeError("Alignment function requires at least one KDMA target.")
    for target_kdma in target_kdmas:
        if isinstance(target_kdma, KDMAValue):
            target_kdma = target_kdma.to_dict()

        if 'kdes' not in target_kdma or target_kdma["kdes"] is None:
            raise RuntimeError("Alignment function requires KDE targets.")

def _euclidean_distance(target, score):
    return math.sqrt((target - score)**2)
