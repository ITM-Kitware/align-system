from abc import ABC, abstractmethod
import numpy as np
import math
import random

from align_system.utils import kde_utils


'''
Abstract class for alignment function 
Inputs:
    - kdma_values: Dictionary of choices and KDMA score(s) (between 0 and 1)
        For example: {'Treat Patient A':{'Moral judgement': [0.1,0.1,0.2], ...}, 'Treat Patient B':{...}, ... }
    - target_kdmas: A list of KDMA alignment targets (typically alignment_target.kdma_values)
        For example: [{'kdma':'Moral judgement', 'value':0}, ...] or [{'kdma':'Moral judgement', 'kdes':{...}}, ...]
    - misaligned (optional): If true will pick the least alignmed option (default is false)
    - kde_norm (optional): Normalization to use if target is KDE
        Options: 'rawscores', 'localnorm', 'globalnorm', 'globalnormx_localnormy'
Returns:
    - The selected choice from kdma_values.keys()
        For example: 'Treat Patient A'
'''
class AlignmentFunction(ABC):
    @abstractmethod
    def __call__(self, kdma_values, target_kdmas, misaligned=False, kde_norm=None):
        '''
        1. Make sure the data is in the right format (scalar vs KDE target)
        2. Compute the distance of each choice to the targets
        3. Call _select_min_dist_choice() and return selected choice and probs
        '''
        pass

    def _select_min_dist_choice(self, choices, dists, misaligned=False):
        if not misaligned:
            # For aligned, want to minimize to distance to target
            # Invert distances so minimal distances have higher probability

            # Small epsilon for a perfect (0 distance) match
            inv_dists = [1/(distance+1e-16) for distance in dists]

            # Convert inverse distances to probabilities
            probs = [inv_dist/sum(inv_dists) for inv_dist in inv_dists]
        else:
            # For misaligned, want to maximize distance to target, so
            # maximize over non-inverted distances

            # Convert distances to probabilities
            probs = [dist/sum(dists) for dist in dists]

        max_prob = max(probs)
        max_actions = [idx for idx, p in enumerate(probs) if p == max_prob]
        # Randomly chose one of the max probability actions
        # TODO could add some tie breaking logic here
        selected_choice = choices[random.choice(max_actions)]

        return selected_choice, probs


class AvgDistScalarAlignment(AlignmentFunction):
    def __call__(self, kdma_values, target_kdmas, misaligned=False): 
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
                kdma = target_kdma['kdma']
                samples = kdma_values[choice][kdma]
                average_score = (sum(samples) / len(samples))
                distance += _euclidean_distance(target_kdma['value'], average_score)
            distances.append(distance)

        selected_choice, probs = self._select_min_dist_choice(choices, distances, misaligned)

        return selected_choice, probs


class MinDistToRandomSampleKdeAlignment(AlignmentFunction): 
    def __call__(self, kdma_values, target_kdmas, misaligned=False, kde_norm='globalnorm'):
        '''
        Returns the choice with min average distance to random sample from the target KDEs
        '''
        _check_if_targets_are_kde(target_kdmas)

        # Sample KDEs to get scalar targets
        sampled_target_kdmas = []
        for target_kdma in target_kdmas:
            sampled_target_kdma = {'kdma':target_kdma.kdma}
            target_kde = kde_utils.load_kde(target_kdma, kde_norm)
            sampled_target_kdma['value']= float(target_kde.sample(1)) # sample returns array
            sampled_target_kdmas.append(sampled_target_kdma)

        # Use avergae distance to sampled scalar targets
        AlignmentFunc = AvgDistScalarAlignment()
        return AlignmentFunc(kdma_values, sampled_target_kdmas, misaligned)


class MaxLikelihoodKdeAlignment(AlignmentFunction):
    def __call__(self, kdma_values, target_kdmas, misaligned=False, kde_norm='globalnorm'):
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
            distance = 0.
            for target_kdma in target_kdmas:
                target_kde = kde_utils.load_kde(target_kdma, kde_norm)
                predicted_samples = kdma_values[choice][target_kdma.kdma]
                log_likelihoods = target_kde.score_samples(np.array(predicted_samples).reshape(-1, 1))
                total_likelihood = np.sum(np.exp(log_likelihoods))
            likelihoods.append(total_likelihood)

        # distances are inverse to likelihood
        distances = [1/(likelihood) for likelihood in likelihoods]

        selected_choice, probs = self._select_min_dist_choice(choices, distances, misaligned)

        return selected_choice, probs


class JsDivergenceKdeAlignment(AlignmentFunction):
    def __call__(self, kdma_values, target_kdmas, misaligned=False, kde_norm='globalnorm'):
        '''
        Creates predicted KDEs for each choice using sampled score predictions
        Returns the selected choice with minimum JS divergence to target KDE
        '''
        kdma_values = _handle_single_value(kdma_values, target_kdmas)
        _check_if_targets_are_kde(target_kdmas)
        target_kdma = target_kdmas[0] # TODO extend to multi-KDMA target scenario

        # Get predicted KDE for each choice and get distance to target
        distances = []
        choices = list(kdma_values.keys())
        for choice in choices:
            distance = 0.
            for target_kdma in target_kdmas:
                target_kde = kde_utils.load_kde(target_kdma, kde_norm)
                predicted_samples = kdma_values[choice][target_kdma.kdma]
                predicted_kde = kde_utils.get_kde_from_samples(predicted_samples)
                distance += kde_utils.js_distance(target_kde, predicted_kde, 100)
            distances.append(distance)

        selected_choice, probs = self._select_min_dist_choice(choices, distances, misaligned)

        return selected_choice, probs


# If score is a single value, then set it to a list containing that value
def _handle_single_value(kdma_values, target_kdmas):
    for choice in kdma_values.keys():
        for target_kdma in target_kdmas:
            kdma = target_kdma['kdma']
            if not isinstance(kdma_values[choice][kdma], list):
                kdma_values[choice][kdma] = [(kdma_values[choice][kdma])]
    return kdma_values

# Raises error if all targets aren't scalar
def _check_if_targets_are_scalar(target_kdmas):
    if len(target_kdmas) == 0:
        raise RuntimeError(f"Alignment function requires at least one KDMA target.")
    for target_kdma in target_kdmas:
        if 'value' not in target_kdma or not isinstance(target_kdma['value'], float):
            raise RuntimeError(f"Alignment function requires scalar KDMA targets.")

# Raises error if all targets aren't KDE
def _check_if_targets_are_kde(target_kdmas):
    if len(target_kdmas) == 0:
        raise RuntimeError(f"Alignment function requires at least one KDMA target.")
    for target_kdma in target_kdmas:
        if 'kde' in target_kdma:
            raise RuntimeError(f"Alignment function requires KDE targets.")

def _euclidean_distance(target, score):
    return math.sqrt((target - score)**2)