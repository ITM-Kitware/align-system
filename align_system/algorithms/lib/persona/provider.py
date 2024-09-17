import json
import logging
import os
import numpy as np
import random
from typing import List, Optional, Dict, Callable

import yaml
from swagger_client.models import  AlignmentTarget

from align_system.algorithms.lib.persona.types import Dialog, Backstory
from align_system.algorithms.lib.persona.templates import BACKSTORY_ASSISTANT_PROMPT
# 'personas_pe' imported here to be used as the base directory for the
# 'backstories.json'; TODO: more consistent / long term plan for
# storing database files etc. for algorithms
from align_system.prompt_engineering import personas as personas_pe
from align_system.utils.alignment_utils import (
    infer_alignment_target_type,
    AlignmentTargetType)
from align_system.utils import kde_utils


KDMA_TO_PROBE_MAPPING = {
    'MoralDesert': 'moral_judgment',
    'maximization': 'maximization',
    'Moral judgement': 'moral_judgment',
    'ingroup_bias': 'ingroup_bias',
    'IngroupBias': 'ingroup_bias',
    'Ingroup Bias': 'ingroup_bias',
    'Ingroup bias': 'ingroup_bias',
    'value_of_life': 'value_of_life',
    'ValueOfLife': 'value_of_life',
    'Value of life': 'value_of_life',
    'perceived_quantity_of_lives_saved': 'value_of_life',
    'PerceivedQuantityOfLivesSaved': 'value_of_life',
    'Perceived quantity of lives saved': 'value_of_life',
    'quality_of_life': 'quality_of_life',
    'QualityOfLife': 'quality_of_life',
    'Quality of life': 'quality_of_life',
}

logger = logging.getLogger(__name__)


def _default_probe_filter(probe):
    return True


class ScalarAlignmentDistanceFunc:
    def __init__(self, alignment_target_dict):
        self.probe_values = {
            KDMA_TO_PROBE_MAPPING[k['kdma']]: k['value'] * 10
            for k in alignment_target_dict.get('kdma_values', [])
            if k['kdma'] in KDMA_TO_PROBE_MAPPING
        }

    def __call__(self, backstories):
        # Now that we have the probe values, we can find a set of
        # backstories that maximize the value.  For each backstory,
        # add the value
        backstories_with_values = []
        for backstory in backstories:
            value = sum(
                np.abs(probe['response_value'] - self.probe_values.get(probe['probe'], 0))
                for probe in backstory['probes']
            )
            backstories_with_values.append((backstory, value))

        # Sort by value (smallest to largest)
        backstories_with_values.sort(key=lambda x: x[1])

        return [b[0] for b in backstories_with_values]


class KDEAlignmentDistanceFunc:
    def __init__(self, alignment_target_dict, kde_norm='globalnorm'):
        self.target_kdes = {}
        for targ in alignment_target_dict['kdma_values']:
            kdma = targ['kdma']
            if kdma in KDMA_TO_PROBE_MAPPING:
                self.target_kdes[KDMA_TO_PROBE_MAPPING[kdma]] =\
                    kde_utils.load_kde(targ, kde_norm)

    def __call__(self, backstories):
        likelihoods = []
        for kdma, kde in self.target_kdes.items():
            backstory_kdma_values = []
            for backstory in backstories:
                probe_responses = [p['response_value']
                                   for p in backstory['probes']
                                   if p['probe'] == kdma]

                # Assuming we only have a single probe response for a
                # given KDMA
                assert len(probe_responses) == 1

                # Backstory response values are from 1-10; KDE values
                # are from 0-1
                backstory_kdma_values.append(probe_responses[0] / 10.0)

            samples = np.array(backstory_kdma_values).reshape(-1, 1)
            likelihoods.append(
                np.exp(kde.score_samples(samples)))

        total_likelihoods = sum(likelihoods)

        backstories_with_total_likelihoods = list(zip(backstories, total_likelihoods))

        # Sort by total likelihood (largest to smallest)
        backstories_with_total_likelihoods.sort(key=lambda x: x[1], reverse=True)

        return [b[0] for b in backstories_with_total_likelihoods]


class PersonaProvider:

    def __init__(self,
                 backstory_collection: str = os.path.join(
                    os.path.abspath(os.path.dirname(personas_pe.__file__)), "backstories.json"
                ),
    ) -> None:
        """
        Initializes the provider with a collection of backstories.

        Args:
            backstory_collection: The path to the file containing the backstories.

        Returns:
            None
        """

        self._backstories: List[Backstory] = self._load_backstories(backstory_collection)
        self._backstory_alignment_cache: Dict[str, List[Backstory]] = {}


    def _load_backstories(self, backstory_collection: str) -> List[Backstory]:
        """Load backstories from a JSON file.

        Args:
            backstory_collection (str): The path to the JSON file containing the backstories.

        Returns:
            List[Backstory]: A list of backstories, each containing a backstory and a list of probes.
        """

        # Load the backstories, and setup configuration for the panel
        logger.info(f"Loading backstories from {backstory_collection}")

        with open(backstory_collection) as jf:
            data = json.load(jf)
            # If backstories are a list of string, reformat into backstory format
            if all(isinstance(b, str) for b in data):
                return [{"backstory": b, "probes": []} for b in data]

            return data

    def _choose_backstories_for_alignment_target(self, alignment_target: Optional[type[AlignmentTarget]], n: int, kde_norm='globalnorm', cache: bool = True) -> List[Backstory]:
        """
        Chooses backstories based on the provided alignment target and number.

        Args:
            alignment_target: The type of alignment target to choose backstories for.
            n: The number of backstories to select.
            kde_norm: Which KDE to use if alignment target is a distribution target (default is "globalnorm")
            cache: A flag indicating whether to cache the selected backstories (default is True).

        Returns:
            List[Backstory]: A list of selected backstories based on the alignment target and number provided.
        """
        if alignment_target is None or len(alignment_target.kdma_values) == 0: # type: ignore
            if cache and 'no_alignment_target' in self._backstory_alignment_cache:
                return self._backstory_alignment_cache['no_alignment_target']
            elif not cache and 'no_alignment_target' in self._backstory_alignment_cache:
                del self._backstory_alignment_cache['no_alignment_target']

            # There's no alignment target, so randomly sample a set of backstories
            sample = random.sample(self._backstories, n)
            if cache:
                self._backstory_alignment_cache['no_alignment_target'] = sample
            return sample

        # Map the alignment target to a probe
        alignment_target_dict = alignment_target.to_dict() # type: ignore

        # Serialize the probe values into a repeatable string
        probe_values_str = yaml.dump(alignment_target_dict, sort_keys=True)
        if cache and probe_values_str in self._backstory_alignment_cache:
            return self._backstory_alignment_cache[probe_values_str]

        target_type = infer_alignment_target_type(alignment_target)
        if target_type == AlignmentTargetType.SCALAR:
            backstory_sorter = ScalarAlignmentDistanceFunc(alignment_target_dict)
        elif target_type == AlignmentTargetType.KDE:
            backstory_sorter = KDEAlignmentDistanceFunc(alignment_target_dict)

        sorted_backstories = backstory_sorter(self._backstories)

        # Cache the panel
        sampled_backstories = sorted_backstories[:n]

        if cache:
            self._backstory_alignment_cache[probe_values_str] = sampled_backstories
        elif probe_values_str in self._backstory_alignment_cache:
            # Clear the cache if it's not supposed to be cached
            del self._backstory_alignment_cache[probe_values_str]

        # Return the top N backstories
        return sampled_backstories

    def _generate_dialog_from_backstory(self, backstory: Backstory,  probe_filter: Callable[[Dict], bool] = _default_probe_filter) -> Dialog:
        """
        Generates a dialog based on the provided backstory.

        Args:
            backstory: The backstory containing user prompts and assistant responses.

        Returns:
            Dialog: A list of dictionaries representing the dialog between user and assistant.
        """

        dialog: Dialog = [{
            "role": "user",
            "content": BACKSTORY_ASSISTANT_PROMPT # Question 1
        }, {
            "role": "assistant",
            "content": backstory['backstory'],
        }]

        for i, probe in enumerate(backstory['probes']):
            if probe_filter is not None and not probe_filter(probe):
                continue

            dialog.extend(
                (
                    {"role": "user", "content": f'Question {i + 2}: {probe["probe_prompt"]}'}, # Question 2, 3, 4, ... etc.
                    {"role": "assistant", "content": probe['response']},
                )
            )

        return dialog

    def get_persona_dialogs(self, alignment_target: Optional[type[AlignmentTarget]], n: int, cache: bool = True, filter_probes_to_target_kdmas: bool = True) -> List[Dialog]:
        """
        Retrieves persona dialogs based on the specified alignment target and number.

        Args:
            alignment_target: The type of alignment target to generate dialogs for.
            n: The number of dialogs to generate.
            cache: A flag indicating whether to cache the generated dialogs (default is True).

        Returns:
            List[Dialog]: A list of dialogs generated based on the alignment target and number provided.
        """


        # Step 1: Choose the backstories
        backstories = self._choose_backstories_for_alignment_target(alignment_target, n, cache=cache)

        if filter_probes_to_target_kdmas:
            kdmas = {KDMA_TO_PROBE_MAPPING[k['kdma']]
                     for k in alignment_target.kdma_values}

            def _backstory_probe_filter(probe):
                return probe['probe'] in kdmas
        else:
            _backstory_probe_filter = _default_probe_filter

        # Step 2: Generate the dialogs
        return [self._generate_dialog_from_backstory(b, _backstory_probe_filter) for b in backstories]
