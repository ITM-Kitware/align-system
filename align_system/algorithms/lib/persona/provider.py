import json
import logging
import os
import numpy as np
import random
from typing import List, Optional, Dict

from swagger_client.models import  AlignmentTarget

from align_system.algorithms.lib.persona.types import Dialog, Backstory
from align_system.algorithms.lib.persona.templates import BACKSTORY_ASSISTANT_PROMPT


KDMA_TO_PROBE_MAPPING = {
    'MoralDesert': 'moral_judgment',
    'maximization': 'maximization',
}

logger = logging.getLogger(__name__)


class PersonaProvider:

    def __init__(self,
                 backstory_collection: str = os.path.join(
                    os.path.abspath(os.path.dirname(__file__)), "..", "prompt_engineering", "personas", "backstories.json"
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

    def _choose_backstories_for_alignment_target(self, alignment_target: Optional[type[AlignmentTarget]], n: int, cache: bool = True) -> List[Backstory]:
        """
        Chooses backstories based on the provided alignment target and number.

        Args:
            alignment_target: The type of alignment target to choose backstories for.
            n: The number of backstories to select.
            cache: A flag indicating whether to cache the selected backstories (default is True).

        Returns:
            List[Backstory]: A list of selected backstories based on the alignment target and number provided.
        """


        if alignment_target is None or len(alignment_target.kdma_values) == 0: # type: ignore
            # There's no alignment target, so randomly sample a set of backstories
            return random.sample(self._backstories, n)

        # Map the alignment target to a probe
        alignment_target_dict = alignment_target.to_dict() # type: ignore
        probe_values = {
            KDMA_TO_PROBE_MAPPING[k['kdma']]: k['value'] * 10
            for k in alignment_target_dict.get('kdma_values', [])
            if k['kdma'] in KDMA_TO_PROBE_MAPPING
        }

        # Serialize the probe values into a repeatable string
        probe_values_str = json.dumps(probe_values, sort_keys=True)
        if probe_values_str in self._backstory_alignment_cache:
            return self._backstory_alignment_cache[probe_values_str]

        # Now that we have the probe values, we can find a set of backstories that maximize the value.
        # For each backstory, add the value
        backstories_with_values = []
        for backstory in self._backstories:
            value = sum(
                np.abs(probe['response_value'] - probe_values.get(probe['probe'], 0))
                for probe in backstory['probes']
            )
            backstories_with_values.append((backstory, value))

        # Sort by value (largest to smallest)
        backstories_with_values.sort(key=lambda x: x[1])

        # Cache the panel
        sampled_backstories = [b[0] for b in backstories_with_values[:n]]
        self._backstory_alignment_cache[probe_values_str] = sampled_backstories

        # Return the top N backstories
        return sampled_backstories


    def _generate_dialog_from_backstory(self, backstory: Backstory) -> Dialog:
        """
        Generates a dialog based on the provided backstory.

        Args:
            backstory: The backstory containing user prompts and assistant responses.

        Returns:
            Dialog: A list of dictionaries representing the dialog between user and assistant.
        """

        dialog: Dialog = [{
            "role": "user",
            "content": BACKSTORY_ASSISTANT_PROMPT
        }, {
            "role": "assistant",
            "content": backstory['backstory'],
        }]

        for probe in backstory['probes']:
            dialog.extend(
                (
                    {"role": "user", "content": probe['probe_prompt']},
                    {"role": "assistant", "content": probe['response']},
                )
            )

        return dialog

    def get_persona_dialogs(self, alignment_target: Optional[type[AlignmentTarget]], n: int, cache: bool = True) -> List[Dialog]:
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

        # Step 2: Generate the dialogs
        return [self._generate_dialog_from_backstory(b) for b in backstories]
