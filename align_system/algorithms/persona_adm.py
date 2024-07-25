import json
import logging
import os
import numpy as np
import random
from typing import List, Optional, TypedDict

import torch
from jinja2.exceptions import TemplateError
from swagger_client.models import Action, ActionTypeEnum, AlignmentTarget, InjuryLocationEnum, State
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList
from transformers.generation.stopping_criteria import (
    StopStringCriteria,
)

from align_system.algorithms.abstracts import ActionBasedADM, AlignedDecisionMaker
from align_system.algorithms.llama_2_single_kdma_adm import (
    TAGGING_MULTIPLE_CHOICE_JSON_FORMAT,
    TREATMENT_MULTIPLE_CHOICE_JSON_FORMAT,
    Llama2SingleKDMAADM,
    build_multiple_choice_dialog,
)


# Some types
class Probe(TypedDict):
    probe: str
    probe_prompt: str
    response: str
    response_value: float


class Backstory(TypedDict):
    backstory: str
    probes: List[Probe]


class ChoiceInfo(TypedDict):
    reasoning: str
    parse_method: str


class Choice(TypedDict):
    choice: int
    info: ChoiceInfo


# Some Exceptions
class ResponseGenerationError(RuntimeError):
    pass


logger = logging.getLogger(__name__)


def log_dialog(dialog):
    for e in dialog:
        color = "yellow" if e.get("role") == "system" else "blue"

        logger.debug(f"[bold {color}]**{e.get('role')}**[/bold {color}]", extra={"markup": True})
        logger.debug(f"[{color}]{e.get('content')}[/{color}]", extra={"markup": True, "highlighter": None})



KDMA_TO_PROBE_MAPPING = {
    'MoralDesert': 'moral_judgment',
    'maximization': 'maximization',
    'Moral judgement': 'moral_judgment',
}



class PersonaADM(AlignedDecisionMaker, ActionBasedADM):
    def __init__(
        self,
        device: str = "auto",
        root_model: str = "meta-llama/Llama-2-7b-chat-hf",
        backstory_collection: str = os.path.join(
            os.path.abspath(os.path.dirname(__file__)), "..", "prompt_engineering", "personas", "backstories.json"
        ),
        panel_size: int = 5,
        generation_retries: int = 3,
        generation_kwargs: Optional[dict] = None,
    ):

        # Load the backstories, and setup configuration for the panel
        logger.info(f"Loading backstories from {backstory_collection}")
        with open(backstory_collection) as jf:
            data = json.load(jf)
            # If backstories are a list of string, reformat into backstory format
            if all(isinstance(b, str) for b in data):
                self._backstories: List[Backstory] = [{"backstory": b, "probes": []} for b in data]
            else:
                self._backstories: List[Backstory] = data

        self._backstory_panel_size = panel_size
        self._backstory_alignment_cache = {}

        # Load and setup the root model for decision making
        logger.info(f"Loading root model: {root_model}")
        self._root_model = AutoModelForCausalLM.from_pretrained(
            root_model, device_map=device, token=os.getenv("HUGGINGFACE_API_KEY"), torch_dtype=torch.bfloat16
        )
        self._root_model_tokenizer = AutoTokenizer.from_pretrained(root_model, token=os.getenv("HUGGINGFACE_API_KEY"))
        self._root_model_generation_kwargs = generation_kwargs or {}
        self._generation_retires = generation_retries
        logger.info(f"Loaded root model {root_model} using generation kwargs: {self._root_model_generation_kwargs}")

    def _sample_backstories(self, alignment_target: Optional[type[AlignmentTarget]] = None) -> List[Backstory]:
        if alignment_target is None or len(alignment_target.kdma_values) == 0: # type: ignore
            # There's no alignment target, so randomly sample a set of backstories
            return random.sample(self._backstories, self._backstory_panel_size)

        # Map the alignment target to a probe
        alignment_target_dict = alignment_target.to_dict()
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
        sampled_backstories = [b[0] for b in backstories_with_values[:self._backstory_panel_size]]
        self._backstory_alignment_cache[probe_values_str] = sampled_backstories

        # Return the top N backstories
        return sampled_backstories

    def _get_model_response(self, prompt: str, choices: List[str], prefix: Optional[str] = None) -> Choice:
        # Tokenize the prompt
        input_ids = self._root_model_tokenizer.encode(prompt, return_tensors="pt").to(self._root_model.device)  # type: ignore

        # Generate a response
        response = None
        with torch.no_grad():
            for retry_index in range(self._generation_retires):
                try:
                    stopping_criteria = StoppingCriteriaList(
                        [
                            StopStringCriteria(
                                tokenizer=self._root_model_tokenizer,
                                stop_strings=[
                                    "\n",
                                    "Question",
                                    "}",
                                ],
                            ),
                        ]
                    )

                    output = self._root_model.generate(
                        input_ids,
                        stopping_criteria=stopping_criteria,
                        **self._root_model_generation_kwargs,
                    )
                    # Decode the response
                    response = self._root_model_tokenizer.decode(output[0], skip_special_tokens=True)
                    response = response.replace(prompt, "").strip()  # Remove the prompt from the response
                    if prefix is not None:
                        response = prefix + response
                    logging.debug(f"Generated response: {response}")
                    reasoning, answer_idx, parse_method = Llama2SingleKDMAADM.parse_generated_output(
                        response, len(choices)
                    )
                    break
                except Exception as e:
                    logger.warning(
                        f"Persona ADM failed to generate a well-formed response on try {retry_index + 1}/{self._generation_retires}: {e}"
                    )
                    continue
            else:
                # Try parsing with semantics
                reasoning, answer_idx, parse_method = Llama2SingleKDMAADM.bert_similarity_parse(response, choices)
                if answer_idx is None:
                    raise ResponseGenerationError(
                        f"Failed to generate a response after {self._generation_retires} retries"
                    )

        return {"choice": answer_idx, "info": {"reasoning": reasoning, "parse_method": parse_method}}

    def choose_action(
        self,
        scenario_state: State,
        available_actions: list[Action],
        alignment_target: Optional[type[AlignmentTarget]],
        **kwargs,
    ) -> Action:
        # Lazy load the JINJA templates for the ADM
        from align_system.prompt_engineering.personas import probe_template

        # Bootstrap sample a population of backstories based on the alignment target
        backstory_population = self._sample_backstories(alignment_target)

        # For each backstory, sample an action/choice
        responses = []
        probe_str = None
        for backstory in backstory_population:
            probe_str = probe_template.render(
                scenario_state=scenario_state,
                available_actions=[str(action.unstructured or "") for action in available_actions],
                question_number=2 + len(backstory["probes"]),
                backstory=backstory,
            )
            response = self._get_model_response(
                probe_str, [str(action.unstructured or "") for action in available_actions], prefix='{"Reasoning": "'
            )
            responses.append(response)

        # Vote on the actions
        action_votes = {}
        for response in responses:
            action_votes[response["choice"]] = action_votes.get(response["choice"], 0) + 1

        # Choose the action with the most votes
        max_votes = max(action_votes.values())
        max_vote_actions = [action for action, votes in action_votes.items() if votes == max_votes]
        chosen_action = random.choice(max_vote_actions)

        # Get the reasonings for the chosen action
        chosen_reasonings = [response["info"]["reasoning"] for response in responses if response["choice"] == chosen_action]

        action_to_take = available_actions[chosen_action]
        action_to_take.justification = "\n\n".join([f'(Expert {i}): {cr}' for i, cr in enumerate(chosen_reasonings)])

        logging.info(f'Scenario: {probe_str}')
        logging.info(f"Chose action: {action_to_take.unstructured} with {max_votes} votes")
        logging.info(f"Justification: {action_to_take.justification}")

        if action_to_take.action_type == ActionTypeEnum.APPLY_TREATMENT:
            # If the additional required fields are already populated
            # for the action, don't need ask the LLM again
            if action_to_take.parameters is None or not {"treatment", "location"}.issubset(
                action_to_take.parameters.keys()
            ):
                action_to_take = self.populate_treatment_parameters(
                    scenario_state, action_to_take, alignment_target, **kwargs
                )
        elif action_to_take.action_type == ActionTypeEnum.TAG_CHARACTER:
            # If the additional required fields are already populated
            # for the action, don't need ask the LLM again
            if (
                action_to_take.character_id is None
                or action_to_take.parameters is None
                or not {"category"}.issubset(action_to_take.parameters.keys())
            ):
                action_to_take = self.populate_tagging_parameters(
                    scenario_state, action_to_take, alignment_target, **kwargs
                )
        elif action_to_take.action_type in {
            ActionTypeEnum.CHECK_ALL_VITALS,
            ActionTypeEnum.CHECK_PULSE,
            ActionTypeEnum.CHECK_RESPIRATION,
            ActionTypeEnum.MOVE_TO_EVAC,
        }:
            # These actions require a `character_id`
            if action_to_take.character_id is None:
                action_to_take = self.generic_populate_character_id(
                    scenario_state, action_to_take, alignment_target, **kwargs
                )

        return action_to_take

    def populate_treatment_parameters(self, scenario_state, treatment_action, alignment_target, **kwargs):
        from swagger_client.models import ActionTypeEnum

        from align_system.prompt_engineering.common import prepare_treatment_selection_prompt
        from align_system.utils import get_swagger_class_enum_values

        assert treatment_action.action_type == ActionTypeEnum.APPLY_TREATMENT

        character_id = treatment_action.character_id
        if character_id is None:
            # Need to populate character_id on treatment action
            treatment_action = self.generic_populate_character_id(
                scenario_state, treatment_action, alignment_target, **kwargs
            )

            character_id = treatment_action.character_id

        matching_characters = [c for c in scenario_state.characters if c.id == character_id]

        assert len(matching_characters) == 1

        character_to_treat = matching_characters[0]

        available_supplies = [s for s in scenario_state.supplies if s.quantity > 0]

        if isinstance(character_to_treat.vitals, dict):
            vitals_dict = character_to_treat.vitals
        else:
            vitals_dict = character_to_treat.vitals.to_dict()

        treatment_prompt = prepare_treatment_selection_prompt(
            character_to_treat.unstructured, vitals_dict, [s.to_dict() for s in available_supplies]
        )

        for _ in range(kwargs.get("answer_attempts", 5)):
            treatment_dialog = build_multiple_choice_dialog(
                treatment_prompt,
                [s.to_dict() for s in available_supplies],
                json_format=TREATMENT_MULTIPLE_CHOICE_JSON_FORMAT,
            )

            logger.debug("[bold]*TREATMENT DIALOG*[/bold]", extra={"markup": True})
            log_dialog(treatment_dialog)

            raw_treatment_response, _ = self.respond_to_dialog(treatment_dialog)

            logger.info(f"** ADM raw treatment response: {raw_treatment_response}")

            parsed_treatment_output = Llama2SingleKDMAADM.attempt_generic_parse(
                raw_treatment_response, ["Reasoning", "Answer", "Location"]
            )

            if parsed_treatment_output is not None:
                try:
                    treatment_idx = int(parsed_treatment_output["Answer"])
                except ValueError:
                    logger.warning("** Treatment index not an integer, retrying!")
                    continue

                if len(available_supplies) <= treatment_idx:
                    logger.info("** Selected treatment_idx out of range of " "available treatment options, retrying!")
                    continue

                treatment = available_supplies[treatment_idx].type

                treatment_location = parsed_treatment_output["Location"]

                if treatment_action.parameters is None:
                    treatment_action.parameters = {}

                treatment_action.parameters["treatment"] = treatment

                valid_treatment_locations = get_swagger_class_enum_values(InjuryLocationEnum)

                if not isinstance(treatment_location, str):
                    # If type is int, could be an index into the
                    # action_to_take)locations provided in the system
                    # action_to_take)prompt, consider handling in the
                    # action_to_take)future
                    logger.warning("*** Treatment location value is not a string" ", retrying!")
                    continue
                if treatment_location in valid_treatment_locations:
                    treatment_action.parameters["location"] = treatment_location
                else:
                    # Ensure that the treatment location is valid
                    _, treatment_loc_idx, _ = Llama2SingleKDMAADM.bert_similarity_parse(
                        treatment_location, valid_treatment_locations
                    )
                    if treatment_loc_idx is None:
                        logger.warning(f"** Treatment location ({treatment_location}) not valid, retrying!")
                        continue

                    treatment_action.parameters["location"] = valid_treatment_locations[treatment_loc_idx]

                break
            else:
                logger.info("** Failed to parse treatment")

        return treatment_action

    def populate_tagging_parameters(self, scenario_state, tagging_action, alignment_target, **kwargs):
        from swagger_client.models import ActionTypeEnum, CharacterTagEnum

        from align_system.prompt_engineering.common import prepare_tagging_selection_prompt
        from align_system.utils import get_swagger_class_enum_values

        assert tagging_action.action_type == ActionTypeEnum.TAG_CHARACTER
        # Ask the system to specify which triage tag to apply

        untagged_characters = [c for c in scenario_state.characters if c.tag is None]

        tagging_prompt = prepare_tagging_selection_prompt(
            [c.to_dict() for c in untagged_characters], get_swagger_class_enum_values(CharacterTagEnum)
        )

        for _ in range(kwargs.get("answer_attempts", 5)):
            tagging_dialog = build_multiple_choice_dialog(
                tagging_prompt,
                [c.unstructured.strip() for c in untagged_characters],
                json_format=TAGGING_MULTIPLE_CHOICE_JSON_FORMAT,
            )

            logger.debug("[bold]*TAGGING DIALOG*[/bold]", extra={"markup": True})
            log_dialog(tagging_dialog)

            raw_tagging_response, _ = self.respond_to_dialog(tagging_dialog)

            logger.info(f"** ADM raw tagging response: {raw_tagging_response}")

            parsed_tagging_output = Llama2SingleKDMAADM.attempt_generic_parse(
                raw_tagging_response, ["Reasoning", "Answer", "Tag"]
            )

            if parsed_tagging_output is not None:
                if len(untagged_characters) == 1:
                    logger.debug("** Force selecting only available character")
                    character_idx = 0
                else:
                    character_idx = parsed_tagging_output["Answer"]

                    if not isinstance(character_idx, int):
                        logger.warning(f"** character_idx ({character_idx}) not an integer, retrying!")
                        continue

                    if len(untagged_characters) <= character_idx:
                        logger.info(
                            "** Selected character_idx out of range of " "available treatment options, retrying!"
                        )
                        continue

                character_to_tag_id = untagged_characters[character_idx].id

                tag = parsed_tagging_output["Tag"]
                if not isinstance(tag, str):
                    logger.warning(f"** Selected tag ({tag}) not of type string, retrying!")
                    continue

                valid_tags = get_swagger_class_enum_values(CharacterTagEnum)
                if tag not in valid_tags:
                    logger.warning(f"** Selected tag ({tag}) is not a valid tag, retrying!")
                    continue

                # Populate required parameters for tagging action
                tagging_action.character_id = character_to_tag_id

                if tagging_action.parameters is None:
                    tagging_action.parameters = {}

                tagging_action.parameters["category"] = tag

                break
            else:
                logger.info("** Failed to parse tagging")

        return tagging_action

    def generic_populate_character_id(self, scenario_state, initial_action, alignment_target, **kwargs):
        from swagger_client.models import ActionTypeEnum

        from align_system.prompt_engineering.common import prepare_character_selection_prompt

        character_selection_prompt = prepare_character_selection_prompt(initial_action)

        filtered_characters = []
        for c in scenario_state.characters:
            if initial_action.action_type in {
                ActionTypeEnum.CHECK_ALL_VITALS,
                ActionTypeEnum.CHECK_PULSE,
                ActionTypeEnum.CHECK_RESPIRATION,
            }:
                # Don't allow the ADM to check vitals on
                # a character that's already been "visited"
                if c.visited:
                    continue

            filtered_characters.append(c)

        for _ in range(kwargs.get("answer_attempts", 5)):
            character_selection_dialog = build_multiple_choice_dialog(
                character_selection_prompt, [c.unstructured.strip() for c in filtered_characters]
            )

            logger.debug("[bold]*CHARACTER SELECTION DIALOG*[/bold]", extra={"markup": True})
            log_dialog(character_selection_dialog)

            raw_character_selection_response, _ = self.respond_to_dialog(character_selection_dialog)

            logger.info(f"** ADM raw character_selection response: {raw_character_selection_response}")

            parsed_character_selection_output = Llama2SingleKDMAADM.attempt_generic_parse(
                raw_character_selection_response, ["Reasoning", "Answer"]
            )

            if parsed_character_selection_output is not None:
                if len(filtered_characters) == 1:
                    logger.debug("** Force selecting only available character")
                    character_idx = 0
                else:
                    character_idx = parsed_character_selection_output["Answer"]

                    if not isinstance(character_idx, int):
                        logger.warning(f"** character_idx ({character_idx}) not an integer" ", retrying!")
                        continue

                    if len(filtered_characters) <= character_idx:
                        logger.warning(
                            "** Selected character_idx out of range of " "available treatment options, retrying!"
                        )
                        continue

                character_id = filtered_characters[character_idx].id

                # Populate required parameters for character_selection action
                initial_action.character_id = character_id

                break
            else:
                logger.info("** Failed to parse character selection")

        return initial_action

    def respond_to_dialog(self, dialog, prefix=None):
        inference_pair = {}
        if prefix is None:
            prefix = '{"Reasoning": "'

        try:
            prompt_tokens = [self._root_model_tokenizer.apply_chat_template(dialog, tokenize=True)]
            inference_pair["input"] = self._root_model_tokenizer.apply_chat_template(dialog, tokenize=False)
        except TemplateError:
            prompt_tokens = self._build_prompt_tokens_from_dialog(dialog, inference_pair)
        prompt_length = len(prompt_tokens[0])

        if prefix is not None:
            prefix_tokens = self._root_model_tokenizer.encode(prefix, add_special_tokens=False)
            prompt_tokens[0] += prefix_tokens  # type: ignore

        prompt_tokens = torch.tensor(prompt_tokens).to(self._root_model.device)

        outputs = self._root_model.generate(
            prompt_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            **self._root_model_generation_kwargs,
        )

        # Print the generated model output
        generated_output = self._root_model_tokenizer.decode(outputs.sequences[0][prompt_length:])
        inference_pair["output"] = generated_output

        return generated_output, inference_pair

    def _build_prompt_tokens_from_dialog(self, dialog, inference_pair):
        new_dialog = []
        for message in dialog:
            if message["role"] == "system":
                message["role"] = "user"

            if not new_dialog:
                new_dialog.append(message)
                continue

            last_message = new_dialog[-1]
            if last_message["role"] == message["role"]:
                last_message["content"] += "\n\n" + message["content"]
            else:
                new_dialog.append(message)
        dialog = new_dialog
        logger.info("INPUT\n%s", dialog)
        result = [self._root_model_tokenizer.apply_chat_template(dialog, tokenize=True)]
        inference_pair["input"] = self._root_model_tokenizer.apply_chat_template(dialog, tokenize=False)

        return result
