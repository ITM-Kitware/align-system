import json
import random

import outlines
from outlines.samplers import MultinomialSampler
from rich.highlighter import JSONHighlighter

from align_system.utils import logging
from align_system.utils import adm_utils
from align_system.utils.voting import (
    simple_majority_vote,
)
from align_system.prompt_engineering.outlines_prompts import (
    action_selection_prompt,
    scenario_state_description_1,
    action_choice_json_schema,
)
from align_system.algorithms.outlines_adm import OutlinesTransformersADM
from align_system.algorithms.lib.persona.provider import PersonaProvider

log = logging.getLogger(__name__)
JSON_HIGHLIGHTER = JSONHighlighter()


class OutlinesPersonaADM(OutlinesTransformersADM):
    def __init__(self,
                 model_name,
                 device='auto',
                 sampler=MultinomialSampler(),
                 persona_provider=PersonaProvider(),
                 n_personas=3,
                 filter_probes_to_target_kdmas=True,
                 **kwargs):
        super().__init__(model_name,
                         device=device,
                         baseline=False,
                         sampler=sampler,
                         **kwargs)

        self.persona_provider = persona_provider
        self.n_personas = n_personas
        self.filter_probes_to_target_kdmas = filter_probes_to_target_kdmas

    def top_level_choose_action(self,
                                scenario_state,
                                available_actions,
                                alignment_target,
                                generator_batch_size=5,
                                shuffle_choices=False,
                                **kwargs):
        persona_dialogs = self.persona_provider.get_persona_dialogs(
            alignment_target,
            self.n_personas,
            filter_probes_to_target_kdmas=self.filter_probes_to_target_kdmas)

        scenario_description = scenario_state_description_1(scenario_state)
        # Important that the choices stay in the same order as the
        # available actions as we'll use the selected index later to
        # map to the corresponding action
        choices = adm_utils.format_choices(
            [a.unstructured for a in available_actions],
            available_actions,
            scenario_state
            )
        if shuffle_choices:
            shuffled_choices = random.sample(choices, len(choices))
            prompt = action_selection_prompt(scenario_description, shuffled_choices)
        else:
            prompt = action_selection_prompt(scenario_description, choices)

        dialogs = []
        for persona_dialog in persona_dialogs:
            dialogs.append(
                [*persona_dialog, {'role': 'user', 'content': prompt}])

        dialog_texts = [self.dialog_to_prompt(d) for d in dialogs]

        for i, dialog_text in enumerate(dialog_texts):
            log.info(f"[bold]*PERSONA {i+1} DIALOG PROMPT*[/bold]",
                     extra={"markup": True})
            log.info(dialog_text)

        # Need to set the whitespace_pattern to prevent the state
        # machine from looping indefinitely in some cases, see:
        # https://github.com/outlines-dev/outlines/issues/690#issuecomment-2102291934
        generator = outlines.generate.json(
            self.model,
            action_choice_json_schema(json.dumps(choices)),
            sampler=self.sampler,
            whitespace_pattern=r"[ ]?")

        responses = self.run_in_batches(generator, dialog_texts, generator_batch_size)
        responses_choices = [r['action_choice'] for r in responses]

        votes = simple_majority_vote(choices, responses_choices)

        log.explain("[bold]*VOTES*[/bold]",
                    extra={"markup": True})
        log.explain(votes, extra={"highlighter": JSON_HIGHLIGHTER})

        # Take top choice by score (votes is a dictionary of choice: score)
        top_choice, top_choice_score = max(votes.items(), key=lambda x: x[1])
        # Just taking first justification from the responses
        # where the top choice was selected.  A better approach might
        # be to somehow summarized all justifications with the
        # matching choice.
        top_choice_justification = ""
        top_choice_response = None
        top_choice_dialog = None
        for response, dialog in zip(responses, dialogs):
            if response['action_choice'] == top_choice:
                top_choice_justification = response['detailed_reasoning']
                top_choice_response = response
                top_choice_dialog = dialog
                break

        selected_choice_idx = choices.index(top_choice)

        log.info("[bold]*STRUCTURED RESPONSE*[/bold]",
                 extra={"markup": True})
        log.info(top_choice_response, extra={"highlighter": JSON_HIGHLIGHTER})

        action_to_take = available_actions[selected_choice_idx]
        action_to_take.justification = top_choice_justification

        return action_to_take, top_choice_dialog
