import json
import os
import random
from typing_extensions import override
import yaml

import outlines
from outlines.samplers import MultinomialSampler
from rich.highlighter import JSONHighlighter
from swagger_client.models import (
    KDMAValue
)
import torch
from tqdm import tqdm

from align_system.algorithms.decision_flow_adm.openai_open_source_adm import OpenSingleDMA
from align_system.algorithms.decision_flow_adm.decision_flow_stages import DecisionFlowStages
from align_system.algorithms.outlines_adm import OutlinesTransformersADM
from align_system.prompt_engineering.outlines_prompts import (
    baseline_system_prompt,
    scenario_state_unstructured,
    action_selection_prompt,
)
from align_system.utils import logging
from align_system.utils import adm_utils

log = logging.getLogger(__name__)
JSON_HIGHLIGHTER = JSONHighlighter()

class DecisionFlowADM(OutlinesTransformersADM):
    def __init__(self,
                 model_name,
                 device='auto',
                 baseline=False,
                 sampler=MultinomialSampler(),
                 scenario_description_template=scenario_state_unstructured,
                 action_selection_prompt_template=action_selection_prompt,
                 baseline_system_prompt=baseline_system_prompt,
                 outlines_seed=None,
                 **kwargs):

        super().__init__(
            model_name=model_name,
            device=device,
            baseline=baseline,
            sampler=sampler,
            scenario_description_template=scenario_description_template,
            action_selection_prompt_template=action_selection_prompt_template,
            baseline_system_prompt=baseline_system_prompt,
            outlines_seed=outlines_seed,
            **kwargs,
        )


    def _generate_outputs(self, dataset, method, model, model_path, results_dir, alignment, temperature):
        detailed_path = os.path.join(results_dir, "detailed_infor.json")
        iol_path = os.path.join(results_dir, "input_output_labels.json")
        if os.path.exists(detailed_path) and os.path.exists(iol_path):
            with open(detailed_path, "r") as f:
                detailed = json.load(f)
            with open(iol_path, "r") as f:
                in_out_labels = json.load(f)
            outputs = [item["output"] for item in in_out_labels]
            start_index = len(outputs)
            print(f"Resuming from index {start_index}")

        else:
            start_index = 0
            outputs = []
            detailed = []

        for count in tqdm(range(start_index, len(dataset))):
            input_, label = dataset[count]
            try:
                if len(label) == 0 or max(map(len, label)) == 0:
                    output = {"choice": None, "info": "no_label"}
                    detail = {"input": input_, "label": label, "info": "no_label"}
                else:
                    output, detail = self._generate_single_output(
                        input_, model, model_path, method, alignment, label, temperature,
                    )

                outputs.append(output)
                detailed.append(detail)

                # Save intermediate results
                in_out_labels = []
                for idx, (generated_output, (input_i, label_i)) in enumerate(
                    zip(outputs, dataset[: count + 1])
                ):
                    in_out_labels.append(
                        {
                            "input": input_i,
                            "label": label_i,
                            "output": generated_output,
                        }
                    )

                detailed_infor = []
                for detail, (input_, label) in zip(detailed, dataset[: count + 1]):
                    detailed_infor.append(
                        {
                            "input": input_,
                            "label": label,
                            "detailed_infor": detail,
                        }
                    )

                with open(os.path.join(results_dir, "input_output_labels.json"), "w") as f:
                    json.dump(in_out_labels, f, indent=4)
                with open(os.path.join(results_dir, "detailed_infor.json"), "w") as f:
                    json.dump(detailed_infor, f, indent=4)
            except Exception as e:
                print(f"Error processing case {count}: {e}")
                outputs.append({"choice": None, "info": "error", "error_message": str(e)})
                detailed.append({"input": input_, "label": label, "error": str(e)})
            count += 1

        return outputs, detailed

    @staticmethod
    def _static_state_to_top_level_prompt(action_selection_prompt_template, scenario_description, scenario_state, actions):
        """
        Generate prompt dialog based on given state and actions
        """
        choices = adm_utils.format_choices(
            [a.unstructured for a in actions],
            actions,
            scenario_state
        )

        prompt = action_selection_prompt_template(scenario_description, choices)

        return prompt, choices

    @override
    @staticmethod
    def get_dialogs(scenario_state,
                    available_actions,
                    alignment_target,
                    num_positive_samples=1,
                    num_negative_samples=0,
                    kdma_descriptions_map='align_system/prompt_engineering/kdma_descriptions.yml',
                    shuffle_choices=True,
                    baseline=False,
                    scenario_description_template=scenario_state_unstructured,
                    action_selection_prompt_template=action_selection_prompt,
                    baseline_system_prompt=baseline_system_prompt,
                    **kwargs):

        scenario_description = scenario_state_unstructured(scenario_state)
        choices = adm_utils.format_choices(
            [a.unstructured for a in available_actions],
            available_actions,
            scenario_state
        )

        if not baseline and alignment_target is not None:
            kdma_values = alignment_target.kdma_values
            if len(kdma_values) != 1:
                raise RuntimeError("This ADM assumes a single KDMA target, aborting!")
            kdma_value = kdma_values[0]
            if isinstance(kdma_value, KDMAValue):
                kdma_value = kdma_value.to_dict()

            kdma = kdma_value['kdma']
            value = kdma_value['value']
            # Assumption here is that KDMA values range from 0-1
            negative_value = 1 - value
            # Get kdma names and descriptions
            with open(kdma_descriptions_map, 'r') as f:
                kdma_descriptions = yaml.load(f, Loader=yaml.FullLoader)
            name = kdma_descriptions[kdma]['name']

            positive_system_prompt = OutlinesTransformersADM.kdma_value_to_system_prompt(kdma, value)
            negative_system_prompt = OutlinesTransformersADM.kdma_value_to_system_prompt(kdma, negative_value)

            if positive_system_prompt is None:
                raise RuntimeError("Couldn't find system prompt for kdma: {}, and "
                                   "value: {}.".format(kdma, value))
            if negative_system_prompt is None:
                raise RuntimeError("Couldn't find system prompt for kdma: {}, and "
                                   "value: {}.".format(kdma, negative_value))
        else:
            positive_system_prompt = baseline_system_prompt()
            negative_system_prompt = None

        positive_dialogs = []
        for _ in range(num_positive_samples):
            dialog = [{'role': 'system', 'content': positive_system_prompt}]
            positive_dialogs.append(dialog)

        negative_dialogs = []
        for _ in range(num_negative_samples):
            dialog = [{'role': 'system', 'content': negative_system_prompt}]
            negative_dialogs.append(dialog)

        return {"scenario_description": scenario_description,
                "choices": choices,
                "positive_system_prompt": positive_system_prompt,
                "negative_system_prompt": negative_system_prompt,
                "positive_dialogs": positive_dialogs,
                "negative_dialogs": negative_dialogs}

    def top_level_choose_action(self,
                                scenario_state,
                                available_actions,
                                alignment_target,
                                num_positive_samples=1,
                                num_negative_samples=0,
                                generator_batch_size=5,
                                kdma_descriptions_map='align_system/prompt_engineering/kdma_descriptions.yml',
                                reasoning_max_length=512,
                                generator_seed=-1,
                                max_generator_tokens=-1,
                                shuffle_choices=True,
                                **kwargs):
        if self.baseline and num_negative_samples > 0:
            raise RuntimeError("No notion of negative samples for baseline run")
        if self.baseline and "incontext" in kwargs and kwargs["incontext"]["number"] > 0:
            raise RuntimeError("No notion of incontext examples for baseline run")

        dialogs_data = DecisionFlowADM.get_dialogs(
            scenario_state,
            available_actions,
            alignment_target,
            num_positive_samples,
            num_negative_samples,
            kdma_descriptions_map,
            shuffle_choices,
            baseline=self.baseline,
            scenario_description_template=self.scenario_description_template,
            action_selection_prompt_template=self.action_selection_prompt_template,
            baseline_system_prompt=self.baseline_system_prompt,
        )
        choices = dialogs_data["choices"]
        positive_dialogs = dialogs_data["positive_dialogs"]
        negative_dialogs = dialogs_data["negative_dialogs"]

        for system_message in positive_dialogs:
            _generate_single_output()


        def _generate_single_output(sample, model, model_path, method, alignment, labels, temperature):
            adm = OpenSingleDMA(method, model, model_path, temperature)

            if method == "decisionflow" and alignment != "unaligned":
                result = DecisionFlowStages(
                    target_bias=system_message,
                    task=prompt,
                    choice=choices,
                    state=state,
                    probe=probe,
                    system_message_keys=system_message_keys,
                    model=model,
                    model_path=model_path,
                    temperature=temperature,
                )
                result()
                attribute_reason = []
                try:
                    for attribute_ in result.attribute:
                        variable = attribute_["Variable"]
                        attribute = attribute_["Attribute"]
                        value = attribute_["Value"]
                        attribute_reason.append(
                            {
                                "Variable": variable,
                                "Attribute": attribute,
                                "Value": value,
                            }
                        )
                except Exception as e:
                    attribute_reason = result.attribute
                response = adm(
                    sample,
                    labels=labels,
                    alignment=alignment,
                    structure=result.express,
                    attribute=attribute_reason,
                )
                return response, {
                    "variables": result.variables,
                    "extraction": result.extraction,
                    "information": result.information,
                    "attribute": result.attribute,
                    "express": result.express,
                }
            else:
                response = adm(
                    sample,
                    labels=labels,
                    alignment=alignment,
                    structure="",
                    attribute=[],
                )
                return response, {}
