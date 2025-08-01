import json
import os

from tqdm import tqdm

from align_system.prompt_engineering.outlines_prompts import (
    baseline_system_prompt
)

from align_system.algorithms.decision_flow_adm.openai_open_source_adm import OpenSingleDMA
from align_system.algorithms.decision_flow_adm.decision_flow_stages import DecisionFlowStages

class DecisionFlowInferenceEngine:
    def __init__(self, model_name):
        self.model_name = model_name

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

    def _generate_single_output(self, sample, model, model_path, method, alignment, labels, temperature):
        prompt = sample["scenario"]
        if sample["state"] is not None:
            state = sample["state"]
            probe = state + sample["probe"]
        else:
            state = None
            probe = sample["probe"]

        choices = sample["choices"]
        target_kdma = next(
            iter(next(iter(filter(lambda x: len(x) > 0, labels))))
        )  # get the frist key of the first label that is not empty


        def __load_system_message(
            alignment=None,
            system_messages_path="align_system/prompt_engineering/single_kdma_adm_system_messges"
        ):
            kdma_remapping = {
                "basicknowledge": "basic_knowledge",
                "protocolfocus": "protocol_focus",
                "riskaversion": "risk_aversion",
                "moraldeservingness": "moral_deservingness",
                "continuationofcare": "continuation_of_care",
                "livesaved": "lives_saved",
                "timepressure": "time_pressure",
            }
            if alignment is None:
                system_message = baseline_system_prompt()
            else:
                sorted_kdmas = sorted(alignment.keys())

                alignment_string = "-".join(
                    "{}-{}".format(alignment[k], kdma_remapping.get(k, k)) for k in sorted_kdmas
                )

                file_name = f"{alignment_string}.txt"

                with open(os.path.join(system_messages_path, file_name), "r") as f:
                    system_message = f.read()
            return system_message

        if alignment == "unaligned":
            pass
        else:
            system_message_keys = {target_kdma: alignment}
            system_message = __load_system_message(system_message_keys)

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
