from align_system.prompt_engineering.decision_flow_prompts.mta_prompt import (
    math_variables,
    math_extraction,
    math_attribute,
    math_filter,
    math_express,
)

import re
import json
from openai import OpenAI
from jinja2 import Environment, FileSystemLoader

from align_system.algorithms.decision_flow_adm.secret import OPENAI_API_KEY, OPENAI_API_BASE

class DecisionFlowStages:

    def __init__(self, target_bias, task, choice, state, probe, system_message_keys, model, model_path, temperature):
        self.target_bias = target_bias
        self.task = task
        self.choice = choice
        self.state = state
        self.probe = probe
        self.model = model
        self.model_path = model_path
        self.temperature = temperature
        self.system_message_keys = f"{list(system_message_keys.values())[0]} {list(system_message_keys.keys())[0]}"
        self.information = {}
        self.actions = [
            "Variables",
            "Extraction",
            "Attribute",
            "Filter",
            "Objective",
            "Express",
        ]

    def __call__(self):
        for action in self.actions:
            if action == "Variables":
                self.variables = None
                prompt = math_variables.format(
                    task=self.task + self.probe, choices=self.choice
                )
                response = model_generate_output(prompt, model=self.model, model_path=self.model_path, temperature=self.temperature)
                try:
                    self.variables = extractJSONToDict(response)
                    self.variables = self.variables["variables"]
                except Exception as e:
                    print(e)
                    self.variables = response
            elif action == "Extraction":
                numAttempts = 0
                self.extraction = []
                prompt = math_extraction.format(task=self.task, variable=self.variables)
                response = model_generate_output(
                    prompt,
                    model=self.model,
                    model_path=self.model_path,
                    temperature=self.temperature,
                )
                try:
                    self.extraction = extractJSONToDict(response)
                    self.extraction = self.extraction["information"]
                    for extracted_information in self.extraction:
                        self.information[extracted_information] = {}
                except Exception as e:
                    print(e)
                    self.extraction = response
            elif action == "Attribute":
                single_attribute = None
                prompt = math_attribute.format(
                    variable=self.variables,
                    information=self.extraction,
                    target_bias=self.system_message_keys,
                )
                response = model_generate_output(
                    prompt,
                    model=self.model,
                    model_path=self.model_path,
                    temperature=self.temperature,
                )
                self.attribute = []
                try:
                    single_attribute = extractJSONToDict(response)
                    for attribute_ in single_attribute["Variable"]:
                        variable_attribute = attribute_["Variable"]
                        for attribute_value in attribute_["Attribute"]:
                            self.attribute.append(
                                {
                                    "Variable": variable_attribute,
                                    "Attribute": attribute_value["Attribute"],
                                    "Value": attribute_value["Value"],
                                }
                            )
                except Exception as e:
                    print(e)
                    self.attribute = response
            elif action == "Filter":
                try:
                    for index, attribute_ in enumerate(self.attribute):
                        if attribute_["Variable"] == "Environment":
                            continue
                        single_filter = None
                        information_attribute = f"Attribute: {attribute_['Attribute']}"
                        prompt = math_filter.format(
                            information=information_attribute,
                            target_bias=self.target_bias,
                        )
                        response = model_generate_output(
                            prompt,
                            model=self.model,
                            model_path=self.model_path,
                            temperature=self.temperature,
                        )
                        try:
                            single_filter = extractJSONToDict(response)

                            attribute_["Value"] = (
                                list(attribute_["Value"])
                                if isinstance(attribute_["Value"], set)
                                else attribute_["Value"]
                            )
                            self.attribute[index]["Weight"] = single_filter[
                                "Weight"
                            ]
                            self.attribute[index]["Explanation"] = single_filter[
                                "Explanation"
                            ]
                            break
                        except Exception as e:
                            print(e)
                            numAttempts += 1
                except Exception as e:
                    print(e)
            elif action == "Objective":
                try:
                    self.objective = []
                    objective_function_attribute = []
                    for group in self.attribute:
                        # You can adjust this to make a better filter
                        if group["Weight"] > 0.3:
                            self.objective.append(group)
                            objective_function_attribute.append(
                                {
                                    "Variable": group["Variable"],
                                    "Attribute": group["Attribute"],
                                    "Weight": group["Weight"],
                                    "Explanation": group["Explanation"],
                                }
                            )
                    self.attribute = self.objective
                    self.objective = "The final formula to be calculated is "
                    for attribute_ in objective_function_attribute:
                        variable = attribute_["Variable"]
                        weight = attribute_["Weight"]
                        attribute = attribute_["Attribute"]
                        self.objective += f"{weight} * ({attribute}) of ({variable}) + "
                except Exception as e:
                    print(e)
                    self.objective = "weight * attribute of variable"
            elif action == "Express":
                self.structure = {}
                self.structure["variables"] = self.variables
                self.structure["objective_function"] = self.objective
                self.structure["attribute"] = []
                try:
                    for filter_ in self.attribute:
                        if filter_["Variable"] == "Environment":
                            continue
                        else:
                            self.structure["attribute"].append(
                                {
                                    "Variable": filter_["Variable"],
                                    "Attribute": filter_["Attribute"],
                                    "Value": filter_["Value"],
                                }
                            )
                except:
                    self.structure["attribute"] = self.attribute

                self.structure["constraints"] = []
                try:
                    for key, value in self.information.items():
                        if value["constraints"] is not None:
                            self.structure["constraints"].append(
                                {
                                    "description": key,
                                    "constraints": value["constraints"],
                                }
                            )
                except:
                    self.structure["constraints"] = []
                prompt = math_express.format(structure=self.structure)
                response = model_generate_output(
                    prompt,
                    model=self.model,
                    model_path=self.model_path,
                    temperature=self.temperature,
                )
                try:
                    self.express = extractJSONToDict(response)
                except:
                    self.express = response

def model_generate_output(prompt, model, model_path, temperature):
    try:
        env = Environment(loader=FileSystemLoader("."))
        template = env.get_template("chat_template.jinja")
        messages = [
            {"role": "assistant", "content": "You are a very helpful assistant"},
            {"role": "user", "content": prompt},
        ]
        prompt = template.render(messages=messages)
        client = OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_API_BASE,
        )

        response = client.completions.create(
            model=model_path,
            prompt=prompt,
            stream=False,
            temperature=temperature,
            max_tokens=4096,
        )
    except Exception as e:
        print(f"An error occurred: {e}")
    return response.choices[0].text

def extractJSONToDict(response: str, language_identifer_optional=True):
    # Define the pattern
    if language_identifer_optional:
        pattern = r"```(?:json)?\s*(.*?)```"
    else:
        pattern = r"```json\s*(.*?)```"
    matches = re.findall(pattern, response, flags=re.DOTALL)

    json_str = None
    if matches:
        json_str = matches[0]
    else:
        print("No JSON found in the text.")
        print(json_str)
        raise Exception("No JSON Found")

    selection = ""
    if json_str:
        try:
            selection = json.loads(json_str)
        except:
            raise Exception("Unable to instantiate JSON.")

    return selection
