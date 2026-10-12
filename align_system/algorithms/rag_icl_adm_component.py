import json

from jsonschema import validate

from align_system.algorithms.abstracts import ADMComponent
from align_system.data_models.dialog import DialogElement
from align_system.prompt_engineering.outlines_prompts import comparative_regression_json_schema
from align_system.utils.adm_utils import format_choices
from align_system.utils.alignment_utils import attributes_in_alignment_target
from align_system.utils.hydrate_state import p2triage_hydrate_scenario_state


class Phase2ExampleText:
    def __init__(self, scenario_description_template):
        self.scenario_description_template = scenario_description_template

    def __call__(self, record):
        state, _ = p2triage_hydrate_scenario_state(record['input'])
        return self.scenario_description_template(state)


class Phase2RegressionExampleFormatter:
    def __init__(self, scenario_description_template, prompt_template):
        self.scenario_description_template = scenario_description_template
        self.prompt_template = prompt_template

    def __call__(self, record, attribute):
        state, actions = p2triage_hydrate_scenario_state(record['input'])
        choices = format_choices([action.unstructured for action in actions], actions, state)
        scores = {choice: {'score': int(label[attribute.kdma] * attribute.factor)}
                  for choice, label in zip(choices, record['label'])
                  if label.get(attribute.kdma) is not None}
        if not scores:
            raise ValueError(f'Example has no labels for {attribute.kdma}')
        best_choice = max(scores, key=lambda choice: scores[choice]['score'])
        response = {'reasoning': f'The labeled {attribute.name} score is highest for {best_choice}.',
                    **scores}
        validate(response, json.loads(comparative_regression_json_schema(list(scores), attribute.factor)))
        prompt = self.prompt_template(
            scenario_description=self.scenario_description_template(state),
            choices=list(scores), attribute=attribute.name)
        return {'prompt': prompt, 'response': response}


class RAGICLADMComponent(ADMComponent):
    def __init__(self, retrievers, example_formatter, attributes, always_include_attributes=()):
        self.retrievers = retrievers
        self.example_formatter = example_formatter
        self.attributes = attributes
        self.always_include_attributes = always_include_attributes

    def run_returns(self):
        return 'icl_dialog_elements', 'icl_example_info'

    def run(self, scenario_state, alignment_target=None):
        target_names = attributes_in_alignment_target(alignment_target) if alignment_target is not None else []
        names = dict.fromkeys([*self.always_include_attributes, *target_names])
        dialogs, example_info = {}, {}
        for name in names:
            attribute = self.attributes[name]
            dialogs[name], example_info[name] = [], []
            for result in self.retrievers[name](scenario_state):
                example = self.example_formatter(result['record'], attribute)
                dialogs[name].extend([
                    DialogElement(role='user', content=example['prompt']),
                    DialogElement(role='assistant', content=json.dumps(example['response']))])
                example_info[name].append({'id': result['id'], 'similarity_score': result['score'], **example})
        return dialogs, example_info
