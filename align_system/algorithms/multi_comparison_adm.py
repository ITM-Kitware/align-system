import json
import yaml
import os

from align_system.algorithms.abstracts import AlignedDecisionMaker
from align_system.algorithms.lib.chat.chat_language_model import ChatLanguageModel


class Option:
    def __init__(self, model, scenario, probe, choice, characteristic, definition, template, log_file=None):
        self.model = model
        self.choice = choice
        self.scenario = scenario
        self.probe = probe
        self.characteristic = characteristic
        self.definition = definition
        self.template = template
        self.log_file = log_file
    
    def __lt__(self, other):
        assert self.scenario == other.scenario
        assert self.probe == other.probe
        
        for _ in range(10):
            response = self.model.generate_from_template(self.template, substitution_dicts={
                'scenario': self.scenario,
                'probe': self.probe,
                'characteristic': self.characteristic,
                'definition': self.definition,
                'option_0': self.choice,
                'option_1': other.choice
            }, log_file=self.log_file)[0]
            
            try:
                response = json.loads(response)
                assert 'decision' in response
                break
            except json.JSONDecodeError:
                continue
            
        return response['decision'] == 1


class MultiComparisonADM(ChatLanguageModel, AlignedDecisionMaker):
    
    def predict_kdma_values(self, sample, kdmas, **kwargs):
        scenario = sample['scenario']
        probe = sample['probe']
        choices = sample['choices']
        
        kdma_descriptions_file = kwargs['kdma_descriptions_file']
        kdma_descriptions_file = os.path.join(os.path.dirname(__file__), kdma_descriptions_file)
        # read yaml file
        with open(kdma_descriptions_file) as f:
            kdma_descriptions = yaml.load(f, Loader=yaml.FullLoader)
            
        comparison_template_file = kwargs['comparison_template_file']
        # make the path relative to the current file
        comparison_template_file = os.path.join(os.path.dirname(__file__), comparison_template_file)
        with open(comparison_template_file) as f:
            template = f.read()
        
        n = len(choices)
        # creates a linear range of points from 0 to 10
        # least aligned option is 0, most aligned option is 10
        points = [round(point/100) for point in range(0, 1000 + int(1000/n), int(1000/(n-1)))]
        predicted_kdmas = {}
        for target_kdma in kdmas:
            # use python's sort to sort the choices by alignment as determined by the language model
            # could be inconsistent due to random sampling in the language model
            # TODO implement self consistency to reduce varaibility in sorting
            sorted_choices = sorted([
                Option(
                    self,
                    scenario,
                    probe,
                    choice,
                    kdma_descriptions[target_kdma]['name'],
                    kdma_descriptions[target_kdma]['description'],
                    template,
                    kwargs['log_file']
                )
                for choice in choices
            ])

            sorted_choices = [choice.choice for choice in sorted_choices]
            predicted_kdmas[target_kdma] = [points[choices.index(choice)] for choice in sorted_choices]
        
        # transpose predicted_kdmas
        predicted_kdmas = [
            {
                target_kdma: predicted_kdmas[target_kdma][i]
                for target_kdma in kdmas
            }
            for i in range(len(choices))
        ]
        
        return predicted_kdmas # {target_kdma: [predicted_value_0, predicted_value_1, ...]}
        
    
    def __call__(self, sample, target_kdma_values, **kwargs):
        # find the most aligned choice
        kwargs['predict_kdma_values']['log_file'] = kwargs['log_file']
        predicted_kdmas = self.predict_kdma_values(sample, list(target_kdma_values.keys()), **kwargs['predict_kdma_values'])
        # target_kdma_values {target_kdma: target_value}
        
        # find most aligned choice based on mse
        mse = [] # [choice_0_mse, choice_1_mse, ...]
        for choice_idx in range(len(sample['choices'])):
            mse.append(sum([
                (predicted_kdmas[choice_idx][target_kdma] - target_kdma_values[target_kdma])**2
                for target_kdma in target_kdma_values
            ]))
        
        most_aligned_choice_idx = mse.index(min(mse))
        
        return {
            'choice': most_aligned_choice_idx,
            'predicted_kdmas': predicted_kdmas
        }
            
        
        
        
        