import json
import yaml
import os
from typing import Union, List, Dict, Tuple, Optional, TextIO
from align_system.algorithms.lib.chat.chat_language_model import ChatLanguageModel
from align_system.algorithms.lib.aligned_decision_maker import AlignedDecisionMaker

class ChatKDMAPredictingADM(ChatLanguageModel, AlignedDecisionMaker):
    
    def predict_outcomes(self,
                         scenario_text: str,
                         probe_text: str,
                         choices: List[str],
                         log_file: Optional[TextIO] = None,
                         max_new_tokens: int = 512,
                         temperature: float = 0.6,
                         template: str = 'pred_outcome.txt') -> List[str]:
        """
        Predicts outcomes for given scenario, probe and choices.

        :param scenario: Scenario text.
        :param probe: Probe text.
        :param choices: Choices text.
        :param log_file: Optional log file.
        :param max_tokens: Maximum number of tokens to generate.
        :param temperature: Temperature for sampling.
        :param outcome_template_file: Template file for Outcomes.
        :return: List of generated predictions.
        """
        return self.generate_from_template(
            template,
            [
                {
                    'scenario': scenario_text,
                    'probe': probe_text, 
                    'choice': choice,
                }
                for choice in choices 
            ],
            log_file=log_file,
            max_tokens=max_new_tokens,
            temperature=temperature
        )

    
    def predict_kdma_values(self,
                            scenario_text: str,
                            probe_text: str,
                            choice_texts: List[str],
                            predicted_outcomes: Optional[List[str]] = None,
                            generate_reasoning: bool = True,
                            log_file: Optional[TextIO] = None,
                            max_new_tokens: int = 512,
                            temperature: float = 0.6,
                            template: str = 'pred_kdma_RO.txt',
                            kdma_descriptions_file: str = 'lib/templates/bbn_kdma_descriptions.yml') -> Union[List[Dict[str, float]], Tuple[List[Dict[str, float]], List[Dict[str, str]]]]:
        """
        Predicts KDMA scores each choice text under the given scenario and probe.

        :param scenario_text: Scenario text.
        :param probe_text: Probe text.
        :param choice_texts: Choices text.
        :param predicted_outcomes: Predicted outcomes.
        :param generate_reasoning: Flag to generate reasoning.
        :param log_file: Optional log file.
        :param max_new_tokens: Maximum number of new tokens to generate.
        :param temperature: Temperature for sampling.
        :param template: Template file for KDMA prediction.
        :param kdma_descriptions_file: Template file for KDMA descriptions.
        :return: KDMA predictions. If generate_reasoning is True, return predictions and reasonings.
        """
        choice_ids = [f'choice_{i}' for i in range(len(choice_texts))]
        substitutions = []
        info = []
        
        relative_dir = os.path.dirname(__file__)
        kdma_descriptions_file_path = os.path.join(relative_dir, kdma_descriptions_file)
        
        with open(kdma_descriptions_file_path, 'r') as f:
            kdma_descriptions = yaml.load(f, Loader=yaml.FullLoader)
        
        if predicted_outcomes is None:
            predicted_outcomes = [None] * len(choice_texts)
        
        for choice_id, choice, outcome in zip(choice_ids, choice_texts, predicted_outcomes):
            for kdma, kdma_info in kdma_descriptions.items():
                substitution = {
                    'kdma': kdma_info['name'],
                    'kdma_description': kdma_info['description'],
                    'scenario': scenario_text,
                    'probe': probe_text,
                    'choice': choice,
                }
                
                if outcome is not None:
                    substitution['outcome'] = outcome
                    
                substitutions.append(substitution)
                info.append((choice_id, kdma))
        
        def parse_kdma_score_response(response: str) -> Dict[str, Union[float, str]]:
            """
            Parses KDMA score response.

            :param response: Response to parse.
            :return: Dictionary with KDMA score and reasoning if generate_reasoning.
            """
            if generate_reasoning:
                start_idx = response.find('{')
                end_idx = response.rfind('}')
                response_json = json.loads(response[start_idx:end_idx+1])
                assert 'score' in response_json, 'score not found in response'
                assert 'reasoning' in response_json, 'reasoning not found in response'
            else:
                # find the first numeric character
                char = None
                for c in response:
                    if c.isnumeric():
                        char = c
                        break
                assert char is not None, 'Could not find numeric character in response'
                response_json = {
                    'score': float(response[response.find(char):])
                }                
            return response_json
            
        generations = self.generate_from_template(
            template,
            substitutions,
            parse_kdma_score_response,
            log_file=log_file,
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        
        predicted_kdma_values = {}
        reasonings = {}
        for (choice_id, kdma), generation in zip(info, generations):
            predicted_choice_kdmas = predicted_kdma_values.get(choice_id, {})
            predicted_kdma_values[choice_id] = predicted_choice_kdmas
            
            choice_reasonings = reasonings.get(choice_id, {})
            reasonings[choice_id] = choice_reasonings
            
            predicted_choice_kdmas[kdma] = generation['score']
            
            if generate_reasoning:
                choice_reasonings[kdma] = generation['reasoning']
        
        predicted_kdma_values = [
            predicted_kdma_values[choice_id]
            for choice_id in choice_ids
        ]
        if generate_reasoning:
            reasonings = [
                reasonings[choice_id]
                for choice_id in choice_ids
            ]
        
        if generate_reasoning:
            return predicted_kdma_values, reasonings
        else:
            return predicted_kdma_values
    
    
    def __call__(self, sample, target_kdma_values, **kwargs):
        scenario_text = sample['scenario']
        if sample['state'] is not None:
            scenario_text += f'\n{sample["state"]}'
        
        predicted_outcomes = None
        if 'predict_outcomes'in kwargs:
            predicted_outcomes = self.predict_outcomes(
                scenario_text,
                sample['probe'],
                sample['choices'],
                **kwargs['predict_outcomes']
            )
        
        predicted_kdma_values, generated_reasoning = self.predict_kdma_values(
            scenario_text,
            sample['probe'],
            sample['choices'],
            predicted_outcomes=predicted_outcomes,
            **kwargs['predict_kdma_values']
        )
        
        def mse(target_kdma_values, predicted_kdma_values):
            kdmas = set(target_kdma_values.keys()) & set(predicted_kdma_values.keys())
            
            if len(kdmas) == 0:
                return 0
        
            return sum([(target_kdma_values[kdma] - predicted_kdma_values[kdma])**2 for kdma in kdmas]) / len(kdmas)

        # find index of min mse
        choice_idx = 0
        min_mse = float('inf')
        for i, choice in enumerate(sample['choices']):
            mse_ = mse(target_kdma_values, predicted_kdma_values[i])
            if mse_ < min_mse:
                min_mse = mse_
                choice_idx = i
        
        return {
            'choice': choice_idx,
            'predicted_kdmas': predicted_kdma_values,
            'info': {
                'predicted_outcomes': predicted_outcomes,
                'generated_reasoning': generated_reasoning,
            }
        }