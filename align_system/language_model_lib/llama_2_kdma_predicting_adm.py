import json

from align_system.language_model_lib.chat_langauge_model import ChatLanguageModel
from align_system.language_model_lib.util import extract_kdma_description

class Llama2KDMAPredictingADM(ChatLanguageModel):
            
    def predict_outcomes(
        self,
        scenario,
        probe,
        choices,
        log_file=None,
        max_tokens=512,
        temperature=0.6,
        outcome_template_file='templates/predict_outcomes.md'
    ):
        return self.generate_from_template(
            outcome_template_file,
            [
                {
                    'scenario': scenario,
                    'probe': probe, 
                    'choice': choice,
                }
                for choice in choices 
            ],
            log_file=log_file,
            max_tokens=max_tokens,
            temperature=temperature
        )
    
    
    def predict_kdma_scores(
        self,
        scenario_text,
        probe_text,
        choice_texts,
        predicted_outcomes=None,
        generate_reasoning=True,
        log_file=None,
        max_new_tokens=512,
        temperature=0.6,
        kdma_template_file='templates/kdma.md',
        kdma_descriptions_file='templates/bbn_kdma_descriptions.md',
    ):
        choice_ids = [f'choice_{i}' for i in range(len(choice_texts))]
        substitutions = []
        info = []
        kdma_descriptions = extract_kdma_description(kdma_descriptions_file)
        if predicted_outcomes is None:
            predicted_outcomes = [None] * len(choice_texts)
        for choice_id, choice, outcome in zip(choice_ids, choice_texts, predicted_outcomes):
            for kdma, kdma_description in kdma_descriptions.items():
                substitution = {
                    'kdma': kdma,
                    'kdma_description': kdma_description,
                    'scenario': scenario_text,
                    'probe': probe_text,
                    'choice': choice,
                }
                
                if outcome is not None:
                    substitution['outcome'] = outcome
                    
                substitutions.append(substitution)
                info.append((choice_id, kdma))
        
        def parse_kdma_score_response(response):
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
            kdma_template_file,
            substitutions,
            parse_kdma_score_response,
            log_file=log_file,
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        
        predicted_kdmas = {}
        reasonings = {}
        for (choice_id, kdma), generation in zip(info, generations):
            predicted_choice_kdmas = predicted_kdmas.get(choice_id, {})
            predicted_kdmas[choice_id] = predicted_choice_kdmas
            
            choice_reasonings = reasonings.get(choice_id, {})
            reasonings[choice_id] = choice_reasonings
            
            predicted_choice_kdmas[kdma] = generation['score']
            
            if generate_reasoning:
                choice_reasonings[kdma] = generation['reasoning']
            
        
        predicted_kdmas = [
            predicted_kdmas[choice_id]
            for choice_id in choice_ids
        ]
        if generate_reasoning:
            reasonings = [
                reasonings[choice_id]
                for choice_id in choice_ids
            ]
        
        if generate_reasoning:
            return predicted_kdmas, reasonings
        else:
            return predicted_kdmas

    def make_aligned_descision(
        self,
        scenario,
        probe,
        choices,
        target_kdmas,
        alignment_fn,
        predict_outcomes=True,
        generate_reasoning=True,
        kdma_descriptions_file='templates/bbn_kdma_descriptions.md',
        outcome_template_file='templates/predict_outcomes.md',
        kdma_template_file='templates/predict_kdma_scores_reasoning.md',
    ):
        # Generate the outcomes for each choice
        outcomes = None
        if predict_outcomes:
            outcomes = self.predict_outcomes(
                scenario,
                probe,
                choices,
                outcome_template_file=outcome_template_file
            )
            
            assert len(choices) == len(outcomes), 'Unexpected state: number of choices and outcomes do not match'
        
        # Get the scores and reasonings for each choice
        predicted_kdma_scores = self.predict_kdma_scores(
            scenario,
            probe,
            choices,
            outcomes=outcomes,
            kdma_template_file=kdma_template_file,
            kdma_descriptions_file=kdma_descriptions_file
        )
        
        if generate_reasoning:
            scores, reasonings = predicted_kdma_scores
        else:
            scores = predicted_kdma_scores
        
        assert len(choices) == len(scores), 'Unexpected state: number of choices and scores do not match'
        
        # Compute the similarity score for each choice
        alignment_scores = []
        for score in scores:
            alignment_scores.append(alignment_fn(target_kdmas, score))
        
        max_idx = alignment_scores.index(max(alignment_scores))
        
        justification = {
            'choice': choices[max_idx],
            'outcome': outcomes[max_idx],
            'kdma_scores': scores[max_idx],
        }
        
        if generate_reasoning:
            justification['kdma_reasonings'] = reasonings[max_idx]
        
        return max_idx, justification