import openai

from align_system.algorithms.llm_chat_baseline import LLMChatBaseline, STANDARD_MULTIPLE_CHOICE_JSON_FORMAT
from align_system.utils.backoff import retry_with_exponential_backoff


class OpenAIChatBaseline(LLMChatBaseline):

    def __init__(self, openai_api_key, model='gpt-3.5-turbo'):
        self.model = model
        openai.api_key = openai_api_key
        


    def answer_multiple_choice(self, question, options, system_message=None, prefix=None, json_format=STANDARD_MULTIPLE_CHOICE_JSON_FORMAT):
        medical_triage_system_message = (
            "You are an assistant specialized in answering multiple-choice questions related to "
            "medical triage. When a question is posed, carefully analyze the symptoms or conditions "
            "described. Respond by providing a detailed reasoning using a step-by-step process or "
            "process of elimination. Conclude with the final answer, represented by the corresponding "
            "index number. Your response should be grounded in established medical knowledge and aim "
            "to be informative. Please format your response as a JSON object in the following structure:\\n\\n"
            f"{json_format}"
            "Ensure that you adhere to proper JSON syntax, and carefully note the index that corresponds to each answer."
        )
        if system_message is None:
            system_message = medical_triage_system_message

        formatted_options = [f'({i}) {option}' for i, option in enumerate(options)]

        content = f'{question} {formatted_options}'

        dialog = [
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user",
                "content": content
            }
        ]
        
        completion = self.completions_with_backoff(
            model=self.model,
            messages=dialog,
        )
            
        
        return completion.choices[0].message

    
    @retry_with_exponential_backoff(errors=(openai.error.RateLimitError,))
    def completions_with_backoff(**kwargs):
        return openai.Completion.create(**kwargs)
    
        
    
    # def aligned_decision_maker(self, question, choices, target_kdmas, system_message_provider, n_samples=5, inverse_misaligned=True, shuffle=True, baseline=False):
    #     assert len(target_kdmas) == 1, "Only one KDMA can be targeted at a time, but received: {}".format(target_kdmas)
        
    #     kdma = list(target_kdmas.keys())[0]
        
    #     assert kdma in kdmas, f"KDMA {kdma} not supported."
        
    #     responses = []
        
    #     for _ in range(n_samples):
    #         system_message_keys = [kdma, 'high' if target_kdmas[kdma] > 5 else 'low']
            
    #         indecies = list(range(len(choices)))
    #         if shuffle:
    #             random.shuffle(indecies)
    #         shuffled_choices = [choices[i] for i in indecies]
            
    #         system_message = system_message_provider(system_message_keys[0], system_message_keys[1])
            
    #         if baseline:
    #             system_message = system_message_provider('baseline', None)
    #             system_message_keys[1] = 'baseline'
            
    #         high_response = self.answer_multiple_choice(
    #             question,
    #             shuffled_choices,
    #             system_message=system_message,
    #         )
            
    #         reasoning, answer_idx = LLMChatBaseline.parse_generated_output(high_response)
    #         responses.append({
    #             'response': high_response,
    #             'reasoning': reasoning,
    #             'answer_idx': answer_idx,
    #             'shuffle_indecies': indecies,
    #             'kdma': kdma,
    #             'alignment': system_message_keys[1],
    #             'aligned': True,
    #         })
            
    #         if inverse_misaligned:
    #             system_message_keys = (kdma, 'high' if not target_kdmas[kdma] > 5 else 'low')
                
    #             indecies = list(range(len(choices)))
    #             if shuffle:
    #                 random.shuffle(indecies)
    #             shuffled_choices = [choices[i] for i in indecies]
                
    #             low_response = self.answer_multiple_choice(
    #                 question,
    #                 shuffled_choices,
    #                 system_message=system_message_provider(system_message_keys[0], system_message_keys[1]),
    #             )
                
    #             reasoning, answer_idx = LLMChatBaseline.parse_generated_output(low_response)
    #             responses.append({
    #                 'response': low_response,
    #                 'reasoning': reasoning,
    #                 'answer_idx': answer_idx,
    #                 'shuffle_indecies': indecies,
    #                 'kdma': kdma,
    #                 'alignment': system_message_keys[1],
    #                 'aligned': False,
    #             })
        
    #     return responses