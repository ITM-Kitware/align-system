from align_system.language_model_lib.language_model import LanguageModel
from align_system.language_model_lib.dialog_tokenizer import dialog_tokenizers
from align_system.language_model_lib.util import read_file, format_template, dialog_from_string, dialog_to_string


class ChatLanguageModel(LanguageModel):
    
    def __init__(self, model, tokenizer):
        super().__init__(model, tokenizer)
        model_name = model.name_or_path
        assert model_name in dialog_tokenizers, f'No dialog tokenizer found for model {model_name}'
        self.dialog_tokenizer = dialog_tokenizers[model_name](tokenizer)
    
    def generate_responses(self, dialogs, log_file=None, max_new_tokens=512, temperature=0.6):
        if log_file is not None:
            log_file.write('**Dialogs:**\n')
            for i, dialog in enumerate(dialogs):
                log_file.write(f'*Dialog {i}:*\n{dialog_to_string(dialog)}\n')
            log_file.flush()
        # Remove the last dialog piece if it is an assistant response
        # Use the assistant response as a prefix
        user_last_dialogs = []
        prefixes = []
        for dialog in dialogs:
            prefix = ''
            if dialog[-1]['role'] == 'assistant':
                prefix = dialog[-1]['content']
                dialog = dialog[:-1]
            user_last_dialogs.append(dialog)
            prefixes.append(prefix)
        dialogs = user_last_dialogs
        
        prompt_token_lists = [
            [self.dialog_tokenizer.dialog_to_tokens(dialog)]
            for dialog in dialogs
        ]

        for prompt_tokens, prefix in zip(prompt_token_lists, prefixes):
            if len(prefix) > 0:
                prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
                prompt_tokens[0] += prefix_tokens
            
        prompt_token_lists = [x[0] for x in prompt_token_lists]
        responses = self.generate_from_tokens(prompt_token_lists, max_new_tokens=max_new_tokens, temperature=temperature)
        
        prefixed_responses = [
            f'{prefix}{response}'
            for prefix, response in zip(prefixes, responses)
        ]
        
        if log_file is not None:
            log_file.write('**Generated Responses:**\n')
            for i, response in enumerate(prefixed_responses):
                log_file.write(f'*Response {i}:*\n{response}\n')
            log_file.flush()
            
        return prefixed_responses
    
    
    def generate_from_template(
        self,
        template_files,
        substitution_dicts,
        parse_generation_fn=None,
        batch_size=5,
        log_file=None,
        max_tokens=512,
        temperature=0.6,
        max_retry=10,
        verbose=False
    ):
        if type(substitution_dicts) is dict:
            substitution_dicts = [substitution_dicts]
            
        if type(template_files) is str:
            template_files = [template_files] * len(substitution_dicts)
        
        assert len(template_files) == len(substitution_dicts), 'Number of templates and substitutions do not match'
        
        dialogs = {
            i: dialog_from_string(format_template(read_file(template_file), **substitutions))
            for i, (template_file, substitutions) in enumerate(zip(template_files, substitution_dicts))
        }
        
        outputs = {}
        input_counts = {}
        while len(dialogs) > 0:
            sample_ids = list(dialogs.keys())[:batch_size]
            batch = [dialogs[i] for i in sample_ids]
            generations = self.generate_responses(batch, log_file=log_file, max_new_tokens=max_tokens, temperature=temperature)
            
            for sample_id, generation  in zip(sample_ids, generations):
                input_counts[sample_id] = input_counts.get(sample_id, 0) + 1
                if input_counts[sample_id] > max_retry:
                    raise Exception(f'Could not generate valid output for sample [{sample_id}]')
                
                if parse_generation_fn is not None:
                    try:
                        outputs[sample_id] = parse_generation_fn(generation)
                        del dialogs[sample_id]
                    except Exception as e:
                        if verbose:
                            print(f'Error: could not parse output for sample [{sample_id}]')
                            print(e)
                        pass
                else:
                    outputs[sample_id] = generation
                    del dialogs[sample_id]

        assert len(outputs) == len(substitution_dicts), 'Unexpected state: number of outputs and substitutions do not match'
        
        return [
            outputs[i]
            for i in range(len(outputs))
        ]