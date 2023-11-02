from typing import List, Dict, Optional, Callable, Union, TextIO

from align_system.algorithms.lib.language_model import LanguageModel
# from align_system.algorithms.lib.chat.dialog_tokenizer import DialogTokenizer
from align_system.algorithms.lib.util import read_template, format_template, dialog_from_string, dialog_to_string
from jinja2.exceptions import TemplateError

class ChatLanguageModel(LanguageModel):

    # def __init__(self, model: LanguageModel, tokenizer: Callable[[str], List[str]]):
    #     """
    #     Initializes the chat language model.

    #     :param model: Pretrained language model.
    #     :param tokenizer: Tokenizer function.
    #     """
    #     super().__init__(model, tokenizer)
    #     # model_name = model.name_or_path
    #     # assert model_name in dialog_tokenizers, f'No dialog tokenizer found for model {model_name}'
    #     # self.dialog_tokenizer = dialog_tokenizers[model_name](tokenizer)

    def generate_responses(self, 
                           dialogs: List[Dict[str, str]], 
                           log_file: Optional[TextIO] = None, 
                           max_new_tokens: int = 512, 
                           temperature: float = 0.6) -> List[str]:
        """
        Generates responses for given dialogs.

        :param dialogs: List of dialogs.
        :param log_file: Optional file to log the process. 
        :param max_new_tokens: Maximum number of new tokens to generate.
        :param temperature: Temperature for sampling.
        :return: Generated responses.
        """
        # If logging is requested, write the dialogues into the log file
        if log_file is not None:
            log_file.write('**Dialogs:**\n')
            for i, dialog in enumerate(dialogs):
                log_file.write(f'*Dialog {i}:*\n{dialog_to_string(dialog)}\n')
            log_file.flush()

        # Prepare lists for the last user dialogues and prefixes.
        # Prefix refers to the assistant's response in the last turn of a dialogue.
        user_last_dialogs = []
        prefixes = []
        for dialog in dialogs:
            prefix = ''
            if dialog[-1]['role'] == 'assistant':
                prefix = dialog[-1]['content']
                dialog = dialog[:-1]
            user_last_dialogs.append(dialog)
            prefixes.append(prefix)

        # Tokenization step
        try:
            prompt_token_lists = [
                [self.tokenizer.apply_chat_template(dialog, tokenize=True)]
                for dialog in user_last_dialogs
            ]
        except TemplateError as e:
            systemless_dialogs = []
            for dialog in user_last_dialogs:
                if dialog[0]['role'] == 'system':
                    dialog[0]['role'] = 'user'
                if dialog[1]['role'] == 'user':
                    dialog[0]['content'] = f"{dialog[0]['content']}\n\n{dialog[1]['content']}"
                    del dialog[1]
                systemless_dialogs.append(dialog)
            
            prompt_token_lists = [
                [self.tokenizer.apply_chat_template(dialog, tokenize=True)]
                for dialog in systemless_dialogs
            ]
                
            

        # Add the prefix tokens to the prompt tokens
        for prompt_tokens, prefix in zip(prompt_token_lists, prefixes):
            if len(prefix) > 0:
                prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
                prompt_tokens[0] += prefix_tokens

        # Generate responses using tokens
        prompt_token_lists = [x[0] for x in prompt_token_lists]
        responses = self.generate_from_tokens(prompt_token_lists, max_new_tokens=max_new_tokens, temperature=temperature)
        prefixed_responses = [
            f'{prefix}{response}'
            for prefix, response in zip(prefixes, responses)
        ]
        
        # If logging is requested, write the generated responses into the log file
        if log_file is not None:
            log_file.write('**Generated Responses:**\n')
            for i, response in enumerate(prefixed_responses):
                log_file.write(f'*Response {i}:*\n{response}\n')
            log_file.flush()

        return prefixed_responses

    def generate_from_template(
        self,
        template_files: Union[List[str], str],
        substitution_dicts: Union[List[Dict[str, str]], Dict[str, str]],
        parse_generation_fn: Optional[Callable[[str], str]] = None,
        batch_size: int = 5,
        log_file: Optional[TextIO] = None,
        max_tokens: int = 512,
        temperature: float = 0.6,
        max_retry: int = 100,
        verbose: bool = False) -> List[str]:
        """
        Generates responses for given templates with substitutions.

        :param template_files: Template files to use for generation.
        :param substitution_dicts: Substitution dictionaries for the templates.
        :param parse_generation_fn: Function to parse the generated responses.
        :param batch_size: Batch size for generating responses.
        :param log_file: Optional file to log the process.
        :param max_tokens: Maximum number of tokens to generate.
        :param temperature: Temperature for sampling.
        :param max_retry: Maximum number of attempts to generate a valid output.
        :param verbose: If True, verbose logging is enabled.
        :return: Generated responses.
        """
        if isinstance(substitution_dicts, dict):
            substitution_dicts = [substitution_dicts]

        if isinstance(template_files, str):
            template_files = [template_files] * len(substitution_dicts)

        assert len(template_files) == len(substitution_dicts), 'Number of templates and substitutions do not match'

        # Create a dialogue for each template/substitution pair
        dialogs = {
            i: dialog_from_string(format_template(read_template(template_file), **substitutions))
            for i, (template_file, substitutions) in enumerate(zip(template_files, substitution_dicts))
        }

        outputs = {}
        input_counts = {}
        while len(dialogs) > 0:
            sample_ids = list(dialogs.keys())[:batch_size]
            batch = [dialogs[i] for i in sample_ids]
            generations = self.generate_responses(batch, log_file=log_file, max_new_tokens=max_tokens, temperature=temperature)

            # Process the generated responses
            for sample_id, generation in zip(sample_ids, generations):
                input_counts[sample_id] = input_counts.get(sample_id, 0) + 1

                # If the maximum number of try-outs is exceeded, throw an error
                if input_counts[sample_id] > max_retry:
                    raise Exception(f'Could not generate valid output for sample [{sample_id}]')

                # If there's a specific function to parse the generations, try to apply it
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