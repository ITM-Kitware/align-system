import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Union, Optional, TextIO, TypeVar

# Using generic type to improve IDE linting of LanguageModel subclasses
T = TypeVar('T', bound='LanguageModel')

class LanguageModel:
    """
    Class to define the Language Model.
    """

    @classmethod
    def load_model(cls, 
                   hf_model_name: str, 
                   precision: torch.dtype = torch.float32, 
                   device: str = 'cuda') -> T:
        """
        Load the language model.

        :param hf_model_name: Name of the model in Huggingface.
        :param precision: Precision of the model's weights.
        :param device: Device to run the model on.
        :return: Initialized LanguageModel object.
        """
        # Load the model from Huggingface
        if type(precision) == str and precision != 'auto':
            precisions = {
                'fp16': torch.float16,
                'float16': torch.float16,
                'half': torch.float16,
                'fp32': torch.float32,
                'float32': torch.float32,
                'full': torch.float32,
            }
            
            if precision not in precisions:
                raise ValueError(f'Precision must be one of {list(precisions.keys())}, got {precision}')
            
            precision = precisions[precision]
        if device == 'auto':
            model = AutoModelForCausalLM.from_pretrained(hf_model_name, torch_dtype=precision, device_map='auto')
        else:
            model = AutoModelForCausalLM.from_pretrained(hf_model_name, torch_dtype=precision)
            model = model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
        return cls(model, tokenizer)

    def __init__(self, 
                 model: AutoModelForCausalLM, 
                 tokenizer: AutoTokenizer) -> None:
        """
        Initializes the language model.

        :param model: Pretrained Huggingface model.
        :param tokenizer: Tokenizer from Huggingface.
        """
        self.model = model
        self.tokenizer = tokenizer

    def generate_from_tokens(self,
                             prompt_token_lists: List[List[int]], 
                             log_file: Union[None, str, object] = None, 
                             max_new_tokens: int = 512, 
                             temperature: float = 0.6, 
                             padding: str='left') -> List[str]:
        """
        Generates text from the given list of tokens.

        :param prompt_token_lists: List of lists of tokens to generate the text.
        :param log_file: Path to the log file.
        :param max_new_tokens: Maximum number of new tokens to be generated.
        :param temperature: Temperature for probability adjustment.
        :param padding: Padding direction, either 'left' or 'right'.
        :return: List of generated texts (not including the prompt).
        """
        # Move to the model's device and unpack
        prompt_token_lists = [
            torch.tensor(prompt_tokens).to(self.model.device).unsqueeze(0)
            for prompt_tokens in prompt_token_lists
        ]
        
        max_length = max([prompt_tokens.size(1) for prompt_tokens in prompt_token_lists])

        pad_token_id = self.tokenizer.pad_token_id

        # Padding function for the desired direction
        assert padding == 'left' or padding == 'right', f"Padding must be either 'left' or 'right', got {padding}"
        pad_fn = lambda prompt_token_size: (max_length - prompt_token_size, 0) if padding == 'left' else (0, max_length - prompt_token_size)

        # Pad each sequence to the max length
        padded_prompt_token_lists = [
            torch.nn.functional.pad(prompt_tokens, pad_fn(prompt_tokens.size(1)), value=pad_token_id)
            for prompt_tokens in prompt_token_lists
        ]
        
        attention_masks = [
            torch.nn.functional.pad(torch.ones_like(prompt_tokens), pad_fn(prompt_tokens.size(1)), value=0)
            for prompt_tokens in prompt_token_lists
        ]
        
        position_ids = [
            torch.nn.functional.pad(torch.arange(prompt_tokens.size(1)).unsqueeze(0), pad_fn(prompt_tokens.size(1)), value=0)
            for prompt_tokens in prompt_token_lists
        ]
        

        # Stack the padded sequences
        stacked_prompt_tokens = torch.cat(padded_prompt_token_lists, dim=0)
        stacked_attention_masks = torch.cat(attention_masks, dim=0)
        stacked_position_ids = torch.cat(position_ids, dim=0)
        
        if log_file is not None:
            prompt_texts = [
                self.tokenizer.decode(prompt_tokens.squeeze(0), skip_special_tokens=True)
                for prompt_tokens in padded_prompt_token_lists
            ]
            log_file.write('**Prompt texts:**\n')
            for i, prompt_text in enumerate(prompt_texts):
                log_file.write(f'Prompt {i}:\n{prompt_text}\n')
            
            log_file.flush()
                
        

        # Generate outputs for all dialogs in a batch
        # TODO ensure the batch size is not too large for the GPU
        outputs = self.model.generate(
            stacked_prompt_tokens, 
            attention_mask=stacked_attention_masks,
            # position_ids=stacked_position_ids, # TODO figure out why including the position ids breaks the model
            return_dict_in_generate=True, 
            output_scores=True, 
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature
        )
        
        # Decode the generated outputs
        decoded_outputs = [
            self.tokenizer.decode(output_tokens[len(prompt_tokens.squeeze(0)):], skip_special_tokens=True)
            for output_tokens, prompt_tokens in zip(outputs.sequences, padded_prompt_token_lists)
        ]
        
        if log_file is not None:
            log_file.write('**Generated texts:**\n')
            for i, decoded_output in enumerate(decoded_outputs):
                log_file.write(f'*Generation {i}:*\n{decoded_output}\n')
            log_file.flush()
        
        return decoded_outputs

    def generate(self, 
                 prompt_texts: List[str], 
                 log_file: Optional[TextIO] = None, 
                 max_new_tokens: int = 512, 
                 temperature: float = 0.6) -> List[str]:
        """
        Generates text from the given list of inputs.

        :param prompt_texts: List of prompts to generate from.
        :param log_file: Optional file object to write to
        :param max_new_tokens: Maximum number of new tokens to be generated.
        :param temperature: Temperature for probability adjustment.
        """
        # Convert the text to tokens and generate the text
        prompt_token_lists = [self.tokenizer.encode(prompt_text) for prompt_text in prompt_texts]
        return self.generate_from_tokens(prompt_token_lists, log_file, max_new_tokens, temperature)

    def generate_with_prefixes(self, 
                               prompt_texts: List[str], 
                               prefixes: List[str], 
                               log_file: Optional[TextIO] = None,
                               max_new_tokens: int = 512, 
                               temperature: float = 0.6) -> List[str]:
        """
        Generates text from the given list of inputs with prefixes.

        :param prompt_texts: List of prompts to generate from.
        :param prefixes: List of prefixes to prepend to the generated text.
        :param log_file: Optional file object to write to
        :param max_new_tokens: Maximum number of new tokens to be generated.
        :param temperature: Temperature for probability adjustment.
        """
        # Combine the inputs with prefixes and generate the text
        combined_texts = [f'{prompt}{prefix}' for prompt, prefix in zip(prompt_texts, prefixes)]
        generations = self.generate(combined_texts, log_file, max_new_tokens, temperature)
        return [f'{prefix}{generation}' for prefix, generation in zip(prefixes, generations)]
