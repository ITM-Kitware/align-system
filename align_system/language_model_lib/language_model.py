import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from typing import List

class LanguageModel:
    """
    A class that handles transformers Language Models
    """

    @classmethod
    def load_model(cls, hf_model_name: str, precision: torch.dtype = torch.float32, device: str = 'cuda') -> 'LanguageModel':
        """
        Loads the specified transformer model and tokenizer.

        Args:
            hf_model_name (str): The huggingface model name.
            precision (torch.dtype, optional): The precision of the model weights. Defaults to torch.float32.
            device (str, optional): The device to move the model to. Defaults to 'cuda'.

        Returns:
            LanguageModel: An instance of this class with the loaded model and tokenizer.
        """
        model = AutoModelForCausalLM.from_pretrained(hf_model_name, torch_dtype=precision)
        tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
        model = model.to(device)
        return cls(model, tokenizer)
    
    
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer) -> None:
        """
        Initializes the LanguageModel instance with the given model and tokenizer.

        Args:
            model (AutoModelForCausalLM): The loaded transformer model.
            tokenizer (AutoTokenizer): The loaded tokenizer.
        """
        self.model = model
        self.tokenizer = tokenizer
    
    
    def generate_from_tokens(self, prompt_token_lists: List[List[int]], log_file=None, max_new_tokens: int=512, temperature: float=0.6, padding='left'):
        """
        Generates text from a list of tokenized prompts.

        Args:
            prompt_token_lists (List[List[int]]): A batch of lists where each list is a sequence of tokens.
            max_new_tokens (int, optional): The maximum number of tokens to generate. Defaults to 512.
            temperature (float, optional): The temperature for the generation algorithm. Defaults to 0.6.

        Returns:
            List[str]: The generated text for each prompt in the input list. Only contains text after the prompt.
        """
        prompt_token_lists = [
            torch.tensor(prompt_tokens).to(self.model.device).unsqueeze(0)
            for prompt_tokens in prompt_token_lists
        ]
        
        max_length = max([prompt_tokens.size(1) for prompt_tokens in prompt_token_lists])

        pad_token_id = self.tokenizer.pad_token_id
        # Pad each sequence to the max length
        assert padding == 'left' or padding == 'right', f"Padding must be either 'left' or 'right', got {padding}"
        pad_fn = lambda prompt_token_size: (max_length - prompt_token_size, 0) if padding == 'left' else (0, max_length - prompt_token_size)
        
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


    
    def generate(self, prompt_texts: List[str], log_file=None, max_new_tokens: int=512, temperature: float=0.6):
        """
        Generates text from a list of prompts.

        Args:
            prompt_texts (List[str]): A list of prompts.
            max_new_tokens (int, optional): The maximum number of tokens to generate. Defaults to 512.
            temperature (float, optional): The temperature for the generation algorithm. Defaults to 0.6.

        Returns:
            List[str]: The generated text for each prompt in the input list. Only contains text after the prompt.
        """
        # Convert text prompts to token prompts
        prompt_token_lists = [self.tokenizer.encode(prompt_text) for prompt_text in prompt_texts]
        return self.generate_from_tokens(prompt_token_lists, log_file, max_new_tokens, temperature)


    def generate_with_prefixes(self, prompt_texts: List[str], prefixes: List[str], log_file=None, max_new_tokens: int=512, temperature: float=0.6):
        """
        Generates text from a list of prompts with a list of prefixes.

        Args:
            prompt_texts (List[str]): A list of prompts.
            prefixes (List[str]): A list of prefixes.
            max_new_tokens (int, optional): The maximum number of tokens to generate. Defaults to 512.
            temperature (float, optional): The temperature for the generation algorithm. Defaults to 0.6.

        Returns:
            List[str]: The generated text for each prompt in the input list. Includes the prefix but not the prompt.
        """
        combined_texts = [f'{prompt}{prefix}' for prompt, prefix in zip(prompt_texts, prefixes)]
        generations = self.generate(combined_texts, log_file, max_new_tokens, temperature)
        return [f'{prefix}{generation}' for prefix, generation in zip(prefixes, generations)]