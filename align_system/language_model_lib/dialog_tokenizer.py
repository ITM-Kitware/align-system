from abc import abstractmethod

class DialogTokenizer:
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    @abstractmethod
    def dialog_to_tokens(self, dialog_messages):
        pass
    

class Llama2DialogTokenizer(DialogTokenizer):
    
    
    def dialog_to_tokens(self, dialog_messages):
        # Define instance and system borders
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

        # If the role of the first message is system
        if dialog_messages[0]["role"] == "system":
            # Create an initial dialog entry combining system and user messages
            system_dialog = {"role": dialog_messages[1]["role"],
                            "content": B_SYS + dialog_messages[0]["content"] + E_SYS + dialog_messages[1]["content"]}
            # Update dialog to start with system_dialog and followed by the rest of the dialog
            dialog_messages = [system_dialog] + dialog_messages[2:]

        # Ensure the correct dialog order (system, user, assistant, user, assistant... )
        assert all([msg["role"] == "user" for msg in dialog_messages[::2]]) and all(
            [msg["role"] == "assistant" for msg in dialog_messages[1::2]]), \
            "Model only supports 'system', 'user' and 'assistant' roles, in the sequence (s/u/a/u/a...)"

        # Encode each user message and its following assistant message into tokens
        dialog_tokens = []
        for prompt, answer in zip(dialog_messages[::2], dialog_messages[1::2]):
            tokenized_message = ([self.tokenizer.bos_token_id] +
                                self.tokenizer.encode(f"{B_INST} {prompt['content'].strip()} {E_INST} {answer['content'].strip()} ",
                                                add_special_tokens=False) +
                                [self.tokenizer.eos_token_id])
            dialog_tokens.extend(tokenized_message)

        # Ensure the final message is from the user
        assert dialog_messages[-1]["role"] == "user", "Last message must be from the user."

        # Encode the user's final message into tokens and add to dialog_tokens
        user_final_message_tokens = ([self.tokenizer.bos_token_id] + self.tokenizer.encode(
            f"{B_INST} {dialog_messages[-1]['content'].strip()} {E_INST}",
            add_special_tokens=False))
        dialog_tokens.extend(user_final_message_tokens)
        
        return dialog_tokens
    
    
dialog_tokenizers = {
    'meta-llama/Llama-2-7b-chat-hf': Llama2DialogTokenizer,
    'meta-llama/Llama-2-13b-chat-hf': Llama2DialogTokenizer,
}