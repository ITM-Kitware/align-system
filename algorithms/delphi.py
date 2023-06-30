from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer
import torch
import os

class Delphi:
    def __init__(self, device="cuda", model="large"):  # device parameter instead of device_id
        self.device = torch.device(device)  # use the provided device
        print(f"Delphi device: {self.device}", model)

        if model == "large":
            self.MODEL_LOCATION = "../models/delphi-large"
            self.MODEL_BASE = "t5-large"

        elif model == "11b":
            self.MODEL_LOCATION = "../models/delphi-11b"
            self.MODEL_BASE = "t5-11b"

        else:
            raise ValueError("Model should be either 'large' or '11b'")

        # verify that the model exists
        if not os.path.exists(self.MODEL_LOCATION):
            raise ValueError("Model file does not exist: {}".format(self.MODEL_LOCATION))

        self.load_model()

    def load_model(self):
        self.model = T5ForConditionalGeneration.from_pretrained(self.MODEL_LOCATION)
        self.model.to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained(self.MODEL_BASE, model_max_length=512)

    def run_inference(self, input_string):
        input_ids = self.tokenizer(input_string, return_tensors='pt').to(self.device).input_ids
        outputs = self.model.generate(input_ids, max_length=200)

        decoded_outputs = self.tokenizer.decode(outputs[0])

        return decoded_outputs