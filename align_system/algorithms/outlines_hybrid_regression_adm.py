from typing import Any, Dict, List

import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer

import outlines
from outlines.samplers import MultinomialSampler
from rich.highlighter import JSONHighlighter

from align_system.utils import logging
from align_system.algorithms.outlines_adm import OutlinesTransformersADM
from align_system.prompt_engineering.outlines_prompts import (
    action_selection_prompt,
    scenario_state_description_1,
    regression_error_alignment_system_prompt
)

log = logging.getLogger(__name__)
JSON_HIGHLIGHTER = JSONHighlighter()


def get_ids_mask(sentences, tokenizer, max_length):
    tokenized = [tokenizer.tokenize(s) for s in sentences]
    tokenized = [t[:(max_length - 1)] + ['SEP'] for t in tokenized]

    ids = [tokenizer.convert_tokens_to_ids(t) for t in tokenized]
    ids = np.array([np.pad(i, (0, max_length - len(i)),
                           mode='constant') for i in ids])

    amasks = []
    for seq in ids:
        seq_mask = [float(i > 0) for i in seq]
        amasks.append(seq_mask)
    return ids, amasks


def load_itm_sentences(data_dict: Dict[str, List]):
    df = pd.DataFrame.from_dict(data_dict)
    scenarios = df['scenario'].values
    # states = df['state'].replace(np.nan, '').values
    answers = df['answer'].values
    sentences = [sc + " [SEP] " + ans for (sc, ans)
                 in zip(scenarios, answers)]
    return sentences


def load_model_data(model_name: str, data_dict: Dict[str, List]) -> TensorDataset:
    sentences = load_itm_sentences(data_dict)
    sentences = ["[CLS] " + s for s in sentences]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ids, amasks = get_ids_mask(sentences, tokenizer, max_length=512)
    inputs, masks = torch.tensor(ids), torch.tensor(amasks)
    data = TensorDataset(inputs, masks)
    return data


def get_kdma_predictions(model_name: str, data_dict: Dict[str, List], model_path: str) -> np.ndarray:
    dataset = load_model_data(
        model_name=model_name,
        data_dict=data_dict
    )
    dataloader = DataLoader(dataset, batch_size=1)
    regression_model = BertRegressionModel(model_name=model_name, load_path=model_path)
    predictions = regression_model.evaluate(dataloader)

    return predictions


class BertRegressionModel:
    def __init__(self, model_name: str, load_path: str,
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model_name = model_name
        self.load_path = load_path
        self.device = device
        config = AutoConfig.from_pretrained(self.model_name, num_labels=1)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, config=config)

    def evaluate(self, dataloader):
        self.model.load_state_dict(torch.load(self.load_path))
        self.model.to(self.device)
        self.model.eval()
        for batch in dataloader:
            # Copy data to GPU if needed
            if self.device == "cuda":
                batch = tuple(t.cuda() for t in batch)
            else:
                batch = tuple(t.cpu() for t in batch)

            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask = batch

            # Forward pass
            with torch.no_grad():
                logits = self.model(b_input_ids, attention_mask=b_input_mask)[0]
            output = logits.squeeze().detach().cpu().numpy()
            predictions = np.clip(output, 0, 1)

        return predictions


class HybridRegressionADM(OutlinesTransformersADM):
    def __init__(self,
                 model_name,
                 device='auto',
                 baseline=False,
                 sampler=MultinomialSampler(),
                 **kwargs):
        self.baseline = baseline
        self.model = outlines.models.transformers(
            model_name,
            device=device,
            model_kwargs=kwargs.get('model_kwargs', {}),
            tokenizer_kwargs=kwargs.get('tokenizer_kwargs', {}))
        # NOTE: In cases where we want multiple samples, we're passing
        # in a list of prompts (this allows us to shuffle answers in
        # each prompt), rather than setting the number of samples in
        # the sampler itself (which defaults to 1); setting the number
        # of samples in the sampler may result in unexpected behavior
        self.sampler = sampler

    def top_level_choose_action(self,
                                scenario_state,
                                available_actions,
                                alignment_target,
                                num_samples=1,
                                predict_outcomes=False,
                                distribution_matching='average',
                                generator_batch_size=5,
                                kdma_descriptions_map='align_system/prompt_engineering/kdma_descriptions.yml',
                                kdma_score_examples=False,
                                **kwargs):

        scenario_description = scenario_state_description_1(scenario_state)

        # Important that the choices stay in the same order as the
        # available actions as we'll use the selected index later to
        # map to the corresponding action
        choices = self.format_choices(
            [a.unstructured for a in available_actions],
            available_actions,
            scenario_state
        )

        # Replicate scenarios for the number of available choices
        scenario = [scenario_description for i in range(len(choices))]

        data_dict = {
            'scenario': scenario,
            'answer': choices
        }

        target_models = kwargs.get('models', {})
        target_model_checkpoints = target_models["target_checkpoint"].keys()
        model_name = target_models["model_name"]

        target_kdmas = alignment_target.kdma_values

        target_kdma_configs = []
        for i, target_kdma in enumerate(target_kdmas):
            target_kdma_name = target_kdma.kdma
            if target_kdma_name in target_model_checkpoints:
                log.info("Chosen KDMA: %s, and target value: %f", target_kdma_name, target_kdma.value)
                predictions = get_kdma_predictions(
                    model_name=model_name,
                    data_dict=data_dict,
                    model_path=target_models["target_checkpoint"][str(target_kdma_name)]
                )
                predictions = np.array([predictions])
                dist_to_align_score = []
                for pred in predictions:
                    target_kdma_value = target_kdma.value
                    dist_to_align_score.append((target_kdma_value - pred)**2)

                choice_index = np.argmin(dist_to_align_score)
                final_choice = choices[choice_index]

                target_kdma_configs.append(
                    {
                        "name": target_kdma_name,
                        "score": predictions[choice_index] * 10,
                        "action_choice": final_choice
                    }
                )

        log.info('Predicted KDMA value from Regression model = %f', predictions[choice_index])
        action_to_take = available_actions[choice_index]
        action_to_take.justification = (
            f"The direct kdma predictions from a pre-trained bert-based-uncased regression model"
            f" for the chosen action - \"{final_choice}\" - is equal to {predictions[choice_index] * 10}."
        )

        # Set up simple diaolg to return for follow-ups
        alignment_system_prompt = regression_error_alignment_system_prompt(target_kdma_configs)
        prompt = action_selection_prompt(scenario_description, choices)
        dialog = [{'role': 'system', 'content': alignment_system_prompt},
                  {'role': 'user', 'content': prompt}]

        return action_to_take, dialog
