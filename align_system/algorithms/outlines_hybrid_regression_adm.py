from typing import Any, Dict, List

import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer

import outlines
from outlines.samplers import MultinomialSampler
from rich.highlighter import JSONHighlighter
from swagger_client.models import (
    kdma_value
)

from align_system.utils import logging
from align_system.algorithms.outlines_adm import OutlinesTransformersADM
from align_system.prompt_engineering.outlines_prompts import (

    action_selection_prompt,
    scenario_state_description_1,
    regression_alignment_system_prompt
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


def load_itm_sentences(data_dict: Dict[str, List], attr: str):
    df = pd.DataFrame.from_dict(data_dict)
    if attr not in df.columns:
        raise RuntimeError("KDMA does not exist in provided data")
    # Identify relevant rows in df based on chosen kdma
    idx = df[attr].dropna().index
    df_kdma = df.loc[idx]

    labels = [v for v in df_kdma[attr].values]
    scenarios = df_kdma['scenario'].values
    states = df_kdma['state'].replace(np.nan, '').values
    answers = df_kdma['answer'].values
    sentences = [sc + st + " [SEP] " + ans for (sc, st, ans)
                 in zip(scenarios, states, answers)]
    return sentences, labels


def load_model_data(model_name: str, data_dict: Dict[str, List], attr: str) -> TensorDataset:
    sentences, labels = load_itm_sentences(data_dict, attr=attr)
    sentences = ["[CLS] " + s for s in sentences]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ids, amasks = get_ids_mask(sentences, tokenizer, max_length=512)
    inputs, labels, masks = torch.tensor(ids), torch.tensor(labels), torch.tensor(amasks)
    data = TensorDataset(inputs, masks, labels)
    return data


def get_kdma_predictions(model_name: str, data_dict: Dict[str, List],
                         attr: str, model_path: str) -> np.ndarray:
    dataset = load_model_data(
        model_name=model_name,
        data_dict=data_dict,
        attr=attr
    )
    regression_model = BertRegressionModel(model_name=model_name, load_path=model_path)
    regression_model.load_model()
    predictions = regression_model.evaluate(dataset)

    return predictions


class BertRegressionModel:
    def __init__(self, model_name: str, load_path: str,
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model_name = model_name
        self.load_path = load_path
        self.device = device

    def load_model(self):
        config = AutoConfig(self.model_name, num_labels=1)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name, config=config)
        self.model = model.load_state_dict(torch.load(self.load_path))

    def evaluate(self, dataset):

        self.model.to(self.device)

        self.model.eval()
        for batch in dataset:
            # Copy data to GPU if needed
            batch = tuple(t.to(self.device) for t in batch)

            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch

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

        scenario = [scenario_description for i in range(len(choices))]

        data_dict = {
            'scenario': scenario,
            'answer': choices
        }

        inference_args = kwargs.get('inference_kwargs', {})

        target_models = inference_args["models"]
        target_model_checkpoints = inference_args["models"]["target_checkpoint"].keys()
        model_name = inference_args["models"]["model_name"]

        target_kdmas = alignment_target.kdma_values

        for i, target_kdma in enumerate(target_kdmas):
            target_kdma_name = target_kdma.kdma
            if target_kdma_name in target_model_checkpoints:
                print(f"Chosen KDMA: {target_kdma_name}")
                predictions = get_kdma_predictions(
                    model_name=model_name,
                    data_dict=data_dict,
                    attr=str(target_kdma_name),
                    model_path=target_models["target_checkpoint"][str(target_kdma_name)]
                )

            choice_scores = []
            for p in predictions:
                choice_scores.append((target_kdma - p)**2)

        choice_index = np.argmin(choice_scores)
        final_choice = choices[choice_index]

        action_to_take = available_actions[final_choice]
        # Set up simple diaolg to return for follow-ups
        alignment_system_prompt = regression_alignment_system_prompt(target_kdmas)
        prompt = action_selection_prompt(scenario_description, choices)
        dialog = [{'role': 'system', 'content': alignment_system_prompt},
                  {'role': 'user', 'content': prompt}]

        return action_to_take, dialog
