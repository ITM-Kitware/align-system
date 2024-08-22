from typing import Dict, List

import numpy as np
import pandas as pd
from swagger_client.models import KDMAValue
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer

from rich.highlighter import JSONHighlighter

from align_system.utils import logging, alignment_utils, adm_utils
from align_system.algorithms.outlines_adm import OutlinesTransformersADM
from align_system.prompt_engineering.outlines_prompts import (
    action_selection_prompt,
    scenario_description_hybrid_regression,
    baseline_system_prompt,
    scenario_state_description_dre
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
    scenarios = df['scenario'].replace('\n', '').values
    answers = df['answer'].replace('\n', '').values
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
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name,
                                                                        config=config)

    def evaluate(self, dataloader):
        self.model.load_state_dict(torch.load(self.load_path))
        self.model.to(self.device)
        self.model.eval()
        predictions = np.array([])
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
            predictions = np.append(predictions, output)

        return np.clip(predictions, 0, 1)


class HybridRegressionADM(OutlinesTransformersADM):

    def top_level_choose_action(self,
                                scenario_state,
                                available_actions,
                                alignment_target,
                                distribution_matching='sample',
                                kde_norm='globalnorm',
                                probabilistic=False,
                                **kwargs):

        scenario = []
        for action in available_actions:
            scenario_description = scenario_description_hybrid_regression(scenario_state)
            scenario.append(scenario_description)
        # Important that the choices stay in the same order as the
        # available actions as we'll use the selected index later to
        # map to the corresponding action
        choices = adm_utils.format_choices(
            [a.unstructured for a in available_actions],
            available_actions,
            scenario_state
        )

        data_dict = {
            'scenario': scenario,
            'answer': choices
        }

        target_models = kwargs.get('models', {})
        if not target_models:
            raise KeyError("'models' field is empty")
        for k in ["target_checkpoint", "model_name"]:
            if k not in target_models:
                raise KeyError(f"{k} does not exist")

        target_model_checkpoints = target_models["target_checkpoint"].keys()
        model_name = target_models["model_name"]

        target_kdmas = alignment_target.kdma_values

        for i, target_kdma in enumerate(target_kdmas):
            if isinstance(target_kdma, KDMAValue):
                target_kdma = target_kdma.to_dict()
            target_kdma_name = target_kdma["kdma"]

            if target_kdma_name not in target_model_checkpoints:
                raise ValueError(f"Model checkpoint does not exist for {target_kdma_name}")

            # Add probe at the end of the ADEPT scenarios text
            if target_kdma_name in ['Moral judgement', 'Ingroup Bias']:
                updated_scenario = []
                for d in data_dict['scenario']:
                    d += "Who would you treat?"
                    updated_scenario.append(d)
                data_dict.update({'scenario': updated_scenario})

            predictions_per_kdma = get_kdma_predictions(
                model_name=model_name,
                data_dict=data_dict,
                model_path=target_models["target_checkpoint"][str(target_kdma_name)]
            )
            predicted_kdma_values = {}
            log.info("ACTION CHOICES AND REGRESSION MODEL PREDICTIONS:")
            for pred, choice in zip(predictions_per_kdma, choices):
                if choice not in predicted_kdma_values:
                    predicted_kdma_values[choice] = {}
                predicted_kdma_values[choice][target_kdma_name] = pred

                log.info(
                    f"Action: \"{choice}\", Model predicted KDMA_value: {pred}"
                )

        # Get type of targets
        all_scalar_targets = True
        all_kde_targets = True
        for target_kdma in target_kdmas:
            if not hasattr(target_kdma, 'value') or target_kdma.value is None:
                all_scalar_targets = False
            if not hasattr(target_kdma, 'kdes') or target_kdma.kdes is None:
                all_kde_targets = False

        # Select aligned choice
        if all_scalar_targets:
            alignment_function = alignment_utils.AvgDistScalarAlignment()
            selected_choice, probs = alignment_function(
                predicted_kdma_values, target_kdmas, probabilistic=probabilistic
            )
        elif all_kde_targets:
            if distribution_matching == 'sample':
                alignment_function = alignment_utils.MinDistToRandomSampleKdeAlignment()
            elif distribution_matching == 'max_likelihood':
                alignment_function = alignment_utils.MaxLikelihoodKdeAlignment()
            elif distribution_matching == 'js_divergence':
                alignment_function = alignment_utils.JsDivergenceKdeAlignment()
            else:
                raise RuntimeError(distribution_matching, "distribution matching function unrecognized.")
            selected_choice, probs = alignment_function(
                predicted_kdma_values, target_kdmas, kde_norm=kde_norm, probabilistic=probabilistic
            )
        else:
            # TODO: Currently we assume all targets either have scalar values or KDES,
            #       Down the line, we should extend to handling multiple targets of mixed types
            raise ValueError("ADM does not currently support a mix of scalar and KDE targets.")

        log.info("\nFINAL CHOICE after distribution matching:")
        log.info(
            f"Action: \"{selected_choice}\", "
        )
        selected_choice_idx = choices.index(selected_choice)
        action_to_take = available_actions[selected_choice_idx]
        action_to_take.justification = (
            f"Based on doing a distribution matching on the predicted kdma values from a "
            f"pre-trained bert-based-uncased regression model, the chosen action is - "
            f"\"{selected_choice}\""
        )

        # Set up simple dialog to return for follow-ups
        alignment_system_prompt = baseline_system_prompt()
        prompt_scenario_description = scenario_state_description_dre(scenario_state)
        prompt = action_selection_prompt(prompt_scenario_description, choices)
        dialog = [{'role': 'system', 'content': alignment_system_prompt},
                  {'role': 'user', 'content': prompt}]

        return action_to_take, dialog
