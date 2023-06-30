import argparse
from dataclasses import dataclass
from typing import List, Set
import sys
from enum import Enum
import re
import random
import json

from swagger_client import ItmTa2EvalApi
from swagger_client.configuration import Configuration
from swagger_client.api_client import ApiClient
from swagger_client import (
    Scenario,
    State,
    Casualty,
    Supplies,
    Environment,
    Probe,
    ProbeOption,
    ProbeResponse,
    AlignmentTarget
)
import BERTSimilarity.BERTSimilarity as bertsimilarity

from algorithms.llm_baseline import (
    LLMBaseline,
    prepare_prompt,
    prepare_prompt_instruct_gpt_j,
)
from algorithms.llama_index import LlamaIndex

# Copy-paste from CACI's `itm_adm_scenario_runner.py` script; ideally
# we could just import this from their client module
@dataclass
class ADMKnowledge:
    """
    What the ADM keeps track of throughout the scenario.
    """
    # Scenario
    scenario_id: str = None
    scenario: Scenario = None
    scenario_complete: bool = False

    # Info
    description: str = None
    environment: Environment = None

    # casualties
    casualties: List[Casualty] = None
    all_casualty_ids: Set[str] = None
    treated_casualty_ids: Set[str] = None

    # Probes
    current_probe: Probe = None
    explanation: str = None
    probes_received: List[Probe] = None
    probes_answered: int = 0

    # Supplies
    supplies: List[Supplies] = None

    alignment_target: AlignmentTarget = None

    probe_options: List[ProbeOption] = None
    probe_choices: List[str] = None


# Copy-paste from CACI's `itm_scenario_runner.py` script; ideally
# we could just import this from their client module
class CommandOption(Enum):
    START = "start"
    PROBE = "probe"
    STATUS = "status"
    VITALS = "vitals"
    RESPOND = "respond"
    HEART_RATE = "heart rate"
    END = "end"


def main():
    parser = argparse.ArgumentParser(
        description="Simple LLM baseline system")

    parser.add_argument('-a', '--api_endpoint',
                        default="http://127.0.0.1:8080",
                        type=str,
                        help='Restful API endpoint for scenarios / probes '
                             '(default: "http://127.0.0.1:8080")')
    parser.add_argument('-u', '--username',
                        type=str,
                        default='ALIGN-ADM',
                        help='ADM Username (provided to TA3 API server, '
                             'default: "ALIGN-ADM")')
    parser.add_argument('-m', '--model',
                        type=str,
                        default="falcon",
                        help="LLM Baseline model to use")
    parser.add_argument('-t', '--align-to-target',
                        action='store_true',
                        default=False,
                        help="Align algorithm to target KDMAs")
    parser.add_argument('-a', '--algorithm',
                        type=str,
                        default="llama_index",
                        help="Algorithm to use")
    parser.add_argument('-A', '--algorithm-kwargs',
                        type=str,
                        required=False,
                        help="JSON encoded dictionary of kwargs for algorithm "
                             "initialization")

    run_baseline_system(**vars(parser.parse_args()))

    return 0


def retrieve_scenario(client, username):
    return client.start_scenario(username)


def retrieve_probe(client, scenario_id):
    return client.get_probe(scenario_id)


def retrieve_alignment_target(client, scenario_id):
    return client.get_alignment_target(scenario_id)


def answer_probe(client,
                 scenario_id,
                 probe_id,
                 explanation,
                 selected_choice_id=None):
    response_data = {
        'scenario_id': scenario_id,
        'probe_id': probe_id,
        'justification': explanation}

    if selected_choice_id is not None:
        response_data['choice'] = selected_choice_id

    return client.respond_to_probe(body=ProbeResponse(**response_data))


def adm_knowledge_from_scenario(scenario):
    adm_knowledge: ADMKnowledge = ADMKnowledge()

    state: State = scenario.state
    adm_knowledge.scenario_id = scenario.id
    adm_knowledge.casualties = state.casualties
    adm_knowledge.all_casualty_ids =\
        {casualty.id for casualty in state.casualties}
    adm_knowledge.treated_casualty_ids = set()
    adm_knowledge.probes_received = []
    adm_knowledge.supplies = state.supplies
    adm_knowledge.environment = state.environment
    adm_knowledge.description = state.mission.unstructured

    return adm_knowledge


# Current version of TA-3 API expects the provided explanation to be
# one of the medical supplies available (rather than a freeform
# response)
def _map_explanation_to_available_supply(
        text_explanation, supplies, fallback_to_random=False):
    supply_names_re = re.compile(
        '({})'.format('|'.join([s.name for s in supplies])), re.I)

    mentioned_supplies = re.findall(supply_names_re, text_explanation)

    if len(mentioned_supplies) == 0:
        selection = None
    else:
        selection = mentioned_supplies[0].lower()

    if selection is None and fallback_to_random:
        selection = random.choice([s.name for s in supplies])

    return selection


def force_choice_with_bert(text: str, choices: List[str]):
    bertsim = bertsimilarity.BERTSimilarity()

    top_score = -float('inf')
    top_choice = None
    top_choice_idx = None
    for i, choice in enumerate(choices):
        score = bertsim.calculate_distance(text, choice)

        if score > top_score:
            top_score = score
            top_choice = choice
            top_choice_idx = i

    return top_choice_idx, top_choice


def run_baseline_system(
        api_endpoint,
        username,
        model,
        align_to_target=False,
        algorithm="llm_baseline",
        algorithm_kwargs=None):
    # Needed to silence BERT warning messages, see: https://stackoverflow.com/questions/67546911/python-bert-error-some-weights-of-the-model-checkpoint-at-were-not-used-when # noqa
    from transformers import logging
    logging.set_verbosity_error()

    _config = Configuration()
    _config.host = api_endpoint
    _api_client = ApiClient(configuration=_config)
    client = ItmTa2EvalApi(api_client=_api_client)

    scenario = retrieve_scenario(client, username)
    adm_knowledge = adm_knowledge_from_scenario(scenario)

    if align_to_target:
        alignment_target = retrieve_alignment_target(client, scenario.id)
        adm_knowledge.alignment_target = alignment_target

    # Load the system / model
    algorithm_kwargs_parsed = {}
    if algorithm_kwargs is not None:
        algorithm_kwargs_parsed = json.loads(algorithm_kwargs)

    if algorithm == "llm_baseline":
        algorithm = LLMBaseline(
            device="cuda", model_use=model, distributed=False,
            **algorithm_kwargs_parsed)
    elif algorithm == "llama_index":
        algorithm = LlamaIndex(
            device="cuda", model_name=model,
            **algorithm_kwargs_parsed)

    algorithm.load_model()

    while not adm_knowledge.scenario_complete:
        current_probe = retrieve_probe(client, scenario.id)
        adm_knowledge.probes_received.append(current_probe)

        if model == "instruct-gpt-j":
            prompt = prepare_prompt_instruct_gpt_j(
                scenario, current_probe,
                alignment_target=adm_knowledge.alignment_target)
        else:
            prompt = prepare_prompt(
                scenario, current_probe,
                alignment_target=adm_knowledge.alignment_target)

        print("* Prompt for ADM: {}".format(prompt))

        raw_response = str(algorithm.run_inference(prompt))

        print("* ADM Raw response: {}".format(raw_response))

        if current_probe.type == 'MultipleChoice':
            selected_choice_idx, selected_choice = force_choice_with_bert(
                raw_response, [o.value for o in current_probe.options])
            selected_choice_id = current_probe.options[selected_choice_idx].id

            print("* ADM Selected: '{}'".format(selected_choice))

        print()
        response = answer_probe(client,
                                scenario.id,
                                current_probe.id,
                                selected_choice_id=selected_choice_id,
                                explanation=raw_response)

        adm_knowledge.scenario_complete = response.scenario_complete


if __name__ == "__main__":
    sys.exit(main())
