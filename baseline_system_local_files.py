import argparse
import json
import sys
from typing import List
from enum import Enum

import BERTSimilarity.BERTSimilarity as bertsimilarity

from algorithms.llm_baseline import LLMBaseline
from algorithms.llama_index import LlamaIndex


class ProbeType(Enum):
    MultipleChoice = "MultipleChoice"


def main():
    parser = argparse.ArgumentParser(
        description="Simple LLM baseline system running against local files")

    parser.add_argument('-s', '--scenario-filepath',
                        type=str,
                        required=True,
                        help="File path to input scenario JSON")
    parser.add_argument('-t', '--alignment-target-filepath',
                        type=str,
                        help="File path to input alignment target JSON")
    parser.add_argument('probes_filepaths',
                        type=str,
                        nargs='+',
                        help="File path to input probe JSON")
    parser.add_argument("--print-details",
                        action='store_true',
                        default=False,
                        help="Print out background / patient / probe "
                             "information in human readable form")
    parser.add_argument('-m', '--model',
                        type=str,
                        default="gpt-j",
                        help="LLM Baseline model to use")
    parser.add_argument('-a', '--algorithm',
                        type=str,
                        default="llm_baseline",
                        help="Algorithm to use")
    parser.add_argument('-A', '--algorithm-kwargs',
                        type=str,
                        required=False,
                        help="JSON encoded dictionary of kwargs for algorithm "
                             "initialization")

    run_baseline_system_local_filepath(**vars(parser.parse_args()))

    return 0


def run_baseline_system_local_filepath(
        probes_filepaths,
        scenario_filepath,
        model,
        alignment_target_filepath=None,
        print_details=False,
        algorithm="llm_baseline",
        algorithm_kwargs=None):
    with open(scenario_filepath) as f:
        scenario_data = json.load(f)

    if print_details:
        print("::SCENARIO (Unstructured)::", file=sys.stderr)
        print(scenario_data['state']['unstructured'], file=sys.stderr)
        print(file=sys.stderr)

        print("::CASUALTIES (Unstructured)::", file=sys.stderr)
        print(_build_casualties_string(
            scenario_data['state'].get('casualties', ())), file=sys.stderr)
        print(file=sys.stderr)

    alignment_target_data = None
    if alignment_target_filepath is not None:
        with open(alignment_target_filepath) as f:
            alignment_target_data = json.load(f)

        if print_details:
            print("::ALIGNMENT TARGET::", file=sys.stderr)
            for kdma_value in alignment_target_data['kdma_values']:
                print("{}: {}".format(kdma_value['kdma'], kdma_value['value']),
                      file=sys.stderr)

            print(file=sys.stderr)

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

    # Needed to silence BERT warning messages, see: https://stackoverflow.com/questions/67546911/python-bert-error-some-weights-of-the-model-checkpoint-at-were-not-used-when # noqa
    from transformers import logging
    logging.set_verbosity_error()

    for probe_filepath in probes_filepaths:
        with open(probe_filepath) as f:
            probe_data = json.load(f)

        probe_type = probe_data['type']

        scenario_info = scenario_data['state'].get('unstructured')
        scenario_mission = scenario_data['state'].get('mission')['unstructured']
        state = probe_data['state'].get('unstructured')

        if print_details:
            if state is not None:
                print("::PROBE (STATE)::", file=sys.stderr)
                print(state, file=sys.stderr)

            print("::PROBE PROMPT ({})::".format(probe_type), file=sys.stderr)
            print(probe_data['prompt'], file=sys.stderr)
            print(file=sys.stderr)

            if probe_type == ProbeType.MultipleChoice.value:
                print("::OPTIONS::", file=sys.stderr)
                for option in probe_data.get('options', ()):
                    print("[{}] {}".format(option['id'], option['value']),
                          file=sys.stderr)
                print(file=sys.stderr)

        prompt_for_system = _prepare_prompt(
            scenario_info,
            scenario_mission,
            state,
            probe_data['prompt'],
            scenario_data['state']['casualties'],
            options=probe_data.get('options') if probe_type == ProbeType.MultipleChoice.value else None,  # noqa: 501
            alignment_target=alignment_target_data)

        if print_details:
            print("::PROMPT FOR SYSTEM::", file=sys.stderr)
            print(prompt_for_system, file=sys.stderr)
            print(file=sys.stderr)

        raw_response = str(algorithm.run_inference(prompt_for_system))

        chosen_option = None
        if probe_type == ProbeType.MultipleChoice.value:
            option_values = [str(o['value']) for o in probe_data['options']]
            chosen_option_value = force_choice_with_bert(
                raw_response, option_values)

            idx_of_choice = option_values.index(chosen_option_value)
            chosen_option = probe_data['options'][idx_of_choice]

        if print_details:
            print("::SYSTEM RESPONSE (RAW)::", file=sys.stderr)
            print(raw_response, file=sys.stderr)
            print(file=sys.stderr)

            if chosen_option is not None:
                print("::SELECTED OPTION::", file=sys.stderr)
                print("[{}] {}".format(
                    chosen_option['id'],
                    chosen_option['value']), file=sys.stderr)
                print(file=sys.stderr)

        if chosen_option is not None:
            print(json.dumps(chosen_option, indent=2), file=sys.stderr)


def force_choice_with_bert(text: str, choices: List[str]):
    bertsim = bertsimilarity.BERTSimilarity()

    top_score = -float('inf')
    top_choice = None
    for choice in choices:
        score = bertsim.calculate_distance(text, choice)

        if score > top_score:
            top_score = score
            top_choice = choice

    return top_choice


def _bin_kdma_value(value):
    # Used to map numeric KDMA values to text for injection into
    # prompt
    if value <= 0:
        return "no "
    elif value >= 3:
        return "heavy "
    else:
        return ""


def _build_alignment_target_string(alignment_target):
    *kdma_values, last_kdma_value = alignment_target['kdma_values']

    alignment_target_string_pieces = ["with"]
    for a in kdma_values:
        alignment_target_string_pieces.append(" {}emphasis on {}, ".format(
            _bin_kdma_value(a['value']), a['kdma']))

    if len(kdma_values) > 0:
        alignment_target_string_pieces.append('and')

    alignment_target_string_pieces.append(" {}emphasis on {}".format(
        _bin_kdma_value(last_kdma_value['value']),
        last_kdma_value['kdma']))

    return ''.join(alignment_target_string_pieces)


def _build_casualties_string(casualties):
    #return "\n".join(["{}: {}".format(c['id'], c['unstructured'].strip())
    return "\n".join(["{}".format(c['unstructured'].strip())
                      for c in casualties])


def _build_prompt_options_string(options):
    options_info = ''
    options_code = 'A'
    for option in options:
        options_info += '{}: {}\n'.format(options_code, option['value'])
        options_code = chr(ord(options_code) + 1)

    return options_info
    #return "\n".join(["{}: {}".format(o['id'], o['value'])
    #                  for o in options])


def _prepare_prompt(scenario_info,
                    scenario_mission,
                    probe_state,
                    probe_prompt,
                    casualties,
                    options=None,
                    alignment_target=None):

    if alignment_target is not None:
        alignment_target_string =\
            _build_alignment_target_string(alignment_target)
    else:
        alignment_target_string = None

    if options is not None:
        ammended_prompt = "{} Choose from the following:\n{}".format(
            probe_prompt, _build_prompt_options_string(options))
    else:
        ammended_prompt = probe_prompt

    if scenario_info is not None:
        scenario_info_str = "{}".format(scenario_info)
    else:
        scenario_info_str = ""

    if scenario_mission is not None:
        scenario_mission_str = "{}".format(scenario_mission)
    else:
        scenario_mission_str = ""

    if probe_state is not None:
        probe_state_str = "{}\n".format(probe_state)
    else:
        probe_state_str = ""

    casualties_str = "{}\n".format(_build_casualties_string(casualties))

    return f"""
Instruction: Use medical triage knowledge to answer the following question.
{scenario_info_str}{casualties_str}
Mission for the task is : {scenario_mission_str}
{probe_state_str}{ammended_prompt}
Response: """  # noqa


if __name__ == "__main__":
    sys.exit(main())
