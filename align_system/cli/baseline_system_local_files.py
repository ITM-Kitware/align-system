import argparse
import json
import sys

from align_system.algorithms.llm_baseline import LLMBaseline
from align_system.algorithms.llama_index import LlamaIndex
from align_system.utils.enums import ProbeType
from align_system.prompt_engineering.common import (
    build_casualties_string, prepare_prompt)
from align_system.similarity_measures import force_choice


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
                        nargs='*',
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
    parser.add_argument('--similarity-measure',
                        type=str,
                        default="bert",
                        help="Similarity measure to use (default: 'bert'")

    run_baseline_system_local_filepath(**vars(parser.parse_args()))

    return 0


def run_baseline_system_local_filepath(
        probes_filepaths,
        scenario_filepath,
        model,
        alignment_target_filepath=None,
        print_details=False,
        algorithm="llm_baseline",
        algorithm_kwargs=None,
        similarity_measure="bert"):
    with open(scenario_filepath) as f:
        scenario_data = json.load(f)

    if print_details:
        print("::SCENARIO (Unstructured)::", file=sys.stderr)
        print(scenario_data['state']['unstructured'], file=sys.stderr)
        print(file=sys.stderr)

        print("::CASUALTIES (Unstructured)::", file=sys.stderr)
        print(build_casualties_string(
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

    # Get the chosen similarity measure function
    if similarity_measure == "bert":
        from align_system.similarity_measures.bert import (
            build_bert_similarity_measure_func)
        similarity_measure_func = build_bert_similarity_measure_func()
    elif similarity_measure == "heuristic":
        from align_system.similarity_measures.heuristics import (
            score_string_similarity)
        similarity_measure_func = score_string_similarity
    else:
        raise NotImplementedError("Unrecognized similarity measure '{}', "
                                  "aborting!".format(similarity_measure))

    # Load the system / model
    algorithm_kwargs_parsed = {}
    if algorithm_kwargs is not None:
        algorithm_kwargs_parsed = json.loads(algorithm_kwargs)

    if algorithm == "llm_baseline":
        algorithm = LLMBaseline(
            model_use=model, distributed=False,
            **algorithm_kwargs_parsed)
    elif algorithm == "llama_index":
        algorithm = LlamaIndex(
            model_name=model,
            **algorithm_kwargs_parsed)

    algorithm.load_model()

    probes = []
    if len(probes_filepaths) > 0:
        for probe_filepath in probes_filepaths:
            with open(probe_filepath) as f:
                probes.append(json.load(f))
    else:
        # Assume probes are embedded in the scenario file
        probes.extend(scenario_data.get('probes', []))

    if len(probes) == 0:
        raise RuntimeError("No probes specified, and none found in scenario")

    for probe_data in probes:
        probe_type = probe_data['type']

        scenario_info = scenario_data['state'].get('unstructured')
        scenario_mission =\
            scenario_data['state'].get('mission')['unstructured']
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

        prompt_for_system = prepare_prompt(
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
            _, chosen_option_value = force_choice(
                raw_response, option_values, similarity_measure_func)

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


if __name__ == "__main__":
    sys.exit(main())