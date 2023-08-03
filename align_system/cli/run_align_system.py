import json

from align_system.interfaces.cli_builder import build_interfaces
from align_system.algorithms.llm_baseline import LLMBaseline
from align_system.algorithms.llama_index import LlamaIndex
from align_system.similarity_measures import build_force_choice_func
from align_system.prompt_engineering.common import prepare_prompt
from align_system.utils.enums import ProbeType


def add_cli_args(parser):
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
    parser.add_argument('--similarity-measure',
                        type=str,
                        default="bert",
                        help="Similarity measure to use (default: 'bert')")


def main():
    run_test_driver(**build_interfaces(add_cli_args, "ALIGN System CLI"))


def run_test_driver(interface,
                    model,
                    align_to_target=False,
                    algorithm="llm_baseline",
                    algorithm_kwargs=None,
                    similarity_measure="bert"):
    scenario = interface.start_scenario()
    scenario_dict = scenario.to_dict()

    if align_to_target:
        alignment_target = scenario.get_alignment_target()
        alignment_target_dict = alignment_target.dict()

    force_choice_func = build_force_choice_func(similarity_measure)

    # Load the system / model
    algorithm_kwargs_parsed = {}
    if algorithm_kwargs is not None:
        algorithm_kwargs_parsed = json.loads(algorithm_kwargs)

    if algorithm == "llm_baseline":
        algorithm = LLMBaseline(
            model_use=model, distributed=False,
            **algorithm_kwargs_parsed)
    elif algorithm == "llama_index":
        # TODO: This is a hacky way to have the "Knowledge" KDMA
        # determine whether or not domain documents should be loaded.
        # Should remove, or move to llama_index code
        if align_to_target:
            for kdma_dict in alignment_target_dict.get('kdma_values', ()):
                if kdma_dict['kdma'].lower() == 'knowledge':
                    if kdma_dict['value'] > 1:
                        print("** Setting 'retrieval_enabled' to True based "
                              "on 'Knowledge' KDMA value ({})".format(
                                  kdma_dict['value']))
                        algorithm_kwargs_parsed['retrieval_enabled'] = True
                    else:
                        print("** Setting 'retrieval_enabled' to False based "
                              "on 'Knowledge' KDMA value ({})".format(
                                  kdma_dict['value']))
                        algorithm_kwargs_parsed['retrieval_enabled'] = False

                    break

        algorithm = LlamaIndex(
            model_name=model,
            **algorithm_kwargs_parsed)

    algorithm.load_model()

    for probe in scenario.iterate_probes():
        probe_dict = probe.to_dict()

        if len(probe_dict['state']) > 0:
            casualties_dicts = probe_dict['state'].get('casualties', [])
            mission_unstructured =\
                probe_dict['state']['mission']['unstructured']
            state_unstructured = probe_dict['state']['unstructured']
        else:
            casualties_dicts = scenario_dict['state'].get('casualties', [])
            mission_unstructured =\
                scenario_dict['state']['mission']['unstructured']
            state_unstructured = None

        if probe_dict['type'] == ProbeType.MultipleChoice.value:
            probe_options_dicts = probe_dict['options']
        else:
            probe_options_dicts = None

        prompt = prepare_prompt(
            scenario_dict['state']['unstructured'],
            mission_unstructured,
            state_unstructured,
            probe_dict['prompt'],
            casualties_dicts,
            options=probe_options_dicts,
            alignment_target=alignment_target_dict if align_to_target else None
        )
        print("* Prompt for ADM: {}".format(prompt))

        raw_response = str(algorithm.run_inference(prompt))
        print("* ADM Raw response: {}".format(raw_response))

        if probe_dict['type'] == ProbeType.MultipleChoice.value:
            selected_choice_idx, selected_choice = force_choice_func(
                raw_response, [o['value'] for o in probe_dict['options']])
            print("* ADM Selected: '{}'".format(selected_choice))

            selected_choice_id =\
                probe_dict['options'][selected_choice_idx]['id']

            probe.respond({'justification': raw_response,
                           'choice': selected_choice_id})
        else:
            probe.respond({'justification': raw_response})


if __name__ == "__main__":
    main()
