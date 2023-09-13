import json

from align_system.interfaces.cli_builder import build_interfaces
from align_system.algorithms.llm_baseline import LLMBaseline
from align_system.algorithms.llama_index import LlamaIndex
from align_system.similarity_measures import build_force_choice_func
from align_system.prompt_engineering.common import (
    prepare_action_based_prompt,
    prepare_treatment_selection_prompt)


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
    run_action_based_align_system(
        **build_interfaces(add_cli_args, "ALIGN System CLI",
                           supported_interfaces={'TA3ActionBased'}))


# TODO: Find a better location for this (or pull in from TA3 client
# module)
TREATMENT_LOCATIONS = ['right forearm',
                       'left forearm',
                       'right calf',
                       'left calf',
                       'right thigh',
                       'left thigh',
                       'right stomach',
                       'left stomach',
                       'right bicep',
                       'left bicep',
                       'right shoulder',
                       'left shoulder',
                       'right side',
                       'left side',
                       'right chest',
                       'left chest',
                       'right wrist',
                       'left wrist',
                       'left face',
                       'right face',
                       'left neck',
                       'right neck',
                       'unspecified']


def run_action_based_align_system(interface,
                                  model,
                                  align_to_target=False,
                                  algorithm="llm_baseline",
                                  algorithm_kwargs=None,
                                  similarity_measure="bert"):
    scenario = interface.start_scenario()
    scenario_dict = scenario.to_dict()

    if align_to_target:
        alignment_target_dict = scenario.get_alignment_target()

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

    current_state = scenario.get_state()
    scenario_complete = current_state.get('scenario_complete', False)

    while not scenario_complete:
        available_actions = scenario.get_available_actions()

        # DO ALGORITHM THINGS HERE

        # prompt = prepare_raw_json_prompt(current_state, available_actions)
        available_actions_unstructured =\
            [a['unstructured'] for a in available_actions]

        prompt = prepare_action_based_prompt(
            scenario_dict['state']['unstructured'],
            current_state['mission'].get('unstructured'),
            current_state['unstructured'],
            current_state['casualties'],
            available_actions_unstructured,
            alignment_target=alignment_target_dict if align_to_target else None
        )
        print("* Prompt for ADM: {}".format(prompt))

        raw_response = str(algorithm.run_inference(prompt))
        print("* ADM raw response: {}".format(raw_response))

        selected_action_idx, selected_action = force_choice_func(
            raw_response, available_actions_unstructured)

        print("* Mapped selection: '{}'".format(selected_action))

        action_to_take = available_actions[selected_action_idx]

        if action_to_take['action_type'] == 'APPLY_TREATMENT':
            # Ask the system to specify the treatment to use and where

            # First casualty with the matching ID (should only be one)
            casualty_id = action_to_take['casualty_id']
            matching_casualties = [c for c in current_state['casualties']
                                   if c['id'] == casualty_id]

            assert len(matching_casualties) == 1
            casualty_to_treat = matching_casualties[0]

            treatment_prompt = prepare_treatment_selection_prompt(
                casualty_to_treat['unstructured'],
                casualty_to_treat['vitals'],
                current_state['supplies'])

            print("** Treatment prompt for ADM: {}".format(treatment_prompt))

            raw_treatment_response =\
                str(algorithm.run_inference(treatment_prompt))

            print("** ADM raw treatment response: {}".format(
                raw_treatment_response))

            # Map response to treatment and treatment location
            _, treatment = force_choice_func(
                raw_treatment_response,
                [s['type'] for s in current_state['supplies']])

            _, treatment_location = force_choice_func(
                raw_treatment_response,
                TREATMENT_LOCATIONS)

            print("** Mapped treatment selection: '{}: {}'".format(
                treatment, treatment_location))

            # Populate required parameters for treatment action
            action_to_take['parameters'] = {
                'treatment': treatment,
                'location': treatment_location}

        import xdev
        with xdev.EmbedOnException():
            current_state = scenario.take_action(action_to_take)

        scenario_complete = current_state.get('scenario_complete', False)


if __name__ == "__main__":
    main()
