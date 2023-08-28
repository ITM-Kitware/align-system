import json

from align_system.interfaces.cli_builder import build_interfaces
from align_system.utils.enums import ProbeType

from align_system.algorithms.llm_chat_baseline import LLMChatBaseline


'''
run_chat_baseline LocalFiles -s example_data/scenario_1/scenario.json -p example_data/scenario_1/probe{1,2,3,4}.json
'''

def add_cli_args(parser):
    # Using argparse to add our system CLI specific arguments.  Can
    # modify or add your own custom CLI arguments here
    # parser.add_argument('-m', '--model',
    #                     type=str,
    #                     help="Example command-line argument")
    # parser.add_argument('-t', '--align-to-target',
    #                     action='store_true',
    #                     default=False,
    #                     help="Align algorithm to target KDMAs")
    pass


def main():
    # The `build_interfaces` call here adds all interfaces as
    # subparsers to your CLI.  (Can specify what interfaces you
    # support explicitly with the optional `supported_interfaces`
    # argument (as a set))
    # The `build_interfaces` call also instantiates an interface
    # object based on the selected interface and interface arguments
    # provided at the command line and passes them to your run
    # function (`run_custom_system` in this case)
    run_custom_system(**build_interfaces(add_cli_args, "ALIGN System CLI - Chat Model"))


def run_custom_system(interface):
    scenario = interface.start_scenario()
    scenario_dict = scenario.to_dict()

    # if align_to_target:
    #     alignment_target = scenario.get_alignment_target()
    #     alignment_target_dict = alignment_target.dict()

    # DO ALGORITHM SETUP THINGS HERE
    print('Creating algorithm')
    algorithm = LLMChatBaseline(hf_model='meta-llama/Llama-2-13b-chat-hf', precision='half')
    # algorithm = LLMChatBaseline()
    algorithm.load_model()

    for probe in scenario.iterate_probes():
        print(probe.pretty_print_str())
        print()

        probe_dict = probe.to_dict()

        # DO ALGORITHM THINGS HERE
        
        casualties_dicts = scenario_dict['state'].get('casualties', [])
        mission_unstructured =\
            scenario_dict['state']['mission']['unstructured']
        state_unstructured = None

        if 'state' in probe_dict:
            probe_state = probe_dict['state']
            if 'casualties' in probe_state:
                casualties_dicts = probe_dict['state']['casualties']

            if('mission' in probe_state and
               'unstructured' in probe_state['mission']):
                mission_unstructured =\
                  probe_state['mission']['unstructured']

            if 'unstructured' in probe_state:
                state_unstructured = probe_state['unstructured']

        if probe_dict['type'] == ProbeType.MultipleChoice.value:
            probe_options_dicts = probe_dict['options']
        else:
            probe_options_dicts = None
        
        # TODO extract this prompt-building logic into a separate function/file
        casualties_str = ''
        for casulaty in casualties_dicts:
            casualties_str += casulaty["unstructured"] + " " + str(casulaty["vitals"])
        
        question = f"# Scenario:\n{scenario_dict['state']['unstructured']}\n{mission_unstructured}\n# Casualties:\n{casualties_str}\n# Question:\n{probe_dict['prompt']}"
        options = [option['value'] for option in probe_options_dicts]
        
        for _ in range(5): # TODO make this a parameter
            # TODO a possible improvement would be to use a separate prompt to parse mis-formatted JSON instead of simply trying again
            generated_output, justification_str, selected_choice_id, probabilities = algorithm.answer_multiple_choice(question, options)
            if justification_str is not None and selected_choice_id is not None:
                break
            print('Failed to parse:', generated_output)
        

        if probe_dict['type'] == ProbeType.MultipleChoice.value:

            probe_response = {'justification': justification_str,
                              'choice': probe_options_dicts[selected_choice_id]}
        else:
            probe_response = {'justification': justification_str}

        print(json.dumps(probe_response, indent=2))
        print()

        probe.respond(probe_response)


if __name__ == "__main__":
    main()
