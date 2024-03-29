import sys
import json

from rich.highlighter import JSONHighlighter

from align_system.utils import logging
from align_system.interfaces.cli_builder import build_interfaces
# from align_system.utils.enums import ProbeType
from align_system.interfaces.abstracts import (
    ScenarioInterfaceWithAlignment,
    ProbeInterfaceWithAlignment)

from align_system.algorithms.llama_2_single_kdma_adm import Llama2SingleKDMAADM


'''
run_chat_baseline LocalFiles -s example_data/scenario_1/scenario.json -p example_data/scenario_1/probe{1,2,3,4}.json
'''

log = logging.getLogger(__name__)
JSON_HIGHLIGHTER = JSONHighlighter()


def add_cli_args(parser):
    # Using argparse to add our system CLI specific arguments.  Can
    # modify or add your own custom CLI arguments here
    parser.add_argument('-m', '--model',
                        type=str,
                        help="Example command-line argument",
                        default='meta-llama/Llama-2-13b-chat-hf')
    parser.add_argument('-r', '--precision',
                        type=str,
                        help="Precision, must be 'full' or 'half' "
                             "(default: 'full')",
                        default='full')
    parser.add_argument('-t', '--align-to-target',
                        action='store_true',
                        default=False,
                        help="Align algorithm to target KDMAs")
    parser.add_argument('-l', '--loglevel',
                        type=str,
                        default='INFO')


def main():
    # The `build_interfaces` call here adds all interfaces as
    # subparsers to your CLI.  (Can specify what interfaces you
    # support explicitly with the optional `supported_interfaces`
    # argument (as a set))
    # The `build_interfaces` call also instantiates an interface
    # object based on the selected interface and interface arguments
    # provided at the command line and passes them to your run
    # function (`run_custom_system` in this case)
    log.debug(f"[bright_black]CMD: {' '.join(sys.argv)}[/bright_black]",
              extra={'markup': True, 'highlighter': None})
    run_custom_system(**build_interfaces(add_cli_args, "ALIGN System CLI - Chat Model"))


def run_custom_system(interface,
                      model,
                      precision,
                      align_to_target,
                      loglevel="INFO"):
    # Set log level on root logger (such that child loggers respect
    # the set log level)
    logging.getLogger().setLevel(loglevel)

    log.info('Creating algorithm')
    algorithm = Llama2SingleKDMAADM(hf_model=model, precision=precision)

    algorithm.load_model()

    while scenario := interface.start_scenario():
        scenario_dict = scenario.to_dict()

        if align_to_target:
            alignment_target_dict = scenario.get_alignment_target()

        for probe in scenario.iterate_probes():
            log.info(probe.pretty_print_str())

            probe_dict = probe.to_dict()

            casualties_dicts = scenario_dict['state'].get('casualties', [])

            mission_unstructured =\
                scenario_dict['state']['mission'].get('unstructured', '')
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

            # Seems like the probe 'type' is incorrect for at least some
            # probes, always assuming multiple choice here
            # if probe_dict['type'] == ProbeType.MultipleChoice.value:
            #     probe_options_dicts = probe_dict['options']
            # else:
            #     probe_options_dicts = None

            probe_options_dicts = probe_dict['options']

            # TODO extract this prompt-building logic into a separate function/file
            # For the MVP2 ADEPT scenarios, the casualties don't have 'unstructured' text
            # casualties_str = ''
            # for casulaty in casualties_dicts:
            #     casualties_str += casulaty["unstructured"] + " " + str(casulaty["vitals"])

            # question = f"#     Scenario:\n{scenario_dict['state']['unstructured']}\n{mission_unstructured}\n#     Casualties:\n{casualties_str}\n# Question:\n{probe_dict['prompt']}"
            question = f"#     Scenario:\n{scenario_dict['state']['unstructured']}\n{mission_unstructured}\n#     Question:\n{probe_dict['prompt']}"
            options = [option['value'] for option in probe_options_dicts]

            for _ in range(5): # TODO make this a parameter
                # TODO a possible improvement would be to use a separate prompt to parse     mis-formatted JSON instead of simply trying again
                if align_to_target:
                    target = {kdma['kdma'].lower(): kdma['value']
                              for kdma in alignment_target_dict['kdma_values']}
                    explanation, action_idx =\
                        algorithm.run_aligned_decision_maker_with_voting(
                            question,
                            options,
                            target)

                    log.info("* ADM Selected: {}".format(
                        options[action_idx]))

                    log.info("* ADM Explanation: {}".format(explanation))
                else:
                    dialog = algorithm.build_multiple_choice_dialog(
                        question, options)

                    log.debug("[bold]*DIALOG*[/bold]", extra={"markup": True})
                    algorithm.log_dialog(dialog)

                    raw_response = algorithm.respond_to_dialog(dialog)

                    log.info("* ADM raw response: {}".format(raw_response))

                    parsed_output = Llama2SingleKDMAADM.attempt_generic_parse(
                        raw_response, ['Reasoning', 'Answer'])

                    if parsed_output is None:
                        explanation, action_idx =\
                            Llama2SingleKDMAADM.parse_generated_output(
                                raw_response)
                    else:
                        explanation = parsed_output['Reasoning']
                        action_idx = parsed_output['Answer']

                if explanation is not None and action_idx is not None:
                    if len(options) > action_idx:
                        break
                    else:
                        log.info('** Selected action_idx out of range of '
                                 'available actions, retrying!')
                        continue

                log.info('** Failed to parse')

            # if probe_dict['type'] == ProbeType.MultipleChoice.value:
            #     probe_response = {'justification': explanation,
            #                       'choice': probe_options_dicts[action_idx]['id']}
            # else:
            #     probe_response = {'justification': explanation}

            probe_response = {'justification': explanation,
                              'choice': probe_options_dicts[action_idx]['id']}

            log.debug("[bold]*PROBE RESPONSE*[/bold]",
                      extra={"markup": True})
            log.debug(json.dumps(probe_response, indent=4),
                      extra={"highlighter": JSON_HIGHLIGHTER})

            probe.respond(probe_response)

            if isinstance(probe, ProbeInterfaceWithAlignment):
                probe_alignment_results = probe.get_alignment_results()
                log.info("* Probe alignment score: {}".format(
                    probe_alignment_results['score']))

        if isinstance(scenario, ScenarioInterfaceWithAlignment):
            scenario_alignment_results = scenario.get_alignment_results()
            log.info("* Scenario alignment score: {}".format(
                scenario_alignment_results['score']))



if __name__ == "__main__":
    main()
