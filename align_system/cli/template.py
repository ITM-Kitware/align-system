import json

from align_system.interfaces.cli_builder import build_interfaces
from align_system.utils.enums import ProbeType
from align_system.interfaces.abstracts import (
    ScenarioInterfaceWithAlignment,
    ProbeInterfaceWithAlignment)


def add_cli_args(parser):
    # Using argparse to add our system CLI specific arguments.  Can
    # modify or add your own custom CLI arguments here
    parser.add_argument('-m', '--model',
                        type=str,
                        help="Example command-line argument")
    parser.add_argument('-t', '--align-to-target',
                        action='store_true',
                        default=False,
                        help="Align algorithm to target KDMAs")


def main():
    # The `build_interfaces` call here adds all interfaces as
    # subparsers to your CLI.  (Can specify what interfaces you
    # support explicitly with the optional `supported_interfaces`
    # argument (as a set))
    # The `build_interfaces` call also instantiates an interface
    # object based on the selected interface and interface arguments
    # provided at the command line and passes them to your run
    # function (`run_custom_system` in this case)
    run_custom_system(**build_interfaces(add_cli_args, "My ALIGN System CLI"))


def run_custom_system(interface,
                      model,
                      align_to_target=False,):
    scenario = interface.start_scenario()
    scenario_dict = scenario.to_dict()

    if align_to_target:
        alignment_target = scenario.get_alignment_target()
        alignment_target_dict = alignment_target.dict()

    # DO ALGORITHM SETUP THINGS HERE

    for probe in scenario.iterate_probes():
        print(probe.pretty_print_str())
        print()

        probe_dict = probe.to_dict()

        # DO ALGORITHM THINGS HERE

        # Placeholder value:
        justification_str = "This seems like the correct answer"

        if probe_dict['type'] == ProbeType.MultipleChoice.value:
            # Placeholder value:
            selected_choice_id = probe_dict['options'][0]['id']  # First option

            probe_response = {'justification': justification_str,
                              'choice': selected_choice_id}
        else:
            probe_response = {'justification': justification_str}

        print(json.dumps(probe_response, indent=2))
        print()

        probe.respond(probe_response)

        # Get KDMA Alignment scores for probe if the interface supports it
        if isinstance(probe, ProbeInterfaceWithAlignment):
            probe_alignment_results = probe.get_alignment_results()
            print(json.dumps(probe_alignment_results, indent=2))
            print()

    # Get KDMA Alignment scores for scenario if the interface supports it
    if isinstance(scenario, ScenarioInterfaceWithAlignment):
        scenario_alignment_results = scenario.get_alignment_results()
        print(json.dumps(scenario_alignment_results, indent=2))
        print()


if __name__ == "__main__":
    main()
