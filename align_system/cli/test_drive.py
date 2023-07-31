import sys

from align_system.interfaces.cli_builder import build_interfaces


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
    run_test_driver(**build_interfaces(add_cli_args, "Test driver script"))


def run_test_driver(interface,
                    model,
                    align_to_target=False,
                    algorithm="llm_baseline",
                    algorithm_kwargs=None,
                    similarity_measure="bert"):
    scenario = interface.start_scenario()

    for probe in scenario.iterate_probes():
        probe_dict = probe.to_dict()
        print(probe.pretty_print_str())
        print()
        # Algo stuff here
        probe.respond(
            {'choice': probe_dict['options'][0]['id'],
             'justification': 'Seems right?'})


if __name__ == "__main__":
    sys.exit(main())
