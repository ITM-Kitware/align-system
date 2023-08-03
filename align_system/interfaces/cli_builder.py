import argparse
import sys

from align_system.interfaces.ta3_caci_service import TA3CACIServiceInterface
from align_system.interfaces.local_files import LocalFilesInterface

INTERFACES = {'TA3': TA3CACIServiceInterface,
              'LocalFiles': LocalFilesInterface}


def build_interfaces(add_args_func,
                     description,
                     supported_interfaces=INTERFACES.keys()):
    def _build_combined_parser(building_for_help_text=False):
        parser = argparse.ArgumentParser(description=description)

        # https://docs.python.org/3.8/library/argparse.html?highlight=argparse#sub-commands
        subparsers = parser.add_subparsers(
            help='Select interface.  Adding --help after interface selection '
                 'will print interface and system specified arguments',
            required=True)

        for interface_name in supported_interfaces:
            interface_parser = subparsers.add_parser(
                interface_name,
                help=INTERFACES[interface_name].cli_parser_description())

            interface_parser =\
                INTERFACES[interface_name].cli_parser(interface_parser)

            # Add ADM specific command line arguments to each interfaces
            # parser; adding this to each subparser is needed to ensure
            # the arguments are shown in the `--help` text
            if building_for_help_text:
                add_args_func(interface_parser)

        return parser

    # Program will exit after the following call if help was requested
    _build_combined_parser(building_for_help_text=True).parse_args()

    # Rebuild the parser with just the interface pieces for
    # instantiating the selected interface object
    interface_parser = _build_combined_parser()
    interface_parsed_args, remaining_args = interface_parser.parse_known_args()

    # Build parser from provided `add_args_func` for parsing out just
    # user specified args
    userland_parser = argparse.ArgumentParser()
    add_args_func(userland_parser)

    userland_parsed_args = userland_parser.parse_args(remaining_args)

    # sys.argv[1] SHOULD always be the selected interface (subparser)
    # name if help text was not requested
    assert sys.argv[1] in supported_interfaces
    selected_interface_class = INTERFACES[sys.argv[1]]
    interface = selected_interface_class.init_from_parsed_args(
        interface_parsed_args)

    return {'interface': interface,
            **vars(userland_parsed_args)}
