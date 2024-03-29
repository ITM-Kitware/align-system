import sys
import yaml

from rich.highlighter import JSONHighlighter

from align_system.utils import logging
from align_system.interfaces.cli_builder import build_interfaces
from align_system.algorithms.kaleido_adm import KaleidoADM
from align_system.prompt_engineering.common import prepare_prompt
from align_system.utils.enums import ProbeType
from align_system.interfaces.abstracts import (
    ScenarioInterfaceWithAlignment,
    ProbeInterfaceWithAlignment)
from align_system.algorithms.lib.util import format_template


log = logging.getLogger(__name__)
JSON_HIGHLIGHTER = JSONHighlighter()


def add_cli_args(parser):
    parser.add_argument('-l', '--loglevel',
                        type=str,
                        default='INFO')
    parser.add_argument('-k', '--kdma-descriptions',
                        type=str,
                        help="Filepath to YAML of KDMA descriptions")


def main():
    log.debug(f"[bright_black]CMD: {' '.join(sys.argv)}[/bright_black]",
              extra={'markup': True, 'highlighter': None})
    run_align_system(
        **build_interfaces(add_cli_args, "ALIGN System CLI for the Kaleido Algorithm",
                           supported_interfaces={'LocalFiles',
                                                 'TA1Soartech',
                                                 'TA1Adept'}))


def run_align_system(interface,
                     loglevel="INFO",
                     kdma_descriptions=None):
    # Set log level on root logger (such that child loggers respect
    # the set log level)
    logging.getLogger().setLevel(loglevel)

    scenario = interface.start_scenario()
    scenario_dict = scenario.to_dict()

    alignment_target_dict = scenario.get_alignment_target()

    kaleido = KaleidoADM(model_name='allenai/kaleido-large', use_tqdm=False)

    if kdma_descriptions is not None:
        with open(kdma_descriptions) as f:
            kdma_descriptions_data = yaml.safe_load(f)

        kdma_descriptions_map = {k: v['description'] for k, v
                                 in kdma_descriptions_data.items()}
    else:
        kdma_descriptions_map = None

    for probe in scenario.iterate_probes():
        probe_dict = probe.to_dict()

        casualties_dicts = scenario_dict['state'].get('casualties', [])

        try:
            mission_unstructured =\
                scenario_dict['state']['mission']['unstructured']
        except KeyError:
            mission_unstructured = None

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

        prompt = prepare_prompt(
            scenario_dict['state']['unstructured'],
            mission_unstructured,
            state_unstructured,
            probe_dict['prompt'],
            [],  # casualties
        )
        log.info("[bold]* Prompt for ADM *[/bold]",
                 extra={"markup": True})
        log.info(prompt)

        prompt_template = '{{probe_prompt}} "{{choice}}" instead of {{other_choices}} in the context of {{scenario_state_unstructured}}'
        # prompt_template = '{{probe_prompt}} "{{choice}}" instead of {{other_choices}}'

        prompt_template_partial = format_template(
            prompt_template,
            partial=True,
            allow_extraneous=True,
            probe_prompt=probe_dict['prompt'],
            scenario_state_unstructured=scenario_dict['state']['unstructured'])

        choices = [str(o['value']) for o in probe_dict['options']]

        kaleido_results = kaleido.estimate_kdma_values(
            prompt_template_partial,
            choices,
            {k['kdma'].lower(): k['value'] for k in alignment_target_dict.get('kdma_values', ())},
            kdma_descriptions_map=kdma_descriptions_map)

        selected_choice_idx = kaleido.force_choice(
            kaleido_results,
            choices)

        selected_choice_id =\
            probe_dict['options'][selected_choice_idx]['id']

        probe.respond({'choice': selected_choice_id})

        if isinstance(probe, ProbeInterfaceWithAlignment):
            probe_alignment_results = probe.get_alignment_results()
            log.info("* Probe alignment score: {:0.5f}".format(
                    probe_alignment_results['score']))

    if isinstance(scenario, ScenarioInterfaceWithAlignment):
        scenario_alignment_results = scenario.get_alignment_results()
        log.info("* Scenario alignment score: {:0.5f}".format(
            scenario_alignment_results['score']))


if __name__ == "__main__":
    main()
