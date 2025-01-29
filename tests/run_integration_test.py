import argparse
from difflib import unified_diff
import os
import tempfile
import atexit
import shutil
import sys
import subprocess
import logging
import re

from rich.logging import RichHandler
from rich.highlighter import NullHighlighter

log = logging.getLogger(__name__)
LOGGING_FORMAT = "%(message)s"
logging.basicConfig(
    level="INFO",
    format=LOGGING_FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler(highlighter=NullHighlighter())])


TESTS_DIR = os.path.abspath(os.path.dirname(__file__))
EXPECTED_OUT_GITIGNORE_CONTENT = '''
*
!.gitignore

!input_output.json
!raw_align_system.log'''.lstrip()


def main():
    parser = argparse.ArgumentParser(
        description="Run integration test")

    parser.add_argument('experiment',
                        type=str,
                        help="Hydra experiment config to run")
    parser.add_argument("-r", "--replace-expected-outputs",
                        action='store_true',
                        default=False,
                        help="Replace expected output files "
                             "(for updating / generating test "
                             "outputs)")
    parser.add_argument("-k", "--keep-temporary-outdir",
                        action='store_true',
                        default=False,
                        help="Don't delete temporary output "
                             "directory (useful for debugging)")
    parser.add_argument("-s", "--show-experiment-stdout",
                        action='store_true',
                        default=False,
                        help="Show stdout from experiment as it's running")

    run_integration_test(**vars(parser.parse_args()))


def markup_diff_lines(diff_lines):
    marked_up_diff_lines = []
    for line in map(str.rstrip, diff_lines):
        if re.match(r'^(\-\-\-|\+\+\+)\s', line):
            marked_up_diff_lines.append('[bold]{}[/bold]'.format(line))
        elif re.match(r'^\-\s', line):
            marked_up_diff_lines.append('[red]{}[/red]'.format(line))
        elif re.match(r'^\+\s', line):
            marked_up_diff_lines.append('[green]{}[/green]'.format(line))
        elif re.match(r'^@@\s', line):
            marked_up_diff_lines.append('[cyan]{}[/cyan]'.format(line))
        else:
            marked_up_diff_lines.append(line)

    return marked_up_diff_lines


def compare_text_files_with_diff(experiment,
                                 expected_outdir,
                                 run_outdir,
                                 file_basename):
    log.info("[{}] Checking `{}`.. ".format(experiment, file_basename))
    expected_fp = os.path.join(expected_outdir, file_basename)
    run_fp = os.path.join(run_outdir, file_basename)

    with open(expected_fp) as expected, open(run_fp) as run:
        diff = unified_diff(list(expected),
                            list(run),
                            fromfile=expected_fp,
                            tofile=run_fp,
                            lineterm='')

        diff_lines = list(diff)

        if len(diff_lines) == 0:
            log.info("[green]OK[/green]", extra={"markup": True})
            return True
        else:
            log.info("[red]Different[/red]", extra={"markup": True})
            log.info('\n'.join(markup_diff_lines(diff_lines)),
                     extra={"markup": True})
            return False


def run_integration_test(experiment,
                         replace_expected_outputs=False,
                         keep_temporary_outdir=False,
                         show_experiment_stdout=False):
    expected_outputs_dir = os.path.join(
        TESTS_DIR, 'data', 'expected_outputs', experiment)

    if replace_expected_outputs:
        # Ensure we're starting fresh for the files we plan to check
        # against
        expected_input_output_fp = os.path.join(expected_outputs_dir, 'input_output.json')
        if os.path.exists(expected_input_output_fp):
            os.remove(expected_input_output_fp)
        expected_raw_log_fp = os.path.join(expected_outputs_dir, 'raw_align_system.log')
        if os.path.exists(expected_raw_log_fp):
            os.remove(expected_raw_log_fp)

        run_dir = expected_outputs_dir

        os.makedirs(run_dir, exist_ok=True)

        # Write a .gitignore to the expected outputs dir for the run
        # to limit git tracking only to output files of interest for
        # testing
        with open(os.path.join(run_dir, '.gitignore'), 'w') as f:
            print(EXPECTED_OUT_GITIGNORE_CONTENT, file=f)
    else:
        temporary_outdir = tempfile.mkdtemp()

        def _remove_tempdir_closure(tempdir):
            # Capture the 'tempdir' path we want to close to maintain
            # a reference (using a closure here saves us if variable
            # value changes further along in the code
            def _remove_tempdir():
                shutil.rmtree(tempdir)
                log.info('Cleaned up run directory: "{}"'.format(run_dir))

            return _remove_tempdir

        if not keep_temporary_outdir:
            atexit.register(_remove_tempdir_closure(temporary_outdir))

        run_dir = temporary_outdir

    log.info('Saving outputs to: "{}"'.format(run_dir))

    run_experiment_command = [
        'run_align_system',
        '+test_data_dir={}'.format(os.path.join(TESTS_DIR, 'data')),
        '+experiment={}'.format(experiment),
        'hydra.run.dir={}'.format(run_dir)]

    log.info('Running: "{}"'.format(' '.join(run_experiment_command)))

    try:
        _ = subprocess.run(
            run_experiment_command,
            check=True,
            capture_output=(not show_experiment_stdout),
            encoding='utf-8')
    except subprocess.CalledProcessError as e:
        log.error(e.stderr)
        raise e

    # Break before doing any checks against expected outputs (since we
    # just generated them)
    if replace_expected_outputs:
        return

    all_good = True
    # Compare run outputs vs. expected outputs
    all_good &= compare_text_files_with_diff(experiment,
                                             expected_outputs_dir,
                                             run_dir,
                                             'raw_align_system.log')

    all_good &= compare_text_files_with_diff(experiment,
                                             expected_outputs_dir,
                                             run_dir,
                                             'input_output.json')

    if not all_good:
        # Exit with status of 1, indicating failure
        exit(1)


if __name__ == "__main__":
    main()
