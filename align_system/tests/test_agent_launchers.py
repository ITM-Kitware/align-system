"""Launcher dry runs must select the intended, distinguishable experiments."""
import os
from pathlib import Path
import shlex
import subprocess

from hydra import compose, initialize_config_dir
import pytest


ROOT = Path(__file__).resolve().parents[2]


def dry_run(overrides=(), launcher='phase2_ow_agent_qwen38.sh', **settings):
    env = os.environ.copy()
    for name in ('EXPERIMENT', 'TOOLKIT', 'ASSESSMENT_MODE', 'AUTO_ASSESS', 'TARGET_CONTEXT',
                 'RUN_USERNAME', 'OUT_DIR', 'SESSION_TYPE', 'SCENARIO_ID', 'MODEL',
                 'PROJECT_DIR', 'OPENAI_MODEL', 'TA3_ENDPOINT', 'REQUIRE_SCORES',
                 'ADM_PROFILE', 'SLURM_ARRAY_TASK_ID'):
        env.pop(name, None)
    env.update(DRY_RUN='1', RUN_MODE='smoke', MODEL_BACKEND='vllm',
               VLLM_URL='http://model.example:8000/v1', SLURM_JOB_ID='42',
               SLURM_SUBMIT_DIR=str(ROOT))
    env.update(settings)
    result = subprocess.run(['bash', str(ROOT / launcher), *overrides],
                            cwd=ROOT, env=env, capture_output=True, text=True, timeout=10)
    return result


def composed(result):
    assert result.returncode == 0, result.stderr
    command = next(line for line in result.stdout.splitlines() if line.startswith('run_align_system '))
    overrides = shlex.split(command)[1:]
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / 'align_system/configs')):
        cfg = compose(config_name='action_based', overrides=overrides, return_hydra_config=True)
    return cfg


@pytest.mark.parametrize('settings', [
    {'TOOLKIT': 'baseline', 'TARGET_CONTEXT': 'true'},
    {'EXPERIMENT': 'phase2_langchain_agent_ow_target_live'},
])
def test_target_baseline_has_no_scorer(settings):
    cfg = composed(dry_run(**settings))
    assert cfg.driver.target_context._target_.endswith('.TargetContext')
    assert not cfg.driver.extra_tools
    assert '-TargetAware' in cfg.interface.username


def test_control_overrides_update_generated_names():
    cfg = composed(dry_run([
        '+driver/target_context@driver.target_context=target']))
    assert cfg.driver.target_context._target_.endswith('.TargetContext')
    assert '-TargetAware' in cfg.interface.username
    cfg = composed(dry_run(['driver.target_context=null'], TARGET_CONTEXT='true'))
    assert cfg.driver.target_context is None
    assert '-TargetAware' not in cfg.interface.username


def test_invalid_target_context_fails_before_launch():
    assert dry_run(TARGET_CONTEXT='maybe').returncode == 2
