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


def test_april_affiliation_launcher_uses_server_target_without_actor_override():
    cfg = composed(dry_run(
        launcher='phase2_ow_agent_qwen38_kaleido_april_affiliation0.sbatch'))
    assert cfg.interface.adm_profile == 'APRIL_AFFILIATION_0'
    assert cfg.interface.session_type == 'eval'
    assert cfg.interface.training_session is None
    assert list(cfg.interface.scenario_ids) == ['April2026-OW_desert3']
    assert cfg.interface.api_endpoint == 'http://127.0.0.1:18081'
    assert cfg.get('alignment_target') is None
    assert cfg.driver.target_context is not None
    assert not cfg.driver.extra_tools.kaleido.auto_assess


def test_april_affiliation_launcher_rejects_conflicting_actor_target():
    result = dry_run(
        ['+alignment_target=june2025/ADEPT-June2025-merit-0.0'],
        launcher='phase2_ow_agent_qwen38_kaleido_april_affiliation0.sbatch')
    assert result.returncode == 2
    assert 'no Hydra overrides' in result.stderr


@pytest.mark.parametrize('launcher', [
    'phase2_ow_agent_qwen38.sh', 'phase2_ow_agent_qwen38_kaleido.sh',
    'phase2_ow_agent_qwen38_kaleido.sbatch', 'phase2_ow_agent_qwen38_live.sbatch',
    'phase2_ow_agent_qwen38_smoke.sbatch', 'phase2_ow_agent_openai.sbatch',
])
def test_kaleido_launchers_instantiate_original_toolkit(launcher):
    from hydra.utils import instantiate
    from align_system.tests.test_agent_assessments import FakeKaleido

    result = dry_run(launcher=launcher, TOOLKIT='kaleido',
                     RUN_MODE='live' if launcher.endswith('_live.sbatch') else 'smoke')
    cfg = composed(result)
    toolkit = instantiate(cfg.driver.extra_tools.kaleido, kaleido_instance=FakeKaleido())
    assert not toolkit.auto_assess
    assert not toolkit.explain
    assert 'assessment_mode' not in cfg.driver.extra_tools.kaleido
    assert '-Kaleido-OnDemand-' in cfg.interface.username


def test_kaleido_original_options_can_be_overridden():
    from hydra.utils import instantiate
    from align_system.tests.test_agent_assessments import FakeKaleido

    cfg = composed(dry_run(['driver.extra_tools.kaleido.explain=true'],
                           TOOLKIT='kaleido', AUTO_ASSESS='true'))
    toolkit = instantiate(cfg.driver.extra_tools.kaleido, kaleido_instance=FakeKaleido())
    assert toolkit.auto_assess
    assert toolkit.explain
    assert '-Kaleido-Auto-' in cfg.interface.username


@pytest.mark.parametrize('overrides,settings', [
    ([], {'ASSESSMENT_MODE': 'evidence'}),
    (['driver.extra_tools.kaleido.assessment_mode=evidence'], {}),
])
def test_kaleido_does_not_accept_cr_assessment_modes(overrides, settings):
    result = dry_run(overrides, TOOLKIT='kaleido', **settings)
    assert result.returncode == 2
    assert 'ASSESSMENT_MODE applies only to compreg' in result.stderr


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
