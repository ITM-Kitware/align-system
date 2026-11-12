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


@pytest.mark.parametrize('settings,mode', [
    ({'TOOLKIT': 'baseline'}, None),
    ({'TOOLKIT': 'compreg', 'ASSESSMENT_MODE': 'ranked', 'AUTO_ASSESS': 'true'}, 'ranked'),
    ({'TOOLKIT': 'compreg', 'ASSESSMENT_MODE': 'evidence', 'AUTO_ASSESS': 'true'}, 'evidence'),
])
def test_study_variants_select_matching_configs_and_names(settings, mode):
    cfg = composed(dry_run(**settings))
    if mode:
        assert cfg.driver.extra_tools.comparative_regression.assessment_mode == mode
        assert cfg.driver.extra_tools.comparative_regression.auto_assess
        assert cfg.driver.extra_tools.comparative_regression.generation_kwargs.max_tokens == 8192
        assert not cfg.driver.extra_tools.comparative_regression.generation_kwargs.extra_body.chat_template_kwargs.enable_thinking
        assert cfg.driver.chat_model.max_tokens == 8192
        assert 'extra_body' not in cfg.driver.chat_model
    else:
        assert not cfg.driver.extra_tools
    assert cfg.interface.username.startswith('testrun-')
    assert cfg.driver.max_actions_per_scenario == 10


@pytest.mark.parametrize('run_mode,require,required', [
    ('live', None, True), ('live', 'false', False), ('smoke', None, False),
])
def test_compreg_wrapper_reports_scores_after_the_run(run_mode, require, required):
    settings = {'RUN_MODE': run_mode}
    if require is not None:
        settings['REQUIRE_SCORES'] = require
    result = dry_run(launcher='phase2_ow_agent_qwen38_compreg.sbatch', **settings)
    cfg = composed(result)
    assert cfg.interface.session_type == ('eval' if run_mode == 'live' else 'test')
    report = next(line for line in result.stdout.splitlines()
                  if line.startswith('python -m align_system.cli.report_agent_scores '))
    assert shlex.split(report)[3] == cfg.hydra.run.dir
    assert ('--require' in shlex.split(report)) == required


@pytest.mark.parametrize('task,profile', [(None, 'APRIL_OPENWORLD3'), ('0', 'APRIL_OPENWORLD3'),
                                          ('1', 'FEB_OPENWORLD3')])
def test_compreg_wrapper_array_tasks_run_distinct_profiles(task, profile):
    settings = {'RUN_MODE': 'live'}
    if task is not None:
        settings['SLURM_ARRAY_TASK_ID'] = task
    cfg = composed(dry_run(launcher='phase2_ow_agent_qwen38_compreg.sbatch', **settings))
    assert cfg.interface.adm_profile == profile
    assert profile in cfg.interface.username and profile in cfg.hydra.run.dir


def test_hosted_wrapper_uses_remote_model_without_exposing_key():
    result = dry_run(launcher='phase2_ow_agent_openai.sbatch',
                     OPENAI_API_KEY='test-only-secret-not-for-output')
    cfg = composed(result)
    assert cfg.driver.chat_model._target_ == 'langchain_openai.ChatOpenAI'
    assert cfg.driver.extra_tools.comparative_regression.assessment_mode == 'evidence'
    assert not cfg.driver.extra_tools.comparative_regression.auto_assess
    assert not cfg.driver.extra_tools.comparative_regression.generation_kwargs
    assert 'vllm serve' not in result.stdout
    assert 'test-only-secret-not-for-output' not in result.stdout + result.stderr


def test_cr_scorer_budget_override_does_not_change_actor():
    cfg = composed(dry_run([
        '++driver.extra_tools.comparative_regression.generation_kwargs.max_tokens=16384'],
        TOOLKIT='compreg'))
    assert cfg.driver.extra_tools.comparative_regression.generation_kwargs.max_tokens == 16384
    assert cfg.driver.chat_model.max_tokens == 8192


def test_cr_launchers_keep_scoring_optional_and_preserve_all_twenty_examples():
    cfg = composed(dry_run(launcher='phase2_ow_agent_qwen38_compreg.sh'))
    cr = cfg.driver.extra_tools.comparative_regression
    assert not cr.auto_assess
    assert cr.max_retries == 2
    assert cr.show_reasoning
    assert cr.icl.icl_generator_partial.incontext_settings.number == 20
    assert cr.icl.icl_generator_partial.scorer.batch_size == 32
    assert '-CompRegression-Ranked-OnDemand-' in cfg.interface.username


def test_cr_direct_config_is_optional_with_bounded_retries():
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / 'align_system/configs')):
        cfg = compose(config_name='action_based', overrides=[
            '+experiment=phase2_feb_openworld/phase2_langchain_agent_ow_comparative_regression_live'])
    assert not cfg.driver.extra_tools.comparative_regression.auto_assess
    assert cfg.driver.extra_tools.comparative_regression.max_retries == 2


def test_cr_scorer_thinking_override_is_only_in_scoring_payload():
    from hydra.utils import instantiate
    from langchain_core.messages import HumanMessage
    from omegaconf import OmegaConf

    cfg = composed(dry_run(TOOLKIT='compreg'))
    model = instantiate(cfg.driver.chat_model)
    options = OmegaConf.to_container(
        cfg.driver.extra_tools.comparative_regression.generation_kwargs, resolve=True)
    messages = [HumanMessage('Compare the current actions.')]
    scorer_payload = model._get_request_payload(messages, **options)
    assert scorer_payload['extra_body']['chat_template_kwargs']['enable_thinking'] is False
    assert scorer_payload['max_completion_tokens'] == 8192
    actor_payload = model._get_request_payload(messages)
    assert not (actor_payload.get('extra_body') or {}).get('chat_template_kwargs')
    assert actor_payload['max_completion_tokens'] == 8192


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
