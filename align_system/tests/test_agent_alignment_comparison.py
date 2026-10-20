import hashlib
import json

import pytest

from align_system.cli.compare_agent_alignment import compare_study, main
from align_system.tests.test_agent_alignment_summary import action, record


def episode(scenario='one', target='target', steps=1, time=4, aid='a'):
    rows = []
    for step in range(steps):
        row = record(aid=aid, target=target)
        row['input'].update(scenario_id=scenario, full_state={'situation': 'Stable facts', 'step': step},
                            choices=[action('a'), action('b')])
        row['choice_info']['alignment_target_entries'] = [{'kdma': 'affiliation', 'value': 0.8}]
        row['choice_info']['langchain_agent'].update(n_actions_taken_in_scenario=step,
                                                    decision_time_s=time)
        rows.append(row)
    return rows


def save_run(tmp_path, variant, rows, preferred=None, replicate='seed-0', suffix=''):
    path = tmp_path / f'{variant}-{replicate}{suffix}.json'
    path.write_text(json.dumps(rows))
    run = {'variant': variant, 'replicate': replicate, 'input_output': path.name}
    if preferred is not None:
        labels = {'source': 'Independent expert fixture',
                  'input_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                  'decisions': [{'record_index': i, 'preferred_actions': [action(preferred)]}
                                for i in range(len(rows))]}
        label_path = tmp_path / f'{path.stem}-labels.json'
        label_path.write_text(json.dumps(labels))
        run['labels'] = label_path.name
    return run


def test_pairing_uses_scenario_weights_not_number_of_actions(tmp_path):
    runs = [
        save_run(tmp_path, 'evidence', episode('one', steps=10, time=4) + episode('two', time=4), 'a'),
        save_run(tmp_path, 'ranked', episode('one', steps=3, time=8, aid='b') + episode('two', time=4), 'a')]
    report = compare_study({'reference': 'evidence', 'runs': runs}, tmp_path, bootstrap_samples=100)
    comparison = report['paired_comparisons']['ranked']
    assert comparison['matched_initial_episodes'] == 2
    time = comparison['metrics']['mean_decision_time_s']
    assert time['mean_delta'] == 2  # mean of episode deltas 4 and 0
    assert time['ci95'] == [0, 4]
    assert time['scenario_count'] == 2
    agreement = comparison['metrics']['preference_agreement']
    assert agreement['mean_delta'] == -0.5
    assert agreement['ci95'] == [-1, 0]
    assert report['variants']['evidence']['overall']['preference_labeled_decisions'] == 11
    assert report['variants']['ranked']['overall']['preference_labeled_decisions'] == 4


def test_targets_and_replicates_share_one_scenario_cluster(tmp_path):
    runs = []
    for seed in ('seed-0', 'seed-1'):
        for variant, time in [('evidence', 4), ('ranked', 5)]:
            runs.append(save_run(tmp_path, variant,
                                 episode(target='high', time=time) + episode(target='low', time=time),
                                 replicate=seed))
    report = compare_study({'reference': 'evidence', 'runs': runs}, tmp_path)
    metric = report['paired_comparisons']['ranked']['metrics']['mean_decision_time_s']
    assert metric['matched_episode_pairs'] == 4
    assert metric['scenario_count'] == 1
    assert metric['mean_delta'] == 1
    assert metric['ci95'] is None


def test_missing_labels_stay_null_and_context_mismatches_are_excluded(tmp_path):
    original = episode('same') + episode('different') + episode('unknown') + episode('only-reference')
    changed = episode('same') + episode('different') + episode('unknown') + episode('only-variant')
    changed[1]['input']['full_state']['situation'] = 'Different initial casualties'
    del changed[2]['input']['full_state']
    runs = [save_run(tmp_path, 'evidence', original), save_run(tmp_path, 'ranked', changed)]
    comparison = compare_study({'reference': 'evidence', 'runs': runs}, tmp_path)['paired_comparisons']['ranked']
    assert comparison['matched_initial_episodes'] == 1
    assert comparison['initial_context_mismatches'] == 1
    assert comparison['unverified_initial_episodes'] == 1
    assert comparison['unmatched_reference_episodes'] == 1
    assert comparison['unmatched_variant_episodes'] == 1
    assert comparison['metrics']['preference_agreement']['mean_delta'] is None
    assert comparison['metrics']['preference_agreement']['pairs_missing_metric'] == 1


@pytest.mark.parametrize('change', ['target', 'choices', 'partial'])
def test_comparison_rejects_changed_targets_options_and_partial_logs(tmp_path, change):
    original, changed = episode(), episode()
    if change == 'target':
        changed[0]['choice_info']['alignment_target_entries'][0]['value'] = 0.2
    elif change == 'choices':
        changed[0]['input']['choices'].pop()
    else:
        changed[0]['choice_info']['langchain_agent']['n_actions_taken_in_scenario'] = 1
    runs = [save_run(tmp_path, 'evidence', original), save_run(tmp_path, 'ranked', changed)]
    result = compare_study({'reference': 'evidence', 'runs': runs}, tmp_path)['paired_comparisons']['ranked']
    assert result['matched_initial_episodes'] == 0


def test_option_order_alone_does_not_prevent_pairing(tmp_path):
    changed = episode()
    changed[0]['input']['choices'].reverse()
    runs = [save_run(tmp_path, 'evidence', episode()), save_run(tmp_path, 'ranked', changed)]
    result = compare_study({'reference': 'evidence', 'runs': runs}, tmp_path)
    assert result['paired_comparisons']['ranked']['matched_initial_episodes'] == 1


def test_duplicate_episodes_cannot_inflate_sample_size(tmp_path):
    runs = [save_run(tmp_path, 'evidence', episode()),
            save_run(tmp_path, 'evidence', episode(), suffix='-duplicate'),
            save_run(tmp_path, 'ranked', episode())]
    with pytest.raises(ValueError, match='Duplicate scenario'):
        compare_study({'reference': 'evidence', 'runs': runs}, tmp_path)
    repeated = save_run(tmp_path, 'evidence', episode() + episode())
    with pytest.raises(ValueError, match='Repeated episodes'):
        compare_study({'reference': 'evidence', 'runs': [repeated, runs[2]]}, tmp_path)


def test_reused_run_paths_and_changed_labels_are_rejected(tmp_path):
    evidence = save_run(tmp_path, 'evidence', episode(), 'a')
    with pytest.raises(ValueError, match='listed more than once'):
        compare_study({'reference': 'evidence', 'runs': [evidence, {**evidence, 'variant': 'ranked'}]}, tmp_path)
    (tmp_path / evidence['input_output']).write_text(json.dumps(episode(aid='b')))
    with pytest.raises(ValueError, match='does not match'):
        compare_study({'reference': 'evidence', 'runs': [evidence]}, tmp_path)


def test_cli_resolves_manifest_relative_paths_and_reproducible_intervals(tmp_path, capsys):
    runs = [save_run(tmp_path, 'evidence', episode('one') + episode('two'), 'a'),
            save_run(tmp_path, 'ranked', episode('one', aid='b') + episode('two'), 'a')]
    manifest = {'reference': 'evidence', 'runs': runs}
    path = tmp_path / 'study.json'
    path.write_text(json.dumps(manifest))
    main([str(path), '--bootstrap-samples', '100', '--seed', '17'])
    result = json.loads(capsys.readouterr().out)
    assert result == compare_study(manifest, tmp_path, bootstrap_samples=100, seed=17)


def test_three_variant_offline_driver_outputs_feed_comparison(tmp_path):
    from langchain_core.messages import AIMessage
    from align_system.drivers.itm_open_world_langchain import ITMOpenWorldLangChainDriver
    from align_system.tests.test_agent_assessments import action_call, session_for, toolkit_for

    class Agent:
        def __init__(self):
            self.calls = 0

        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            self.calls += 1
            return AIMessage(content='', tool_calls=[action_call('a')])

    runs = []
    for variant in ('baseline', 'ranked', 'evidence'):
        session, records = session_for()
        extras = {} if variant == 'baseline' else {
            'comparative_regression': toolkit_for(
                'comparative_regression', 'ranked' if variant == 'ranked' else 'evidence')}
        agent = Agent()
        driver = ITMOpenWorldLangChainDriver(
            chat_model=agent, extra_tools=extras,
            apply_action_filtering=False, max_actions_per_scenario=1)
        driver._run_agent_loop(session)
        assert agent.calls == len(records) == 1
        runs.append(save_run(tmp_path, variant, records))
    report = compare_study({'reference': 'evidence', 'runs': runs}, tmp_path)
    assert len(report['variants']) == 3
    for comparison in report['paired_comparisons'].values():
        assert comparison['matched_initial_episodes'] == 1
        assert comparison['metrics']['preference_agreement']['mean_delta'] is None


@pytest.mark.parametrize('manifest', [[], {}, {'runs': []}, {'runs': [{}]}])
def test_malformed_manifests_fail_with_clear_errors(tmp_path, manifest):
    with pytest.raises(ValueError):
        compare_study(manifest, tmp_path)
