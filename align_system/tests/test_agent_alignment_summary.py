import hashlib
import json

import pytest

from align_system.cli.summarize_agent_alignment import main, summarize


def action(aid='a', **kwargs):
    return dict(action_id=aid, action_type='TAG_CHARACTER', character_id='p1',
                parameters={'category': 'IMMEDIATE'}, **kwargs)


def record(aid='a', target='target', source='agent'):
    return {'input': {'scenario_id': 'scenario', 'alignment_target_id': target},
            'output': {'action': action(aid)},
            'choice_info': {'langchain_agent': {'decision_source': source, 'decision_time_s': 4}}}


def test_operational_rates_do_not_claim_alignment():
    records = [record(), record(source='fallback'), record(source='forced_end_scene')]
    report = summarize(records)
    overall = report['overall']
    assert overall['executed_actions'] == 3
    assert overall['fallback_rate'] == pytest.approx(1 / 3)
    assert overall['forced_end_rate'] == pytest.approx(1 / 3)
    assert overall['mean_decision_time_s'] == 4
    assert overall['preference_agreement'] is None
    assert overall['mean_annotated_regret'] is None
    assert report['worst_labeled_target_agreement'] is None


def test_independent_labels_compute_agreement_regret_and_target_breakdown():
    labels = {'source': 'Held-out expert annotations', 'decisions': [
        {'record_index': 0, 'preferred_actions': [action('a'), action('b')],
         'action_utilities': [{'action': action('a'), 'utility': 2},
                              {'action': action('b'), 'utility': 5}]},
        {'record_index': 1, 'preferred_actions': [action('b')]},
    ]}
    report = summarize([record(target='high'), record(target='low'), record(target='low')], labels)
    assert report['overall']['preference_labeled_decisions'] == 2
    assert report['overall']['preference_agreement'] == 0.5
    assert report['overall']['utility_labeled_decisions'] == 1
    assert report['overall']['mean_annotated_regret'] == 3
    assert report['by_target']['high']['preference_agreement'] == 1
    assert report['by_target']['low']['preference_agreement'] == 0
    assert report['worst_labeled_target_agreement'] == 0


def test_labels_match_completed_parameters_and_ignore_justification():
    wrong = action()
    wrong['parameters'] = {'category': 'DELAYED'}
    labels = {'source': 'Experts', 'decisions': [
        {'record_index': 0, 'preferred_actions': [wrong]}]}
    assert summarize([record()], labels)['overall']['preference_agreement'] == 0
    labels['decisions'][0]['preferred_actions'] = [action(justification='Different prose')]
    assert summarize([record()], labels)['overall']['preference_agreement'] == 1


@pytest.mark.parametrize('decisions', [
    [{'record_index': 2, 'preferred_actions': [action()]}],
    [{'record_index': 0, 'preferred_actions': [action()]}] * 2,
    [{'record_index': 0, 'preferred_actions': []}],
    [{'record_index': 0, 'preferred_actions': [{'action_id': 'a'}]}],
    [{'record_index': 0, 'action_utilities': [{'action': action('b'), 'utility': 1}]}],
    [{'record_index': 0, 'action_utilities': [{'action': action(), 'utility': float('nan')}]}],
])
def test_invalid_or_incomplete_labels_are_rejected(decisions):
    with pytest.raises(ValueError):
        summarize([record()], {'source': 'Experts', 'decisions': decisions})


def test_empty_run_and_legacy_timing_remain_missing_not_perfect_scores():
    assert summarize([])['overall']['preference_agreement'] is None
    old = record()
    del old['choice_info']['langchain_agent']['decision_time_s']
    report = summarize([old])['overall']
    assert report['mean_decision_time_s'] is None
    assert report['timed_decisions'] == 0
    del old['choice_info']['langchain_agent']['decision_source']
    report = summarize([old])['overall']
    assert report['source_labeled_decisions'] == 0
    assert report['fallback_rate'] is None and report['forced_end_rate'] is None


def test_unnamed_distribution_targets_are_not_collapsed():
    first, second = record(target=None), record(target=None)
    first['choice_info']['alignment_target_entries'] = [
        {'kdma': 'affiliation', 'parameters': [{'value': 0.1}]}]
    second['choice_info']['alignment_target_entries'] = [
        {'kdma': 'affiliation', 'parameters': [{'value': 0.9}]}]
    report = summarize([first, second])
    assert len(report['by_target']) == 2
    assert report['overall']['scenario_target_pairs'] == 2


@pytest.mark.parametrize('labels', [[], 'bad labels', {'source': 'Experts', 'decisions': [None]}])
def test_bad_label_shapes_are_rejected(labels):
    with pytest.raises(ValueError):
        summarize([record()], labels)


@pytest.mark.parametrize('value', [-1, float('nan'), float('inf'), '4'])
def test_invalid_timings_cannot_make_misleading_reports(value):
    row = record()
    row['choice_info']['langchain_agent']['decision_time_s'] = value
    with pytest.raises(ValueError, match='finite nonnegative'):
        summarize([row])


def test_cli_binds_labels_to_exact_run_file(tmp_path, capsys):
    path = tmp_path / 'input_output.json'
    path.write_text(json.dumps([record()]))
    labels = {'input_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'source': 'Experts', 'decisions': [
                  {'record_index': 0, 'preferred_actions': [action()]}]}
    label_path = tmp_path / 'labels.json'
    label_path.write_text(json.dumps(labels))
    main([str(path), '--labels', str(label_path)])
    assert json.loads(capsys.readouterr().out)['overall']['preference_agreement'] == 1
    path.write_text(json.dumps([record('b')]))
    with pytest.raises(SystemExit) as exc:
        main([str(path), '--labels', str(label_path)])
    assert exc.value.code == 2
    assert 'does not match' in capsys.readouterr().err
