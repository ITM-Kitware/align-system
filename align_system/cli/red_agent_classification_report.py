"""Combine saved CIA classifier predictions into confusion matrices and accuracy."""

import argparse
import csv
import json
from pathlib import Path


CIA_LABELS = ('B-line-C', 'B-line-I', 'B-line-A', 'Meander-C', 'Meander-I', 'Meander-A')


def build_report(predictions, labels=CIA_LABELS):
    predictions = list(predictions)
    units = {'window' if p.get('window_size') is not None else 'trajectory' for p in predictions}
    if len(units) > 1:
        raise ValueError('Cannot combine window predictions with whole-trajectory predictions')
    labels = list(labels)
    indices = {label: index for index, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for prediction in predictions:
        actual = prediction.get('true_label')
        predicted = prediction.get('predicted_label')
        if actual not in indices or predicted not in indices:
            raise ValueError(f'Unknown or missing class: actual={actual!r}, predicted={predicted!r}')
        matrix[indices[actual]][indices[predicted]] += 1

    total = sum(map(sum, matrix))
    if not total:
        raise ValueError('No predictions to report')
    correct = sum(matrix[index][index] for index in range(len(labels)))
    per_class = {}
    normalized = []
    for index, label in enumerate(labels):
        support = sum(matrix[index])
        per_class[label] = {
            'support': support,
            'n_trajectories': len({p.get('trajectory_id') or p.get('scenario_id')
                                   for p in predictions if p.get('true_label') == label}),
            'correct': matrix[index][index],
            'accuracy': matrix[index][index] / support if support else None,
        }
        normalized.append([count / support if support else None for count in matrix[index]])
    accuracies = [row['accuracy'] for row in per_class.values() if row['accuracy'] is not None]
    return {
        'labels': labels,
        'matrix_orientation': 'rows=true_label, columns=predicted_label',
        'evaluation_unit': units.pop(),
        'n_trajectories': sum(row['n_trajectories'] for row in per_class.values()),
        'n_samples': total,
        'n_correct': correct,
        'overall_accuracy': correct / total,
        'macro_accuracy': sum(accuracies) / len(accuracies),
        'per_class': per_class,
        'confusion_matrix': matrix,
        'row_normalized_confusion_matrix': normalized,
    }


def load_predictions(paths, labels=CIA_LABELS):
    predictions = []
    seen = set()
    for path in paths:
        path = Path(path)
        with path.open() as stream:
            result = json.load(stream)
        if result.get('complete') is not True:
            raise ValueError(f'Classifier run is incomplete: {path}')
        if set(result.get('labels', [])) != set(labels):
            raise ValueError(f'Classifier labels do not match the CIA classes: {path}')
        if not result.get('predictions'):
            raise ValueError(f'Classifier run contains no predictions: {path}')
        for record in result['predictions']:
            if not record.get('scenario_id'):
                raise ValueError(f'Prediction has no scenario_id: {path}')
            key = (record.get('true_label'),
                   record.get('trajectory_id') or record['scenario_id'],
                   record.get('window_start'), record.get('window_end'))
            if key in seen:
                raise ValueError(f'Duplicate prediction: {key}')
            seen.add(key)
            predictions.append({**record, 'source_file': str(path.resolve())})
    return predictions


def format_report(report):
    labels = report['labels']
    width = max(12, *(len(label) + 1 for label in labels))
    lines = [f"Evaluation unit: {report['evaluation_unit']}; "
             f"{report['n_samples']} predictions from {report['n_trajectories']} trajectories", '']
    for title, key, percentages in (
        ('Confusion matrix (counts; rows = actual, columns = predicted)', 'confusion_matrix', False),
        ('Confusion matrix (percent of each actual class)', 'row_normalized_confusion_matrix', True),
    ):
        lines.extend([title, 'Actual'.ljust(width) + ''.join(label.rjust(width) for label in labels)])
        for label, row in zip(labels, report[key]):
            values = [
                'N/A' if value is None else f'{value:.1%}' if percentages else str(value)
                for value in row
            ]
            lines.append(label.ljust(width) + ''.join(value.rjust(width) for value in values))
        lines.append('')
    lines.append('Per-class accuracy (correct / actual class total):')
    for label, row in report['per_class'].items():
        accuracy = 'N/A' if row['accuracy'] is None else f"{row['accuracy']:.2%}"
        lines.append(f"  {label}: {accuracy} ({row['correct']}/{row['support']})")
    lines.extend([
        f"Overall accuracy: {report['overall_accuracy']:.2%} ({report['n_correct']}/{report['n_samples']})",
        f"Macro-average class accuracy: {report['macro_accuracy']:.2%}",
    ])
    return '\n'.join(lines)


def write_report(report, predictions, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / 'classification_report.json').open('w') as stream:
        json.dump(report, stream, indent=2)
    (output_dir / 'classification_report.txt').write_text(format_report(report) + '\n')
    for filename, key, percentage in (
        ('confusion_matrix.csv', 'confusion_matrix', False),
        ('confusion_matrix_percent.csv', 'row_normalized_confusion_matrix', True),
    ):
        with (output_dir / filename).open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['actual / predicted', *report['labels']])
            for label, row in zip(report['labels'], report[key]):
                values = [
                    '' if value is None else round(value * 100, 4) if percentage else value
                    for value in row
                ]
                writer.writerow([label, *values])
    with (output_dir / 'predictions.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            'source_file', 'scenario_id', 'trajectory_id', 'seed', 'red_agent_name',
            'window_index', 'window_start', 'window_end', 'window_size', 'classification_time_s',
            'true_label', 'predicted_label', 'justification',
        ], extrasaction='ignore')
        writer.writeheader()
        writer.writerows(predictions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', nargs='+', type=Path, required=True,
                        help='Completed classification_results.json files')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    try:
        predictions = load_predictions(args.results)
        report = build_report(predictions)
        missing = [label for label, row in report['per_class'].items() if not row['support']]
        if missing:
            raise ValueError(f'No ground-truth examples for classes: {", ".join(missing)}')
        write_report(report, predictions, args.output_dir)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(format_report(report))
    print(f'\nReports saved to {args.output_dir.resolve()}')


if __name__ == '__main__':
    main()
