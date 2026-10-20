"""Report each scenario's server alignment score, or why it is missing.

Reads the ``scoring_status.json`` that the LangChain agent driver writes
(see docs/agent_alignment.md). Scores come from the TA3 server at scenario
end; this command never computes one itself.
"""
import argparse
import json
from pathlib import Path
import re
import sys


FINAL_STATE_HINTS = {
    'Error': 'the TA3 server failed to get the score from TA1; check the TA3 server log',
    'Not calculated': 'the TA3 server has no TA1 integration (started in testing mode?)',
    'Not requested': 'alignment calculation is disabled on the TA3 server (CALC_ALIGNMENT)',
}


def score_of(record):
    """Return (score, source) from the API response or the final state text."""
    api = record.get('api') or {}
    if api.get('status') == 'available':
        return (api.get('result') or {}).get('score'), 'api'
    final_state = record.get('final_state') or {}
    if final_state.get('status') == 'available':
        return final_state.get('score'), 'final_state'
    return None, None


def explain(record):
    status = record.get('status')
    if status == 'pending':
        return 'scenario never finished (process killed?)'
    if status in ('incomplete', 'aborted'):
        return record.get('reason') or record.get('error') or status
    final_state = record.get('final_state') or {}
    if re.match(r'\s*Test scenario\b', final_state.get('text') or ''):
        return 'test session: the TA3 server never scores test sessions; use a live eval run'
    raw = final_state.get('raw_value')
    if raw in FINAL_STATE_HINTS:
        return f'final state says "{raw}": {FINAL_STATE_HINTS[raw]}'
    api = record.get('api') or {}
    api_note = api.get('reason') or api.get('error') or api.get('status') or 'not attempted'
    return f'no score in the final state; score API {api_note}'


def load_statuses(path):
    path = Path(path)
    if path.is_dir():
        path = path / 'scoring_status.json'
    with open(path) as f:
        statuses = json.load(f)
    if not isinstance(statuses, list):
        raise ValueError(f'Expected a list of scenario records: {path}')
    return statuses


def report(statuses, out=None):
    out = sys.stdout if out is None else out
    scored = 0
    for record in statuses:
        label = '{} target={}'.format(record.get('scenario_id'), record.get('alignment_target_id'))
        score, source = score_of(record)
        if score is not None:
            scored += 1
            print(f'{label}: score={score} ({source})', file=out)
        else:
            print(f'{label}: NO SCORE ({record.get("status")}): {explain(record)}', file=out)
    print(f'{scored} of {len(statuses)} scenarios scored', file=out)
    return scored


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path,
                        help='Run output directory or its scoring_status.json')
    parser.add_argument('--require', action='store_true',
                        help='Exit non-zero unless every started scenario has a score')
    args = parser.parse_args(argv)
    try:
        statuses = load_statuses(args.run)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    scored = report(statuses)
    if args.require and (not statuses or scored < len(statuses)):
        print('Error: alignment scores are missing; see docs/agent_alignment.md', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
