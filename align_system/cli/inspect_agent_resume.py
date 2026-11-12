"""Validate a recovery source and describe its replay plan without contacting TA3."""
import argparse
import json

from align_system.drivers.agent_resume_trace import ReplayPlan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', help='Previous run directory containing the journal and action trace')
    args = parser.parse_args(argv)
    try:
        plan = ReplayPlan(args.run)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps(plan.summary(), indent=2))


if __name__ == '__main__':
    main()
