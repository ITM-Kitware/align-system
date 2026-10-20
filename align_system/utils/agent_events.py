"""Durable run journal, including episodes that never execute an action."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from uuid import uuid4


class AgentEventLog:
    def __init__(self, path):
        self.path = Path(path)
        self.run_id = str(uuid4())
        self.episode_id = None
        self.sequence = 0

    def __enter__(self):
        # Refuse to mix two runs, including accidental reuse of an output dir.
        self.file = self.path.open('x', encoding='utf-8')
        self.emit('run_started')
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.emit('run_aborted' if exc_type else 'run_finished',
                      **({'error_type': exc_type.__name__} if exc_type else {}))
        finally:
            self.file.close()

    def emit(self, event, **fields):
        entry = dict(fields, schema_version=1, run_id=self.run_id,
                     episode_id=self.episode_id, sequence=self.sequence,
                     timestamp=datetime.now(timezone.utc).isoformat(), event=event)
        self.file.write(json.dumps(entry, ensure_ascii=False, default=str, allow_nan=False) + '\n')
        self.file.flush()
        os.fsync(self.file.fileno())
        self.sequence += 1

    @contextmanager
    def episode(self, scenario_id):
        self.episode_id = str(uuid4())
        self.emit('episode_started', scenario_id=scenario_id)
        try:
            yield self.emit
        except BaseException as exc:
            self.emit('episode_aborted', error_type=type(exc).__name__)
            raise
        finally:
            self.episode_id = None


def write_json_atomic(path, value):
    """Preserve the preceding complete trace if a process dies mid-write."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as output:
        json.dump(value, output, indent=2)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)
