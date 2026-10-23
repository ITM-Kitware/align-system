"""A readable companion to the complete agent event capture.

Request histories and schemas remain in the raw log with exact line references.
Tool arguments, results, model output, review feedback, and failures stay inline.
This module consumes the already sanitized presentation, never provider objects.
"""

import logging
import os
from pathlib import Path
import re
import textwrap

from align_system.utils.agent_logging import (
    AgentPlainFormatter, _event_body, _literal,
)


class AgentTranscriptWriter:
    """Render chronological interactions, grouping context without buffering calls."""

    def __init__(self, stream, source_name='raw_align_system.log'):
        self.stream = stream
        self.source_name = _literal(str(source_name))
        self._scenario = None
        self._action = None
        self._last_event = None
        self.stream.write(
            'AGENT INTERACTIONS\n'
            f'Complete source: {self.source_name}\n'
            'Request histories, schemas, context budgets, and action catalogs are\n'
            'referenced by source line below. Full outputs and tool results are inline.\n'
            'Transport/debug diagnostics remain in the source. No detail is truncated here.\n'
            'Payloads appear when captured in the source; older INFO logs may contain summaries only.\n')
        self.stream.flush()

    def _reference(self, source_line):
        return f'{self.source_name}:{source_line}'

    def _write_body(self, body):
        # This is a prose transcript, not a JSON interchange format. Wrap for an
        # editor while keeping long identifiers intact and preserving all words.
        for line in _literal(body).splitlines():
            indent = line[:len(line) - len(line.lstrip())]
            self.stream.write('\n'.join(textwrap.wrap(
                line, width=112, subsequent_indent=indent + '    ',
                break_long_words=False, break_on_hyphens=False,
                replace_whitespace=False)) + '\n')

    def _write_context(self, fields):
        if fields.get('scenario_id') is not None:
            scenario = (fields.get('scenario_id'), fields.get('target_id'))
            if scenario != self._scenario:
                self._scenario = scenario
                self._action = None
                self.stream.write('\n' + '=' * 80 + '\n')
                self._write_body(f'SCENARIO: {scenario[0]} | target={scenario[1]}')
                self.stream.write('=' * 80 + '\n')
        if fields.get('action_index') is not None:
            action = (fields.get('scene_id'), fields['action_index'])
            if action != self._action:
                self._action = action
                self._write_body(f'\n=== ACTION {action[1]} | scene={action[0]} ===')

    @staticmethod
    def _reference_only(event):
        return (
            (event['role'] in {'DECISION', 'REVIEWER'} and event['direction'] == 'INPUT')
            or event['direction'] == 'CONTEXT BUDGET'
            or (event['role'] == 'ENVIRONMENT' and event['direction'] in {
                'AVAILABLE ACTIONS', 'ELIGIBLE ACTIONS'}))

    def write_event(self, event, body, source_line):
        fields = event['fields']
        self._write_context(fields)
        identity = (event['role'], event['direction'], fields)
        adjacent_detail = event['detail'] and identity == self._last_event
        if not adjacent_detail:
            title = event['role']
            if title == 'TOOL':
                name = fields.get('tool_name')
                if not name:
                    match = re.search(r'\bname=([^\s|]+)', event.get('summary', ''))
                    name = match[1] if match else 'unknown'
                title = f'TOOL CALL: {name}'
            self._write_body(f'\n--- {title} | {event["direction"]} ---')
            correlation = ' | '.join(
                f'{key.removesuffix("_id")}={value}'
                for key, value in fields.items()
                if value is not None and key not in {
                    'scenario_id', 'target_id', 'scene_id', 'action_index', 'tool_name'})
            if correlation:
                self._write_body(correlation)
        reference = self._reference(source_line)
        if event['detail'] and self._reference_only(event):
            self._write_body(f'Full {event["direction"].lower()} payload: {reference}')
        else:
            self._write_body(f'{"Details" if event["detail"] else "Source"}: {reference}')
            self._write_body(body)
        # Only a summary can absorb the following detail. Repeated events still
        # receive distinct headings, even when they share their correlation IDs.
        self._last_event = None if event['detail'] else (
            event['role'], event['direction'], dict(fields))
        self.stream.flush()

    def write_diagnostic(self, body, source_line):
        self._last_event = None
        self._write_body(f'\n--- LOG | {self._reference(source_line)} ---')
        self._write_body(body)
        self.stream.flush()


class AgentTranscriptFileHandler(logging.FileHandler):
    """Write both files under one handler lock so source line references are exact.

    Opt into full agent event payloads at INFO as well as DEBUG. The raw format
    and legacy LogRecord are unchanged; additional INFO payloads go only to this
    handler, keeping the console concise and the raw source references complete.
    """

    capture_agent_details = True

    def __init__(self, filename, transcript_filename, mode='w'):
        raw_path = Path(filename)
        transcript_path = Path(transcript_filename)
        if (raw_path.resolve() == transcript_path.resolve()
                or (raw_path.exists() and transcript_path.exists()
                    and raw_path.samefile(transcript_path))):
            raise ValueError('Transcript must be separate from the raw log')
        self._raw_lines = 0
        if mode == 'a' and raw_path.exists():
            with raw_path.open('rb') as existing:
                self._raw_lines = sum(line.count(b'\n') for line in existing)
        super().__init__(filename, mode=mode, encoding='utf-8')
        self.setFormatter(AgentPlainFormatter())
        try:
            self._transcript_stream = transcript_path.open(mode, encoding='utf-8')
            self.transcript = AgentTranscriptWriter(
                self._transcript_stream,
                source_name=os.path.relpath(raw_path.resolve(), transcript_path.resolve().parent))
        except Exception:
            self.close()
            raise

    def emit(self, record):
        try:
            # FileHandler normally formats inside emit. Do it once here, so the
            # counted text and the bytes actually written cannot diverge.
            if self.stream is None:
                if self.mode == 'w' and self._closed:
                    return
                self.stream = self._open()
            rendered = self.format(record) + self.terminator
            source_line = self._raw_lines + 1
            self.stream.write(rendered)
            self.flush()
            self._raw_lines += rendered.count('\n')
            event = getattr(record, 'agent_event', None)
            if event is not None:
                self.transcript.write_event(event, _event_body(event), source_line)
            elif (record.levelno >= logging.WARNING
                  or (record.name.startswith('align_system.')
                      and record.levelno >= logging.INFO)):
                self.transcript.write_diagnostic(
                    f'{record.levelname} [{record.name}]\n{rendered.rstrip()}', source_line)
        except Exception:
            self.handleError(record)

    def close(self):
        self.acquire()
        try:
            if hasattr(self, '_transcript_stream'):
                self._transcript_stream.close()
            super().close()
        finally:
            self.release()
