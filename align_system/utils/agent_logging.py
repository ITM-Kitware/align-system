"""Opt-in, literal agent event presentation.

The annotation lives beside the original log message. In particular, diagnostic
consumers can continue to parse ``Action review:`` records while humans see a
short console summary and full visible exchanges in the opt-in transcript.
"""

from contextlib import contextmanager
from contextvars import ContextVar
import copy
import json
import logging
import re
from collections.abc import Mapping

from rich.logging import RichHandler
from rich.text import Text

_EVENT_CONTEXT = ContextVar('agent_log_context', default=None)
_CALL_COUNTER = ContextVar('agent_log_call_counter', default=None)
_PRIVATE_TEXT = ContextVar('agent_log_private_text', default=None)
_PRIVATE_KEYS = frozenset({
    'api_key', 'apikey', 'authorization', 'password', 'secret', 'client_secret',
    'access_token', 'refresh_token', 'credentials', 'headers', 'reasoning',
    'reasoning_content', 'thinking', 'additional_kwargs', 'response_metadata',
})
_ROLE_STYLES = {
    'DECISION': 'bold cyan',
    'REVIEWER': 'bold magenta',
    'TOOL': 'bold green',
    'WORKFLOW': 'bold yellow',
    'ENVIRONMENT': 'bold yellow',
    'WORKFLOW/ENVIRONMENT': 'bold yellow',
}
_FIELD_LABELS = {
    'scenario_id': 'scenario', 'target_id': 'target', 'scene_id': 'scene',
    'action_index': 'action', 'call_id': 'call', 'review_attempt': 'attempt',
    'tool_call_id': 'tool_call',
}


@contextmanager
def agent_log_scope(*, reset_calls=False, **fields):
    """Add correlation fields without leaking them into another scenario/task."""
    context_token = _EVENT_CONTEXT.set({**(_EVENT_CONTEXT.get() or {}), **fields})
    counter_token = None
    private_token = None
    if reset_calls or _CALL_COUNTER.get() is None:
        counter_token = _CALL_COUNTER.set([0])
    if reset_calls or _PRIVATE_TEXT.get() is None:
        private_token = _PRIVATE_TEXT.set({'texts': set(), 'pattern': None})
    try:
        yield
    finally:
        _EVENT_CONTEXT.reset(context_token)
        if counter_token is not None:
            _CALL_COUNTER.reset(counter_token)
        if private_token is not None:
            _PRIVATE_TEXT.reset(private_token)


def agent_log_private(text):
    """Register exception text for logging-only redaction in the current scenario.

    Callers retain the original feedback sent to models. Accept only strings so
    this helper never asks an opaque provider/exception object to serialize itself.
    """
    if not isinstance(text, str) or not text.strip():
        return
    registry = _PRIVATE_TEXT.get()
    if registry is None:
        registry = {'texts': set(), 'pattern': None}
        _PRIVATE_TEXT.set(registry)
    variants = {text}
    encoded = {text}
    # Feedback can recur inside a JSON context, then inside a JSON-encoded
    # review_response. Bound expansion to those two serialization layers while
    # covering both literal Unicode and ASCII-escaped provider representations.
    for _ in range(2):
        encoded = {json.dumps(item, ensure_ascii=ascii_only)[1:-1]
                   for item in encoded for ascii_only in (False, True)}
        variants.update(encoded)
    if not variants.issubset(registry['texts']):
        registry['texts'].update(variants)
        registry['pattern'] = re.compile('|'.join(
            re.escape(item) for item in sorted(registry['texts'], key=len, reverse=True)))


def _redact(text):
    registry = _PRIVATE_TEXT.get()
    pattern = registry['pattern'] if registry else None
    if pattern is None:
        return text
    # A single pass avoids accidentally redacting parts of our replacement when
    # multiple registered exception messages overlap.
    return pattern.sub('<exception details omitted>', text)


@contextmanager
def agent_log_call(**fields):
    """Allocate a model invocation ID shared by decision and reviewer scopes."""
    counter = _CALL_COUNTER.get()
    counter_token = None
    if counter is None:
        counter = [0]
        counter_token = _CALL_COUNTER.set(counter)
    counter[0] += 1
    call_id = counter[0]
    try:
        with agent_log_scope(**{**fields, 'call_id': call_id}):
            yield call_id
    finally:
        if counter_token is not None:
            _CALL_COUNTER.reset(counter_token)


def _literal(text):
    """Escape terminal controls, preserving ordinary Unicode and multiline text."""
    return ''.join(
        f'\\x{ord(char):02x}' if (ord(char) < 32 and char not in '\n\t')
        or 127 <= ord(char) <= 159 else char
        for char in text)


def preview(text, limit=240):
    """Return at most ``limit`` visible characters, explicitly marking omission."""
    text = _literal(_redact(visible_text(text))).replace('\n', '\\n').replace('\t', '\\t')
    if len(text) <= limit:
        return text
    return text[:limit] + f' … [truncated; {len(text)} characters total]'


def visible_text(response_or_content):
    """Read only text blocks; never stringify provider metadata or reasoning."""
    content = response_or_content
    if isinstance(content, Mapping):
        block_type = content.get('type')
        if block_type is not None and block_type not in {
                'text', 'input_text', 'output_text', 'message', 'chat',
                'system', 'developer', 'human', 'user', 'ai', 'assistant',
                'tool', 'function'}:
            return ''
        if 'content' in content:
            content = content['content']
        elif content.get('type') in {'text', 'input_text', 'output_text'}:
            return content.get('text', '') if isinstance(content.get('text'), str) else ''
        else:
            return ''
    elif not isinstance(content, (str, list, tuple)):
        content = getattr(content, 'content', None)
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        return ''.join(
            block if isinstance(block, str) else visible_text(block)
            for block in content if isinstance(block, str)
            or (isinstance(block, Mapping)
                and block.get('type') in {'text', 'input_text', 'output_text'}))
    return ''


def _safe_payload(value, *, schema_names=False):
    """Keep JSON-like data; never call repr/str on provider or exception objects."""
    if isinstance(value, str):
        return _redact(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {
            key: _safe_payload(item, schema_names=(
                key in {'properties', 'patternProperties', '$defs', 'definitions'}
                and isinstance(item, Mapping)
                and all(isinstance(schema, Mapping) for schema in item.values())))
            for key, item in value.items()
            if isinstance(key, str) and (schema_names or key.lower() not in _PRIVATE_KEYS)
        }
    if isinstance(value, (list, tuple)):
        return [_safe_payload(item) for item in value]
    return f'<{type(value).__name__} omitted>'


def _tool_calls(calls):
    result = []
    for call in calls or ():
        if not isinstance(call, Mapping):
            continue
        item = {key: _safe_payload(call[key])
                for key in ('id', 'name', 'args', 'type') if key in call}
        if isinstance(call.get('function'), Mapping):
            item['function'] = {key: _safe_payload(call['function'][key])
                                for key in ('name', 'arguments')
                                if key in call['function']}
        result.append(item)
    return result


def message_payload(messages):
    """Serialize ordered visible messages and tool calls without provider extras."""
    result = []
    for message in messages:
        if isinstance(message, Mapping):
            get = message.get
        else:
            get = lambda key, default=None: getattr(message, key, default)
        role = get('role') or get('type') or 'unknown'
        role = {'human': 'user', 'ai': 'assistant'}.get(role, role)
        item = {'role': role, 'content': visible_text(get('content'))}
        for key in ('name', 'tool_call_id'):
            value = get(key)
            if isinstance(value, str):
                item[key] = value
        for key in ('tool_calls', 'invalid_tool_calls'):
            value = _tool_calls(get(key))
            if value:
                item[key] = value
        result.append(item)
    return result


def _detail_capture_handlers(logger, level):
    """Find opted-in handlers along the same propagation path as normal logging."""
    handlers = []
    while logger is not None:
        handlers.extend(
            handler for handler in logger.handlers
            if getattr(handler, 'capture_agent_details', False)
            and not handler._closed and level >= handler.level)
        if not logger.propagate:
            break
        logger = logger.parent
    return handlers


def agent_log_event(logger, role, direction, summary, *, payload=None,
                    level=logging.INFO, legacy_message=None, legacy_args=(),
                    **fields):
    """Emit a compact event and opt-in detail with the same identifiers.

    DEBUG broadcasts details normally. At INFO, transcript handlers can capture
    details without sending them to the console or enabling transport debugging.
    With neither enabled, callable payloads stay unevaluated. Presentation never
    replaces the legacy message used by structured diagnostic readers.
    """
    metadata = {
        'role': role.upper(), 'direction': direction.upper(),
        'fields': _safe_payload({**(_EVENT_CONTEXT.get() or {}), **fields}),
        'summary': _literal(_redact(summary)), 'payload': None, 'detail': False,
    }
    if logger.isEnabledFor(level):
        logger.log(level, metadata['summary'] if legacy_message is None else legacy_message,
                   *(legacy_args if legacy_message is not None else ()),
                   extra={'agent_event': metadata}, stacklevel=2)
    if payload is None:
        return
    debug_enabled = logger.isEnabledFor(logging.DEBUG)
    capture_handlers = []
    if not debug_enabled:
        if not logger.isEnabledFor(logging.INFO) or not logger.isEnabledFor(level):
            return
        capture_handlers = _detail_capture_handlers(logger, level)
        if not capture_handlers:
            return
    try:
        detail = _safe_payload(payload() if callable(payload) else payload)
    except Exception as error:
        # Diagnostics must not turn a successful model/tool call into failure.
        detail = {'logging_detail': 'unavailable',
                  'error_type': type(error).__name__}
    extra = {'agent_event': {**metadata, 'payload': detail, 'detail': True}}
    if debug_enabled:
        logger.debug(metadata['summary'], extra=extra, stacklevel=2)
    else:
        # Keep a real record (including custom record factories and filters),
        # but send it only to handlers explicitly requesting transcript detail.
        pathname, lineno, func, _ = logger.findCaller(stacklevel=2)
        record = logger.makeRecord(logger.name, level, pathname, lineno,
                                   metadata['summary'], (), None, func=func,
                                   extra=extra)
        filtered = logger.filter(record)
        if filtered:
            # Python 3.12 filters can replace a record instead of just allowing it.
            if isinstance(filtered, logging.LogRecord):
                record = filtered
            for handler in capture_handlers:
                if record.levelno >= handler.level:
                    handler.handle(record)


def _event_heading(event):
    fields = ' | '.join(
        f'{_FIELD_LABELS.get(key, key)}={_literal(str(value))}'
        for key, value in event['fields'].items() if value is not None)
    detail = ' | DETAIL' if event['detail'] else ''
    return (f"=== {event['role']} | {event['direction']}{detail}"
            f"{' | ' + fields if fields else ''} ===")


def _event_body(event):
    if event['detail']:
        payload = event['payload']
        if isinstance(payload, Mapping):
            blocks = []
            for key, value in payload.items():
                if key == 'messages' and isinstance(value, list):
                    messages = []
                    for index, message in enumerate(value, start=1):
                        if not isinstance(message, Mapping):
                            messages.append(json.dumps(message, ensure_ascii=False))
                            continue
                        messages.append(f"--- message {index} | {message.get('role', 'unknown')} ---")
                        messages.append(_text_block(message.get('content', '')))
                        other = {name: item for name, item in message.items()
                                 if name not in {'role', 'content'}}
                        if other:
                            messages.append(json.dumps(other, indent=2, ensure_ascii=False))
                    rendered = '\n'.join(messages)
                elif isinstance(value, str):
                    rendered = _text_block(value)
                else:
                    rendered = json.dumps(value, indent=2, ensure_ascii=False)
                blocks.append(f'{key}:\n{rendered}')
            return _literal('\n\n'.join(blocks))
        return _literal(_text_block(payload) if isinstance(payload, str)
                        else json.dumps(payload, indent=2, ensure_ascii=False))
    return event['summary']


def _text_block(value):
    """Pretty-print serialized JSON while keeping other visible text literal."""
    try:
        structured = json.loads(value)
    except (TypeError, ValueError):
        if not isinstance(value, str):
            return value
        # Context messages also contain prose headings followed by complete
        # compact JSON lines. Expand those lines without changing the captured
        # payload or interpreting arbitrary fragments as JSON.
        lines = []
        for line in value.split('\n'):
            try:
                structured_line = json.loads(line)
            except ValueError:
                lines.append(line)
                continue
            if isinstance(structured_line, (dict, list)):
                line = json.dumps(structured_line, indent=2, ensure_ascii=False)
            lines.append(line)
        return '\n'.join(lines)
    if isinstance(structured, (dict, list)):
        return json.dumps(structured, indent=2, ensure_ascii=False)
    return value


class AgentPlainFormatter(logging.Formatter):
    """Render annotated events in plain text; leave ordinary records unchanged."""

    def format(self, record):
        event = getattr(record, 'agent_event', None)
        if event is None:
            return super().format(record)
        return _event_heading(event) + '\n' + _event_body(event)


class AgentRichHandler(RichHandler):
    """Use consistent agent colors without interpreting any payload as markup."""

    def emit(self, record):
        # Rich/stdlib formatters may annotate a record; do not change a record
        # shared with another handler or structured diagnostic consumer.
        super().emit(copy.copy(record) if hasattr(record, 'agent_event') else record)

    def render_message(self, record, message):
        event = getattr(record, 'agent_event', None)
        if event is None:
            return super().render_message(record, message)
        text = Text()
        text.append(_event_heading(event), style=_ROLE_STYLES.get(event['role'], 'bold'))
        text.append('\n')
        text.append(_event_body(event), style='none')
        return text
