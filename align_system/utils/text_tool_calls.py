import json
import re


def parse_text_tool_calls(content):
    """Recover tool calls that a model emitted as plain JSON text
    (e.g. '{"name": "take_action", "parameters": {...}}') instead of
    as structured tool calls; some smaller models fall back to this
    style mid-conversation.

    `content` is a LangChain message content (a string or a list of
    string/dict parts).  Returns a list of tool-call dicts in
    LangChain's {'name', 'args', 'id'} shape; best-effort, so
    unparseable candidates are simply skipped.
    """
    if isinstance(content, list):
        content = "\n".join(
            part if isinstance(part, str) else part.get('text', '')
            for part in content)
    if not content:
        return []

    text = re.sub(r'```(?:json)?', '', content)

    # Extract top-level {...} blocks with a simple depth counter (note
    # braces inside JSON strings aren't accounted for; a candidate
    # split that way just fails to parse and is skipped)
    candidates = []
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}' and depth > 0:
            depth -= 1
            if depth == 0:
                candidates.append(text[start:i + 1])
                start = None

    tool_calls = []
    for idx, candidate in enumerate(candidates):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue

        if not isinstance(obj, dict) or 'name' not in obj:
            continue

        args = obj.get('parameters',
                       obj.get('arguments', obj.get('args', {})))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                continue
        if not isinstance(args, dict):
            continue

        tool_calls.append({'name': obj['name'],
                           'args': args,
                           'id': f'text-tool-call-{idx}'})

    return tool_calls
