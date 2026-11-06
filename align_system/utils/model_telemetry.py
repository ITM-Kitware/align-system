"""Opt-in model-call measurements, separate from evaluation action records."""

from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
import logging
from pathlib import Path
import platform
import subprocess
from threading import Lock
from time import perf_counter
from uuid import uuid4

from align_system.utils.agent_logging import agent_log_context


log = logging.getLogger(__name__)
_RECORDER = ContextVar('model_telemetry_recorder', default=None)
_SETTINGS = (
    'model', 'temperature', 'top_p', 'seed', 'max_tokens',
    'max_completion_tokens', 'max_output_tokens', 'reasoning_effort',
    'thinking_token_budget', 'presence_penalty', 'frequency_penalty',
    'parallel_tool_calls', 'stream',
)
_PROMPTS = (
    'system_prompt', 'alignment_guidance', 'alignment_review_guidance',
    'patient_history_guidance', 'scene_end_guidance', 'repair_guidance',
    'location_guidance', 'qualitative_output_contract', 'weighted_output_contract',
    'review_guidance', 'citation_guidance', 'opening_prompt',
)
_HTTP_FIELDS = frozenset({
    'api_key', 'authorization', 'headers', 'extra_headers', 'extra_query',
    'timeout', 'max_retries', 'base_url', 'http_client', 'http_async_client',
})


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      allow_nan=False, separators=(',', ':')).encode('utf-8')


def _digest(value):
    return sha256(_json_bytes(value)).hexdigest()


def _static_prompts(adm):
    result = {}
    for role, component in (('decision', adm), ('reviewer', getattr(adm, 'reviewer', None))):
        result[role] = {}
        for name in _PROMPTS:
            value = getattr(component, name, None)
            if isinstance(value, (str, dict)) or value is None:
                result[role][name] = value
    return result


def _configuration_fingerprint(config, adm, *, include_experiment_factors):
    ignored_root = {
        'hydra', 'loglevel', 'save_log', 'save_raw_log', 'save_input_output',
        'save_scoring_output', 'save_alignment_targets', 'save_timing',
        'save_last_unstructured_state_per_scenario', 'save_model_telemetry',
        'save_model_requests', 'model_server_manifest', 'agent_log_rendering',
        'save_agent_transcript', 'raw_log_mode',
    }
    secret_fields = {'api_key', 'headers', 'extra_headers', 'authorization', 'password'}
    factors = {'reasoning_effort', 'thinking_token_budget', 'enable_thinking'}

    def canonical(value, path=()):
        if isinstance(value, Mapping):
            result = {}
            for key, item in value.items():
                if (key in secret_fields or (not path and key in ignored_root)
                        or (path == ('interface',) and key == 'username')):
                    continue
                in_model = 'chat_model' in path or path[:1] == ('effective_models',)
                if in_model and (key == 'base_url' or (
                        not include_experiment_factors and key in factors)):
                    continue
                result[key] = canonical(item, (*path, key))
            return result
        if isinstance(value, (list, tuple)):
            return [canonical(item, path) for item in value]
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        return type(value).__module__ + '.' + type(value).__qualname__

    decision_model = getattr(adm, '_chat_model', None)
    reviewer = getattr(adm, 'reviewer', None)
    review_model = getattr(reviewer, 'chat_model', None)
    if review_model is None:
        review_model = decision_model
    effective = {}
    for role, model in (('decision', decision_model), ('reviewer', review_model)):
        settings = getattr(model, '_default_params', {})
        effective[role] = settings if isinstance(settings, Mapping) else {}
    # Include effective values as well as Hydra source settings: unresolved
    # environment interpolations alone cannot prove matching model parameters.
    return _digest(canonical({**dict(config or {}), 'effective_models': effective}))


def runtime_manifest(adm=None, model_revision=None, configuration=None):
    """Fingerprint local implementation and prompts without persisting their text."""
    package_root = Path(__file__).resolve().parents[1]
    source_hash = sha256()
    for path in sorted(package_root.rglob('*.py')):
        if 'tests' in path.relative_to(package_root).parts:
            continue
        source_hash.update(str(path.relative_to(package_root)).encode())
        source_hash.update(b'\0')
        source_hash.update(path.read_bytes())
        source_hash.update(b'\0')
    versions = {}
    for name in ('vllm', 'torch', 'transformers', 'outlines', 'outlines_core',
                 'xgrammar', 'openai', 'langchain-core', 'langchain-openai'):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=package_root, text=True,
            stderr=subprocess.DEVNULL, timeout=5).strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    model = getattr(adm, '_chat_model', None)
    model_name = getattr(model, 'model_name', getattr(model, 'model', None))
    return {
        'schema_version': 1, 'python': platform.python_version(),
        'run_id': str(uuid4()), 'started_at': datetime.now(timezone.utc).isoformat(),
        'versions': versions, 'git_commit': commit,
        'implementation_fingerprint': source_hash.hexdigest(),
        'prompt_fingerprint': _digest(_static_prompts(adm)),
        'configuration_fingerprint': _configuration_fingerprint(
            configuration, adm, include_experiment_factors=False),
        'experiment_config_fingerprint': _configuration_fingerprint(
            configuration, adm, include_experiment_factors=True),
        'model': model_name if isinstance(model_name, str) else None,
        # This is the configured revision; the linked server manifest verifies
        # what was actually launched. A model name alone is not a revision.
        'model_revision': model_revision,
    }


class _Recorder:
    def __init__(self, stream, capture_requests):
        self.stream = stream
        self.capture_requests = capture_requests
        self.lock = Lock()

    def write(self, entry):
        encoded = _json_bytes(entry).decode('utf-8')
        with self.lock:
            self.stream.write(encoded + '\n')
            self.stream.flush()


@contextmanager
def model_telemetry_scope(output_dir, *, enabled=True, capture_requests=False,
                          adm=None, model_revision=None, server_manifest=None,
                          configuration=None):
    """Keep measurements in the Hydra run directory and reset state on exit."""
    if capture_requests and not enabled:
        raise ValueError('save_model_requests requires save_model_telemetry')
    if not enabled:
        yield
        return
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = runtime_manifest(adm, model_revision, configuration)
    if server_manifest is not None:
        server_data = Path(server_manifest).read_bytes()
        # Parse before copying so a malformed manifest cannot establish provenance.
        json.loads(server_data)
        manifest['server_manifest_sha256'] = sha256(server_data).hexdigest()
        (directory / 'server_runtime.json').write_bytes(server_data)
    (directory / 'run_runtime.json').write_bytes(_json_bytes(manifest) + b'\n')
    with (directory / 'model_calls.jsonl').open('w', encoding='utf-8') as stream:
        token = _RECORDER.set(_Recorder(stream, capture_requests))
        try:
            yield
        finally:
            _RECORDER.reset(token)


def _contains_private_reasoning(value):
    if isinstance(value, Mapping):
        if value.get('type') in {'reasoning', 'reasoning_text', 'thinking'}:
            return True
        if any(key in value for key in ('reasoning_content', 'thinking')):
            return True
        if isinstance(value.get('reasoning'), (str, list)):
            return True
        return any(_contains_private_reasoning(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_private_reasoning(item) for item in value)
    return False


def _request_metadata(model, messages, kwargs, capture):
    from align_system.algorithms.agents.chat_transport import effective_request_payload

    payload = effective_request_payload(model, messages, kwargs)
    body = {key: value for key, value in payload.items() if key not in _HTTP_FIELDS}
    # Verify serializability before generation, without stringifying SDK clients,
    # credentials, exceptions, or any other opaque objects.
    fingerprint = _digest(body)
    settings = {key: body[key] for key in _SETTINGS if key in body}
    if isinstance(body.get('reasoning'), Mapping):
        settings['reasoning'] = {
            key: value for key, value in body['reasoning'].items()
            if key in ('effort', 'summary')}
    if isinstance(body.get('chat_template_kwargs'), Mapping):
        settings['chat_template_kwargs'] = {
            key: value for key, value in body['chat_template_kwargs'].items()
            if key in ('reasoning_effort', 'enable_thinking', 'preserve_thinking')}
    result = {'request_fingerprint': fingerprint, 'settings': settings}
    if capture:
        if _contains_private_reasoning(body.get('messages', body.get('input'))):
            result['request_capture_error'] = 'private_reasoning_in_history'
        else:
            result['request'] = {
                'api': 'responses' if 'input' in body else 'chat_completions',
                'body': body,
            }
    return result


def invoke_with_telemetry(model, messages, invoke_kwargs=None, *, phase,
                          invoke=None, tokenization_elapsed_s=None):
    """Measure exactly one provider invocation; never retry or alter its request."""
    from align_system.algorithms.agents.chat_transport import (
        generation_metadata, normalize_chat_response,
    )

    kwargs = dict(invoke_kwargs or {})
    invoke = model.invoke if invoke is None else invoke
    recorder = _RECORDER.get()
    if recorder is None:
        return normalize_chat_response(invoke(messages, **kwargs))
    entry = {
        'schema_version': 1, **agent_log_context(), 'phase': phase,
        'role': 'reviewer' if phase in ('review', 'review_repair') else 'decision',
        'tokenization_elapsed_s': tokenization_elapsed_s,
    }
    preparation_start = perf_counter()
    try:
        entry.update(_request_metadata(model, messages, kwargs, recorder.capture_requests))
    except Exception as error:
        # An opaque/custom model can still be timed. Missing request provenance
        # remains explicit and cannot be mistaken for a replayable request.
        entry['request_capture_error'] = type(error).__name__
    entry['telemetry_preparation_s'] = perf_counter() - preparation_start
    start = perf_counter()
    response = None
    try:
        response = normalize_chat_response(invoke(messages, **kwargs))
        return response
    except Exception as error:
        entry['error_type'] = type(error).__name__
        raise
    finally:
        entry['elapsed_s'] = perf_counter() - start
        entry['generation'] = generation_metadata(response)
        try:
            recorder.write(entry)
        except OSError:
            # Recording must not turn a completed model call into a retry or
            # replace its original exception. Do not expose storage error text.
            log.warning('Model telemetry write failed; this call record may be missing or incomplete.')
