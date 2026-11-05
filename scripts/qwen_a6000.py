#!/usr/bin/env python3
"""Preflight and launch the pinned Qwen server; never changes drivers or installs packages."""
import argparse
import errno
import importlib.metadata
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys

MODEL = 'Qwen/Qwen3.8-27B-FP8'
VERSION = '0.32.0+cu129'


def command(args):
    legacy = getattr(args, 'runtime', 'modern') == 'legacy'
    result = [sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
              '--model', MODEL, '--revision', args.revision,
              '--tensor-parallel-size', '1', '--language-model-only',
              '--max-model-len', '65536', '--max-num-seqs', str(args.max_num_seqs),
              '--max-num-batched-tokens', str(args.max_num_batched_tokens),
              '--gpu-memory-utilization', '0.90', '--kv-cache-dtype', 'auto',
              '--enable-chunked-prefill',
              '--reasoning-parser', 'qwen3', '--enable-auto-tool-choice',
              '--tool-call-parser', 'qwen3_xml' if legacy else 'qwen3_coder', '--host', '127.0.0.1',
              '--port', str(args.port)]
    result += ['--enable-prefix-caching' if args.prefix_cache else '--no-enable-prefix-caching']
    result += ['--mamba-cache-mode', 'align' if args.prefix_cache else 'none']
    if args.mtp:
        result += ['--speculative-config', json.dumps({
            'method': 'mtp', 'num_speculative_tokens': args.mtp})]
    return result


def capture(argv):
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=20)
        return {'returncode': proc.returncode, 'stdout': proc.stdout.strip(),
                'stderr': proc.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'returncode': None, 'error': str(exc)}


def selected_device(rows, visible_devices):
    """nvidia-smi ignores CUDA_VISIBLE_DEVICES, so filter its physical inventory."""
    if visible_devices is None:
        if len(rows) != 1:
            raise ValueError('Set CUDA_VISIBLE_DEVICES to one GPU index or full UUID on a multi-GPU host.')
        return rows[0]
    selected = visible_devices.strip()
    if not selected or ',' in selected:
        raise ValueError('CUDA_VISIBLE_DEVICES must select exactly one GPU index or full UUID.')
    matches = [row for row in rows if selected == row[0] or selected == row[3]]
    if len(matches) != 1:
        raise ValueError('CUDA_VISIBLE_DEVICES does not identify one physical GPU in nvidia-smi.')
    return matches[0]


def same_gpu_uuid(left, right):
    def normalized(value):
        return str(value).lower().removeprefix('gpu-').replace('-', '')
    left, right = normalized(left), normalized(right)
    return bool(re.fullmatch(r'[0-9a-f]{32}', left)) and left == right


def preflight(args):
    legacy = getattr(args, 'runtime', 'modern') == 'legacy'
    expected_version = '0.19.0' if legacy else VERSION
    expected_cuda = '12.8' if legacy else '12.9'
    minimum_driver = (570, 26) if legacy else (575, 57, 8)
    packages = {}
    for name in ('vllm', 'torch', 'transformers', 'xgrammar', 'flashinfer-python'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    devices = capture(['nvidia-smi', '--query-gpu=index,name,memory.total,uuid,driver_version',
                       '--format=csv,noheader,nounits'])
    issues = []
    selected_uuid = None
    rows = [row.split(', ') for row in devices.get('stdout', '').splitlines() if row]
    if devices['returncode'] != 0 or not rows:
        issues.append('Cannot inspect NVIDIA hardware with nvidia-smi.')
    else:
        try:
            row = selected_device(rows, os.environ.get('CUDA_VISIBLE_DEVICES'))
            selected_uuid = row[3]
            devices['stdout'] = ', '.join(row)
            devices['physical_device_count'] = len(rows)
            if 'RTX A6000' not in row[1]:
                issues.append('The selected device must be an NVIDIA RTX A6000.')
            driver = tuple(int(n) for n in row[-1].split('.'))
            if driver < minimum_driver:
                floor = '.'.join(map(str, minimum_driver))
                issues.append(f'Driver must be >={floor} for the selected CUDA {expected_cuda}/JIT baseline.')
        except (ValueError, IndexError) as exc:
            issues.append(str(exc))
    if packages['vllm'] != expected_version:
        issues.append(f'Expected vllm=={expected_version}; found {packages["vllm"]!r}.')
    if not legacy and sys.version_info < (3, 11):
        issues.append('vLLM 0.32.0 requires Python >=3.11.')
    if legacy and sys.version_info[:2] != (3, 10):
        issues.append('The legacy comparison profile preserves the original Python 3.10 environment.')
    if legacy and args.mtp:
        issues.append('MTP is disabled for the legacy comparison baseline.')
    if os.environ.get('VLLM_BATCH_INVARIANT', '0') not in ('0', ''):
        issues.append('VLLM_BATCH_INVARIANT is incompatible with Ampere FP8 Marlin.')
    if not re.fullmatch(r'[0-9a-f]{40}', args.revision):
        issues.append('--revision must be an immutable 40-character Hugging Face commit SHA.')
    try:
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(('127.0.0.1', args.port))
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            issues.append(f'Port {args.port} is already in use; stop the previous server first.')
        else:
            issues.append(f'Cannot verify port {args.port} availability: {exc}.')
    cuda = capture([sys.executable, '-c',
                    'import torch; print(torch.version.cuda); print(torch.cuda.device_count()); print(torch.cuda.get_device_capability()); print(torch.cuda.get_device_properties(0).uuid)'])
    if cuda['returncode'] != 0:
        issues.append('PyTorch CUDA device initialization failed.')
    elif not cuda['stdout'].startswith(expected_cuda + '\n1\n(8, 6)'):
        issues.append(f'Expected CUDA {expected_cuda}, exactly one visible device and SM86; inspect torch_cuda in the manifest.')
    elif selected_uuid is not None:
        torch_lines = cuda['stdout'].splitlines()
        if len(torch_lines) != 4 or not same_gpu_uuid(selected_uuid, torch_lines[-1]):
            issues.append('PyTorch GPU UUID does not match the selected nvidia-smi device. Use its full UUID in CUDA_VISIBLE_DEVICES.')
    return {'schema_version': 1, 'runtime': 'legacy' if legacy else 'modern',
            'model': MODEL, 'revision': args.revision,
            'packages': packages, 'python': sys.version, 'hardware': devices,
            'torch_cuda': cuda, 'command': command(args), 'issues': issues,
            'kernel_expectation': 'FP8 weight-only Marlin; no native Ampere FP8 W8A8',
            'environment': {key: os.environ[key] for key in (
                'CUDA_VISIBLE_DEVICES', 'VLLM_USE_FLASHINFER_SAMPLER',
                'CUDA_DEVICE_ORDER', 'VLLM_BATCH_INVARIANT') if key in os.environ}}


class ServerStop(BaseException):
    def __init__(self, signum):
        self.signum = signum


def stop_server_group(process):
    """Bound shutdown and clean up our descendants even if the leader exits first."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--runtime', choices=('modern', 'legacy'), default='modern')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--max-num-seqs', type=int, choices=(1, 2, 4, 8), default=8)
    parser.add_argument('--max-num-batched-tokens', type=int, choices=(2048, 4096, 8192), default=2048)
    parser.add_argument('--prefix-cache', action=argparse.BooleanOptionalAction, default=None,
                        help='Defaults to enabled for modern and disabled for legacy.')
    parser.add_argument('--mtp', type=int, choices=(0, 1, 3), default=0)
    parser.add_argument('--launch', action='store_true', help='Actually launch after all preflight checks pass.')
    args = parser.parse_args()
    if args.prefix_cache is None:
        args.prefix_cache = args.runtime == 'modern'
    # Retain the existing host workaround unless the operator explicitly opts in.
    os.environ.setdefault('VLLM_USE_FLASHINFER_SAMPLER', '0')
    manifest = preflight(args)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'server_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))
    if manifest['issues']:
        return 2
    if not args.launch:
        return 0
    with (args.output / 'server.log').open('w') as logfile:
        process = subprocess.Popen(manifest['command'], stdout=logfile,
                                   stderr=subprocess.STDOUT, start_new_session=True)

        def forward(signum, _frame):
            # Unwind Popen.wait before waiting again; avoid its non-reentrant lock.
            raise ServerStop(signum)

        previous = {sig: signal.signal(sig, forward) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            return process.wait()
        except ServerStop as stop:
            return 128 + stop.signum
        finally:
            # A second signal must not interrupt cleanup and leave GPU workers.
            for sig in previous:
                signal.signal(sig, signal.SIG_IGN)
            try:
                stop_server_group(process)
            finally:
                for sig, handler in previous.items():
                    signal.signal(sig, handler)


if __name__ == '__main__':
    raise SystemExit(main())
