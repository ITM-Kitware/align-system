"""Self-managed vLLM backend for LangChain tool-calling agents.

vLLM's tool-call parsing lives in its OpenAI-compatible server (not in
the in-process ``vllm.LLM`` API), so a LangChain agent talks to vLLM
through ``langchain_openai.ChatOpenAI`` pointed at a ``vllm serve``
endpoint.  ``VLLMServerChatModel`` removes the manual serving step: it
starts the server itself on first use (reusing one already listening at
``base_url``) and shuts it down at exit.
"""

import atexit
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from timeit import default_timer as timer
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from align_system.utils import logging

log = logging.getLogger(__name__)

LOCAL_HOSTS = {'localhost', '127.0.0.1', '0.0.0.0', '::1'}


class VLLMServerChatModel:
    """Chat model that serves `model` with vLLM's OpenAI-compatible
    server and delegates to ``langchain_openai.ChatOpenAI``.

    The server is managed lazily: nothing is started at construction
    time, so instantiating configs stays cheap.  On first use (or an
    explicit ``ensure_ready()``), a server already running at
    ``base_url`` is reused; otherwise -- for a local ``base_url`` --
    ``vllm serve`` is launched as a subprocess with tool calling
    enabled, waited on until it answers, and terminated when the
    process exits.

    Any extra keyword arguments are passed through to ``ChatOpenAI``
    (e.g. ``temperature``); ``serve_args`` appends raw CLI arguments to
    the ``vllm serve`` command (e.g. ``['--max-model-len', '8192']``).
    """

    def __init__(self,
                 model,
                 base_url='http://localhost:8000/v1',
                 api_key='EMPTY',
                 tool_call_parser='hermes',
                 enable_auto_tool_choice=True,
                 serve_args=None,
                 startup_timeout_s=600,
                 **chat_model_kwargs):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.tool_call_parser = tool_call_parser
        self.enable_auto_tool_choice = enable_auto_tool_choice
        self.serve_args = [str(a) for a in (serve_args or [])]
        self.startup_timeout_s = startup_timeout_s
        self.chat_model_kwargs = chat_model_kwargs

        self._client = None
        self._server_process = None

    # -- LangChain chat model surface (delegated) ---------------------
    # Note this duck-typed surface is all this class provides; in
    # particular bind_tools returns the underlying ChatOpenAI runnable,
    # so everything downstream of binding bypasses this wrapper (the
    # server is guaranteed up by then)

    def bind_tools(self, tools, **kwargs):
        return self._ensure_client().bind_tools(tools, **kwargs)

    def invoke(self, *args, **kwargs):
        return self._ensure_client().invoke(*args, **kwargs)

    # -- Server management --------------------------------------------

    def ensure_ready(self):
        """Make sure a vLLM server is answering at `base_url`, starting
        one if needed; called implicitly on first use."""
        self._ensure_client()

    def _ensure_client(self):
        if self._client is None:
            if not self._server_is_up():
                self._start_server()

            from langchain_openai import ChatOpenAI
            self._client = ChatOpenAI(
                model=self.model,
                base_url=self.base_url,
                api_key=self.api_key,
                **self.chat_model_kwargs)

        return self._client

    def _served_models(self):
        """The model ids served at `base_url`, or None when no server
        answers there."""
        request = Request(
            f"{self.base_url.rstrip('/')}/models",
            headers={'Authorization': f'Bearer {self.api_key}'})
        try:
            with urlopen(request, timeout=5) as response:
                if response.status != 200:
                    return None
                return [m.get('id') for m in
                        json.load(response).get('data', [])]
        except (URLError, OSError, ValueError):
            return None

    def _server_is_up(self):
        served = self._served_models()
        if served is None:
            return False

        if self.model not in served:
            # Somebody else's server on this port; failing beats
            # silently chatting with the wrong model
            raise RuntimeError(
                f"The server at {self.base_url} is serving "
                f"{served}, not {self.model}; stop it or point "
                "base_url at a free port")

        return True

    @staticmethod
    def _vllm_executable():
        # Prefer the `vllm` console script of the running interpreter's
        # environment over whatever is first on PATH
        candidate = Path(sys.executable).with_name('vllm')
        if candidate.is_file():
            return str(candidate)

        on_path = shutil.which('vllm')
        if on_path is not None:
            return on_path

        raise RuntimeError(
            "Cannot find the `vllm` command to serve "
            f"{VLLMServerChatModel.__name__}'s model; is vllm installed "
            "in this environment?")

    def _start_server(self):
        parsed = urlparse(self.base_url)
        if parsed.hostname not in LOCAL_HOSTS:
            raise RuntimeError(
                f"No vLLM server answering at {self.base_url}, and it "
                "is not a local address this process can start a "
                "server on")

        command = [self._vllm_executable(), 'serve', self.model,
                   '--host', parsed.hostname,
                   '--port', str(parsed.port or 8000)]
        if self.enable_auto_tool_choice:
            command.append('--enable-auto-tool-choice')
        if self.tool_call_parser:
            command.extend(['--tool-call-parser', self.tool_call_parser])
        command.extend(self.serve_args)

        server_log = tempfile.NamedTemporaryFile(
            mode='w', prefix='vllm_serve_', suffix='.log', delete=False)

        log.info(f"Starting vLLM server: {' '.join(command)} "
                 f"(log: {server_log.name})")

        self._server_process = subprocess.Popen(
            command, stdout=server_log, stderr=subprocess.STDOUT)
        atexit.register(self._stop_server)

        start = timer()
        while timer() - start < self.startup_timeout_s:
            if self._server_process.poll() is not None:
                raise RuntimeError(
                    "vLLM server exited during startup (status "
                    f"{self._server_process.returncode}); see "
                    f"{server_log.name}")

            # Not _server_is_up: this is our own server coming up, so
            # a not-yet-registered model just means keep waiting (the
            # foreign-server check ran before starting it)
            if self.model in (self._served_models() or []):
                log.info(f"vLLM server for {self.model} is up at "
                         f"{self.base_url}")
                return

            time.sleep(2)

        self._stop_server()
        raise RuntimeError(
            f"vLLM server did not come up within "
            f"{self.startup_timeout_s}s; see {server_log.name}")

    def _stop_server(self):
        if self._server_process is None:
            return

        if self._server_process.poll() is None:
            log.info("Shutting down managed vLLM server")
            self._server_process.terminate()
            try:
                self._server_process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self._server_process.kill()

        self._server_process = None
