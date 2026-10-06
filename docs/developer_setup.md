## Installation from source

We use [uv](https://docs.astral.sh/uv/) to manage dependencies. Install uv
using its [installation instructions](https://docs.astral.sh/uv/getting-started/installation/).
From a local clone of this repository, install the project and its locked
dependencies with Python 3.12 (Python 3.10–3.12 is supported):

```bash
uv sync --locked --python 3.12
```

uv creates a `.venv` in the repository and installs the project in editable
mode. Run commands from the repository root with `uv run`:

```bash
uv run run_align_system
```

Alternatively, activate the environment with `source .venv/bin/activate`
and run `run_align_system` directly.

## Optional backend dependencies

Dependency groups in `pyproject.toml` provide the `openai`, `anthropic`,
`langchain-agent`, and `llama-index-retriever` integrations. Include the
required group when syncing and running, for example:

```bash
uv sync --locked --group openai
uv run --group openai run_align_system
```

The selected ADM or driver still needs its corresponding Hydra configuration
and any provider credentials or local model server.
