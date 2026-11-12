# Recovering an agent run

The LangChain open-world driver accepts `resume_from=/absolute/path/to/old/run`.
It starts a fresh TA3 session, replays recorded actions in episode order, and
continues model decisions at the end of the last unfinished episode. Completed
episodes replay without actor or toolkit inference. This feature works with
the baseline driver and optional toolkits; it does not require an alignment
toolkit.

Keep the same scenario profile, alignment settings, model, and toolkit settings.
The server must reproduce the same scenario/target sequence and environment
states. Recovery checks the target, initial state, action availability, and
recorded action results, stopping when they diverge. Replay reconstructs the
environment; a conversation checkpoint alone cannot restore a lost TA3 session.

## Inspecting the source

Validate the source files and describe the replay plan without contacting TA3:

```bash
python -m align_system.cli.inspect_agent_resume /absolute/path/to/old/run
```

Both `agent_events.jsonl` and `input_output.json` are required. New runs write
these records as actions execute; keep `save_input_output=true` enabled.
Preserve the entire run directory, including
its `.hydra` configuration and `agent_checkpoint.json` when present.

## Resuming

Use the source run's saved configuration and a **new** output directory:

```bash
run_align_system \
  --config-path /absolute/path/to/old/run/.hydra --config-name config \
  ++resume_from=/absolute/path/to/old/run \
  hydra.run.dir=/absolute/path/to/recovered/run
```

An endpoint may change after an outage. For example, if the saved chat model
uses an OpenAI-compatible server, append
`driver.chat_model.base_url=http://127.0.0.1:8000/v1` to the command above.
Keep the model and experiment settings identical to the source. The CLI checks
the recovery configuration before constructing the live interface.

Starting recovery submits actions to a fresh TA3 session. It does not attach to
the old session. The original files remain untouched, and the new journal and
action records identify the source run, episode, and action sequence. If replay
is interrupted, recover again from the same original source into another new
output directory.

## Saved context and replay boundaries

New runs save `agent_checkpoint.json` before model invocations by default;
`save_agent_checkpoint=false` disables it. At a matching action boundary,
recovery restores the conversation and saved assessment context. Patient
tracking is reconstructed through the normal action handler during replay.
Toolkits may still need to rebuild private in-memory state that they do not
store in the session checkpoint.

If no current checkpoint exists, recovery reconstructs complete assistant/tool
exchanges from the journal and adds a recovery explanation. Missing assistant
prose is not invented. This provides usable continuation, without guaranteeing
identical future model outputs. An older last action requires a complete saved
tool observation to validate its result; newer journals also save raw states
before and after each successful action.

The journal also records terminal attempts whose HTTP response failed while a
subsequent state read confirmed completion. Those attempts are replayed and
completion is checked, even when they are absent from the successful-action
trace. Other unconfirmed attempts are not automatically replayed. The agent
continues from the validated successful prefix and decides what to do next.
A truncated final journal line is tolerated; earlier corruption is rejected.

Recovery supports the LangChain open-world driver without an advisory reviewer.
It does not restore reviewer cycles or resume a separate agentic ADM. Replay
time is excluded from new model decision time. Replaying a fully completed
source reconstructs all its episodes; use the inspector for offline validation.
