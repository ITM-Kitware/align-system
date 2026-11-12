# Recorded attacker-classification replay

This workflow evaluates a local model on recorded system/user dialogues.
Historical replay requires no simulator rollouts or defender checkpoints.
Defender observation capture requires the simulator; active capture also requires
defender checkpoints. Neither requires a hosted inference account or training.
Inference runs as a Slurm batch job and continues after the submitting terminal
disconnects. Input preparation and validation need only Python's standard library.

## Prepare inputs

Use a saved classification run containing each class's final predictions,
request history, and raw dialogue log. The preparation command validates the
source evidence and recovers the logged base messages for those predictions.
Historical retries can add correction messages that the raw dialogue log does
not retain. Provider-side formatting and schema handling are also not recovered.

```sh
python scripts/prepare_classifier_replay.py \
  --run-dir "$HISTORICAL_RUN" --output-dir "$REPLAY_INPUTS"
```

Add `--per-class 1` for a six-request smoke test or `--per-class 10` for a
60-request timing pilot. Each output directory must be new. Subsets use a
recorded fixed sampling seed. A small subset tests execution and throughput;
its accuracy is not a reliable estimate of the full experiment.

The output includes `requests.jsonl` and a manifest of source hashes, selected
records, historical settings, model identity, and historical accuracy. Scoring
labels and historical predictions are stored separately from the messages.

```sh
python scripts/run_classifier_replay.py \
  --requests "$REPLAY_INPUTS/requests.jsonl" --validate-only
```

## Install the inference environment

Keep the virtual environment and package cache on the data drive. The dedicated
lock avoids installing the full simulator and unrelated application dependencies.

```sh
uv venv --python python3.10 .venv
uv pip sync --python .venv/bin/python requirements-classifier-replay.txt \
  --extra-index-url https://download.pytorch.org/whl/cu118 \
  --index-strategy unsafe-best-match
```

Use an existing complete local model snapshot. The job enables offline loading;
missing files fail instead of downloading a second model copy. It uses bfloat16,
greedy JSON-constrained generation, the recorded per-request output limit, and
the model's chat template with a generation prompt. The encoded tokens must
match direct chat-template tokenization, including exactly one beginning token.

## Run on ITM

Read the current machine and workspace instructions first. The provided job uses
the `community` partition, one GPU, four CPUs, 32 GB of host RAM, and a 30-minute
smoke-test limit. It preserves Slurm's `CUDA_VISIBLE_DEVICES`.

Set `REPLAY_CHECKOUT` to the remote source directory, `REPLAY_CACHE_DIR` and
`REPLAY_TMP_ROOT` to personal data-drive locations, and create the log directory
before submitting:

```sh
export REPLAY_CHECKOUT REPLAY_CACHE_DIR REPLAY_TMP_ROOT
sbatch --output="$REPLAY_LOGS/%j.out" --error="$REPLAY_LOGS/%j.err" \
  scripts/classifier_replay.slurm \
  "$REPLAY_INPUTS/requests.jsonl" "$REPLAY_OUTPUT" "$MODEL_SNAPSHOT"
```

Hugging Face configuration stays in the user's private `HF_HOME`. Model and
dataset caches use the machine's shared `/data` locations. The model snapshot
is read in place; the job does not save or copy model weights. Generated output
consists of predictions, manifests, scalar reports, and logs. Estimate the full
job's runtime and output size from the pilot before overriding `--time` for a
production submission.

Each prediction is flushed and synchronized to disk. Resubmitting with the same
input file, model, runtime, runner, and output directory resumes completed work.
A changed run identity fails; use a new output directory for changed settings.
An incomplete final append is preserved separately and replayed. Concurrent
writers to the same output directory are rejected.

## Interpretation

A local model evaluation is not guaranteed to reproduce the numerical accuracy
of a hosted service, even with identical messages and greedy decoding. The
prepared manifest preserves the historical requested and served model IDs and
per-request settings so those differences remain visible. Do not silently use
the current branch's prompt when the goal is to replay historical messages.

The result report distinguishes partial and complete evaluations, gives the
confusion matrix and per-class scores, and records inference time and peak GPU
allocation. Windows from the same trajectory overlap, so they must not be
treated as independent episodes for uncertainty estimates.

## Qwen pilot

`scripts/run_qwen_classifier_replay.py` uses a separate `.venv-qwen` environment
and `requirements-qwen-replay.txt` lock. The initial runtime pins follow
[the published CUDA 12.9 runtime](https://github.com/ITM-Kitware/align-system/blob/8071eb49b0caa8ba8619ef3af9ca053de227bd31/pyproject.toml).
Compile and sync using both `https://wheels.vllm.ai/0.32.0/cu129` and
`https://download.pytorch.org/whl/cu129` as extra indexes. Use Python 3.12.

The Qwen runner keeps the recorded dialogues, output labels, generation limits,
and deterministic sampling. It uses the model's own chat template with thinking
disabled, JSON constraints through xgrammar, and cached FP8 weights. Model and
runtime changes mean this is a new baseline, not an exact replication of a
previous Qwen result. The manifest records the snapshot, tokenizer hashes,
dependency versions, and engine settings. Each result directory retains source
copies. No true label or historical answer is passed to the model.

Use the same submission variables as above, with
`scripts/qwen_classifier_replay.slurm`. Start with the balanced 60-request pilot.
The default allocation is one GPU, four CPUs, 64 GB host RAM, and 30 minutes.
Context is capped at 8192 tokens; any longer input fails before model loading.
The model runs text-only with one sequence, eager execution, and prefix caching.
Use a new output directory for every changed setting.

## Verify and capture defender observations

Defender-only input is the normal classification view. The existing
`CAGEActionBasedServiceInterface` and base classification configuration default to
`observation_fields: [blue_actions, defender_observations]`. Set `trajectory_dir`
to a completed Monitor or active capture directory containing `summary.json`.
The interface verifies capture hashes and reads only primary episodes named in
that summary, excluding repeated captures and metadata. Its window formatter is
also used by both replay preparation scripts.

Historical action-only datasets do not contain Blue's observations. They fail
under the default instead of silently exposing Red actions. For the explicit
privileged baseline, select `observation_fields: [blue_actions, red_actions]`.
The six historical per-class configurations retain that explicit selection.
`prepare_classifier_replay.py` likewise retains the original privileged dialogues
for baseline comparisons. There is no automatic fallback between input views.

`capture_defender_observations.py` runs a recorded Monitor episode twice with
the saved environment seed. It compares both sides' actions and success flags,
initial network data, and per-step CIA/resilience scores with the reference.
It retains the defender's raw observation and `BlueTableWrapper` table after
each action. It never uses the true-state table as classifier input.

`verify_recorded_episodes.py` finds the unique episodes referenced by a replay
input file and repeats each capture in fresh Python processes. The summary
records exact-match checks and hashes. The corresponding Slurm entrypoint is
CPU-only. Use the separate `requirements-observation-capture.txt` environment
and a pinned simulator source directory. These checks establish consistency
with retained historical evidence; the historical defender observations were
not independently recorded, so their exact identity cannot be proven.
An action, score, or initial-network mismatch marks the audit failed and exits
unsuccessfully. Both Monitor preparation and active capture reject failed audits.

`prepare_defender_replay.py` refuses incomplete or mismatched captures. It
verifies every recovered attacker-action window against the historical prompt,
then produces paired inputs with only the current observation section changed.
The original privileged inputs are copied unchanged. Instructions, examples,
labels, sampling settings, and window boundaries remain fixed. Historical
examples still describe attacker actions; defender-specific prompt adaptation
is a separate experiment. For Qwen's longer table inputs, use
`qwen_defender_replay.slurm`, which records its submitted launcher and permits
16384 context tokens. `check_classifier_context.py` checks all prompts on CPU
before submission.

## Active hybrid defenders

`capture_active_defender.py` uses the same initial networks, environmental
seeds, attackers, and observation schedule as the verified Monitor controls.
It balances episodes over five PPO checkpoint training seeds. Stochastic policy
sampling is seeded and repeated; `--policy-deterministic` instead selects greedy
actions. Policy inference runs on CPU. The runtime first reproduces each Monitor
control, then captures both defender-visible observations and separate attacker
actions for the active episode. Checkpoint hashes and action mappings are saved.
The simulator source is fingerprinted directly and must match the verified
Monitor capture before loading a policy or generating active episodes.

Install `requirements-active-capture.txt` in a separate environment. Use
`capture_active_defender.slurm` for capture and `prepare_active_replay.py` to
produce classifier inputs. The resulting active trajectories are new episodes
paired to the historical controls, not historical trajectory replays. Captures
do not switch policies in response to LLM predictions.

## Compare and retain evidence

`summarize_classifier_comparison.py` reads a configuration listing each input
file, output directory, model, and comparison pair. It verifies input hashes and
model identities, joins changed input views by their original request IDs, and
writes `comparison.json` and `scores.csv`. Partial results remain explicitly
marked. CPU report jobs can depend on inference-job completion through
`classifier_comparison_report.slurm`. Updates use a lock and atomic replacements.

`analyze_observation_identifiability.py` counts identical model inputs associated
with different true labels. Its majority-label score is an optimistic in-sample
ceiling for deterministic classifiers, not a generalization estimate.

Keep each experiment's input manifests, source snapshot, dependency locks,
checkpoint/model locators and hashes, predictions, logs, and reports together.
Model weights and environments stay in their existing caches or personal code
directories. Never overwrite historical result directories.

Inference manifests and reports identify the input view, defender condition, and
held-fixed historical prompt protocol. Source hashes and retained files identify
the code that actually ran. Keep its Git revision and bundle locator in the run
registry. A code review or later fix gets a separate revision and record; it must
not change the revision attributed to existing runs. Use a new output directory
for inference with revised code, because resume checks include source hashes.
Native classification manifests also retain the audit-summary hash, input-file
hashes, source hashes, and Git revision and dirty state when run from a checkout.
