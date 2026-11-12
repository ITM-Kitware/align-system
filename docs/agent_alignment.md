# Agent alignment experiments

Compare baseline, target-aware, and Kaleido agents using the same scenarios
and independently supplied alignment labels.
See the [README](../README.md#slurm-launchers) for April OpenWorld launch commands.

## PR #295 driver integration

The shared `ITMOpenWorldLangChainDriver` incorporates the workflow, grounded
reviewer, history packing, request budgets, and logging from
[PR #295](https://github.com/ITM-Kitware/align-system/pull/295), head
`fda93690432d352d52d06ff4211a40686fa0fc5e`. The existing Kaleido toolkit, exact action-ID selection, patient safeguards, event
journal, and score recording remain integrated with this driver.

The normal launchers use the updated driver. Advisory review remains optional,
as in the PR. Enable it on the existing Kaleido launcher with:

```bash
TOOLKIT=kaleido sbatch phase2_ow_agent_qwen38.sbatch driver/reviewer=weighted
# Or omit numerical switching guidance:
TOOLKIT=kaleido sbatch phase2_ow_agent_qwen38.sbatch driver/reviewer=qualitative
```

The reviewer shares the decision model unless configured otherwise. It reviews
a proposal without executing it; the decision agent then makes the final choice
and must acknowledge usable advice with verified visible evidence. Both model
roles consume the per-action call budget. `driver.reviewer=null` disables review.
The dependency files include the PR's pinned evaluation client revision; sync
the deployment environment with the `langchain-agent` dependency group.

The Hydra driver config now enables the PR's context budgeting and shared scalar
KDMA guidance. That guidance treats targets as medical-versus-KDMA preferences,
not raw action-feature scores to match, and takes precedence over toolkit
distance hints. The default `scene_end_policy=auto` permits aligned agents to
choose an offered `END_SCENE`; use `driver.scene_end_policy=legacy` to retain
automatic scene closure. Record these settings when comparing experiments.

Run 3972 exposed a separate issue still present in the upstream PR: a final
evacuation returned HTTP 500, followed by HTTP 400 `"Scenario Complete"`.
The integrated driver reads back state once after an ambiguous action error or
an explicit completion response. A confirmed final state supplies the actual
score and allows the run to continue. The event journal marks the failed-response
action outcome as unconfirmed; it is excluded from the successful-action trace.
The driver never resubmits that action or invents
a completion/score; failed readback or an unconfirmed HTTP 500 aborts explicitly.

## Experimental controls

### Kaleido patient assessments

Kaleido now defaults to `assessment_subject: patients`. The agent can call
`assess_patients(decision="treat"|"evacuate", justification="...")`.
It compares prioritizing individual patient IDs using current observed
descriptions, injuries, vitals, demographics, tags, current-scene care tracking,
and labeled descriptions from earlier observations. Unknown current details
remain unknown; hidden KDMA labels are excluded. Previously observed descriptions
may contain stale clinical facts and are explicitly labeled as history.

`treat` means prioritizing care: a visible casualty may need navigation or
assessment first. It does not assert that a specific treatment is executable.
`evacuate` compares patients permitted by the offered evacuation action, excluding
already evacuated patients. Fixed server patient constraints are preserved.
An unseen evacuation candidate includes only identity and any previously observed
description, without exposing hidden current clinical fields.

By default, `auto_assess=false`: the agent requests assessments on demand.
With `auto_assess=true`, patient evidence is delivered before decisions, including
when a single evacuation template represents multiple patient choices. With
`auto_assess=false`, the agent can request it on demand. The tool does not act;
its structured evidence is recorded under `kaleido_patients` alongside the next
executed action. Repeated calls reuse scores only while evidence, target, and
eligible choices remain unchanged.

Each row contains relevance, support/opposition/indifference probabilities,
`supports + 0.5 * either` as a support estimate, and a model explanation.
These are advisory value signals, **not** clinical urgency scores, calibrated
TA1 scores, or a target-distance ranking. The driver's alignment guidance
determines how the target weights medical urgency versus the KDMA.
Patient explanations add inference cost; set
`driver.extra_tools.kaleido.patient_explain=false` to disable them.

Select the medium-sized patient prompt with
`driver.extra_tools.kaleido.patient_prompt_format=compact` (default: `full`).
For example:

```bash
sbatch phase2_ow_agent_qwen38_kaleido.sbatch \
  driver.extra_tools.kaleido.patient_prompt_format=compact
```

Compact mode deterministically formats the decision, scene, selected patient,
and **all** eligible alternatives as prose. It retains observed demographics,
injuries, vitals, proximity, tags, and scene care tracking. History omits exact
sentences already in the current description or a newer history entry; remaining
sentences keep their original wording and are labeled historical, potentially
stale, newest first. This is conservative text deduplication, not semantic
extraction of relevant facts. It removes JSON wrappers, action metadata, history
provenance metadata, and repeated guidance from the model input. The value
definition, model, scoring, and full evidence returned to the agent are unchanged.
The assessment record includes `patient_prompt_format` for comparison of runs.
Prompt length still depends on the observed facts; there is no token cap or
two-patient reduction. `auto_assess=false` remains the default. Compare both
formats on the same saved observations before drawing conclusions about quality.

The existing `assess_available_actions` tool remains available. Set
`driver.extra_tools.kaleido.assessment_subject=actions` to restore its automatic
assessment and prompt behavior. No TA3 configuration or model retraining is
required for the patient tool. Improvements in decision quality still require
evaluation; earlier offline checks validated evidence handling and integration
only.

### Reset awareness and navigation (run 3965)

Run 3965's saved observations confirm a mismatch between treatment descriptions
and the driver's scene-completion credit. In the affiliation 0.0 episode,
action 17 changed `treat_and_tag` to `building_explosion`. The resulting tool
observation still described Patient 5's stomach injury and Patient 9's lower-body
injury as treated. The saved system prompt instructed the actor to move on from
treated/tagged patients without explaining scene resets. Later, the actor said
all 12 patients had been treated and attempted to end the scene. The driver
correctly retained two unresolved patients under its current reset rule.
This is evidence of ambiguous presentation and inconsistent agent tracking,
not proof that affiliation caused the failure or that the reset rule itself is
a medically meaningful treatment requirement.

The original artifacts are on `itm` under
`/data/users/vincenzo.dimatteo/code/cleanup/align-system/outputs/ow_agent/Qwen/Qwen3.8-27B-FP8/live/Kaleido-OnDemand-TargetAware-APRIL_OPENWORLD3/3965`.

| Recorded episode | Completed | Post-explosion treatments | Moves | Longest move sequence | Total actions | Revisit proxy |
| --- | --- | --- | --- | --- | --- | --- |
| Merit 0.0 | Yes | 12/12 | 17 | 5 | 77 | 5 |
| Affiliation 0.0 | No | 10/12 | 47 | 16 | 100 | 28 |

The missing post-explosion treatments are Patients 5 and 9. The revisit proxy
counts another move to a patient without an intervening successful check,
treatment, tag, or evacuation for that patient in the same scene. It can include
clinically justified visits; it must not be described as a validated count of
unnecessary movement. These are single episodes, not estimated completion rates
across seeds. The journal does not contain complete per-call system messages,
so the exact dynamically appended prompt cannot be certified from the saved
system-prompt log alone.

The driver also carries the reset-handling context from
`dev/agentic-ow-reviewer-baseline`: every model request includes current visible
observations and `handled_patients` (treated, evacuated, and tagged in the current
scene), independently of conversation trimming. Scene transitions clear that
care tracking and invalidate action listings and assessments. Action listings
are refreshed before the next model request, including after automatic scene
closure.

`patient_history` retains exact visible patient descriptions across scenes,
deduplicated by patient ID and text with first/last observation provenance.
Last-known tags continue to use the existing tag memory. Historical clinical
descriptions and tags do not override current care credit or action eligibility.
Hidden descriptions and evaluation labels are excluded from this context.
The history used for a decision is saved in `choice_info.patient_history`.
This context is always enabled, so new baseline runs differ from older runs
without it; compare experimental switches using the same code revision.

Two independent, default-off switches add further interventions:

- `driver.reset_awareness=true` adds a patient progress ledger to observations,
  action results, and every model request, independently of conversation
  trimming. It separates previous treatment, current treatment credit, reset
  reason, current-scene tagging, and historical tags. Scene transitions and
  already-supported clinical invalidations are explained without changing
  completion, action budgets, triage policy, or the underlying reset rules.
- `driver.navigation_safeguards=true` checks that a movement target has a
  pending offered care action. Repeating a visit without changed care or
  disclosed clinical facts requires `reassess_navigation` with a pending next
  action and justification. This permits a necessary return after an explicit
  reassessment; it does not choose a patient or a treatment for the actor.
  Fixed-ID calls and fallback obey the same guard. New scenes clear the visit
  history; lost visibility alone does not. Evacuation-only movement retains
  its existing rules.

The `patient_progress` journal event records driver credit after successful
actions in both conditions. The active switches are recorded in `episode_context`.
With reset awareness enabled, the ledger shown in an action's decision is also
saved in `choice_info.patient_progress`. Hidden clinical fields are never added
to the ledger. Recorded old actions can be replayed against these assertions;
that validates bookkeeping and presentation, not new LLM behavior.

For the primary two-by-two comparison, use both targets (`merit-0.0` and
`affiliation-0.0`) with reset awareness off/on and navigation safeguards **off**.
Add a third condition with both on to measure the additional navigation effect.
Otherwise any improvement could be caused by the movement restriction rather
than reset awareness. Preserve the source snapshot and resolved configuration
for each condition, and use a fresh output directory for every run.

Use paired model seeds, for example 11, 22, and 33. Keep Qwen3.8-27B-FP8,
temperature 0.0, 8192 output tokens, 12 retained messages, the 100-action cap,
Kaleido on demand, target context, and the same server scenario/reset rules.
The local Python/NumPy/Torch seeds do not seed a remote model; set
`++driver.chat_model.seed=SEED` as well. At temperature zero, different seeds
need not produce different behavior. Do not silently increase temperature to
manufacture variability.

For example, the existing launcher accepts the controlled settings below.
Replace `MODEL_URL`, `SEED`, `ATTRIBUTE`, and `RESET_AWARE` consistently for each
cell; run these only on a host with the chosen model and Kaleido available:

```bash
RUN_MODE=live TOOLKIT=kaleido AUTO_ASSESS=false TARGET_CONTEXT=true \
SCENARIO_ID=April2026-OW_desert3 VLLM_URL="$MODEL_URL" \
OUT_DIR="outputs/reset-study/$SEED/$ATTRIBUTE/reset-$RESET_AWARE" \
bash phase2_ow_agent_qwen38.sh \
  "+alignment_target=june2025/ADEPT-June2025-$ATTRIBUTE-0.0" \
  "driver.reset_awareness=$RESET_AWARE" \
  driver.navigation_safeguards=false \
  driver.max_actions_per_scenario=100 \
  driver.max_messages_in_context=12 \
  driver.chat_model.temperature=0.0 driver.chat_model.max_tokens=8192 \
  "++driver.chat_model.seed=$SEED" \
  "++random_seed=$SEED" "++numpy_random_seed=$SEED" "++torch_random_seed=$SEED"
```

The explicit alignment target controls actor/toolkit behavior. It does not
change a target assigned by TA3 for scoring. This study concerns behavior and
completion; do not reinterpret server scores as scores for an overridden
target. The TA3 client exposes neither an environment seed nor restoration of
a saved initial state. Verify the initial `episode_context.full_state` across
paired runs and record server/scenario versions; if they differ, the comparison
is not controlled. Exact environment-seed pairing requires server support.
Live multi-seed results are still pending; offline replay does not fill the
retest cells.

`TARGET_CONTEXT=true` adds the requested target and its attribute definitions to
the actor's opening message, which remains pinned when conversation history is
trimmed. This performs no model calls and provides no action scores. Use it with
`TOOLKIT=baseline` for the target-aware baseline. Full distribution target entries
are preserved. The direct Hydra experiment is
`phase2_langchain_agent_ow_target_live`.

## Server scores and missing-score diagnostics

With `save_scoring_output=true` (the default), the driver creates `scores.json`
and `scoring_status.json` at startup. Structured score API responses keep their
existing format in `scores.json` and are saved after each completed scenario,
so a later failure does not discard earlier scores.

`scoring_status.json` records every started scenario as `pending`, then updates
it to `available`, `unavailable`, `incomplete`, `stalled`, or `aborted`. An abruptly killed
process can leave a pending entry. Completed scenarios include separate `api`
and `final_state` outcomes. The latter preserves the returned final text and
extracts a finite numeric `Session alignment score = ...` when present;
`Error`, `Not requested`, `Not calculated`, and missing text are not scores.
Text-derived scores stay in this diagnostic file and are not presented as full
API responses in `scores.json`.

If every valid recovery action is rejected, the episode is `stalled` and its
state and rejection diagnostics are retained under `stall`. No final score is
requested. The TA3 interface currently exposes no explicit scenario abort or
reset operation, so the driver stops without pretending the scenario completed
or advancing an unfinished server session.

Reaching `max_actions_per_scenario` also stops the run with a nonzero exit,
without forcing `END_SCENE` or requesting the next scenario. Ending a scene
can advance to evacuation while leaving the scenario active. The episode is
recorded as `incomplete` with `ScenarioActionLimitError`; its action count,
limit, current state, pending patients, and known care are saved under `stall`
and in `<episode_id>.stall.json`. A completed scenario at the exact action
limit still receives normal scoring.

Patient progress uses tags observed or successfully applied in the current
scene, even after those tags disappear from distant observations. Unseen
patients with completed care are not listed as casualties to reach within that
scene. Every scene-ID change resets per-scene treatment, vitals, tagging,
evacuation, and movement guards, including the associated clinical snapshots.
There are no scenario-specific exceptions: prior treatment does not count as
current-scene completion merely because patient IDs or reported facts match.
Existing patients can be checked and treated again when those actions are
offered. Prior-scene tag records remain explicitly historical until freshly
observed; currently visible tags are recorded as new-scene evidence. Within a
scene, newly disclosed injuries or changed structured clinical fields reopen
treatment, while missing fields alone do not erase care progress.

The recorded April trace ends with six patients treated only before its scene
change. Under the reset rule that trace remains incomplete; its old completion
count is not a completion guarantee. Earlier replay checks verified both this
boundary and that fresh checks and treatment are executable after a transition.
Actual agent behavior and completion still require a new live run.

An EXPECTANT tag alone does not establish that treatment is complete: the
driver has no validated general rule for a no-treatment disposition. This fix
retains the existing treatment requirement rather than inferring one from a
triage label. Completion without feasible actions retains its existing rule.

When the server offers `MOVE_TO_EVAC`, server-listed patient IDs remain eligible
even when currently unseen or distant. Evacuation does not require a preceding
`MOVE_TO`; the server may offer evacuation alone while preserving visibility
from the treatment scene. Fixed patient targets and duplicate evacuation guards
still apply. The actor, action filter, concrete assessments, and fallback use
the same eligibility rule. Hidden clinical fields remain excluded from the
actor's observation, and checking vitals, treatment, and tagging still require
visible patients who are not known to be distant. Evacuation-only decisions
receive a list of evacuation candidates instead of instructions to revisit
patients for treatment. Server rejections and completion remain authoritative.
Earlier regression checks replayed run 3964's saved all-unseen state; their
simulated server responses do not establish acceptance or completion in a fresh
live run.

If the agent exhausts its allowed attempts while multiple concrete evacuation
options remain, recovery stops with `ScenarioStalledError` instead of choosing
by patient or action order. Diagnostics include `reason: evacuation_choice_required`
and the unresolved `evacuation_candidates`. The episode stays incomplete, with
no final score request or advance to the next scenario. This rule depends on
offered actions, not scenario names, patient names, or transport capacity.
An explicit agent choice still executes normally; a single remaining evacuation
option can use the existing fallback and server-rejection handling.

Ordinary TA3 eval sessions skip the separate score API: this client calls it
only for `training_session=full`. That is recorded as `api.status=skipped`,
not as a failed server request. Changing to training mode is not a way to
retrieve an existing eval session's score. If the final state also lacks a
score, no launcher setting can manufacture one. `meta.json` now preserves the
server session ID, endpoint, session type, and training mode for investigation.
Retrieval exceptions are logged with a traceback and retained in diagnostics.

The OpenWorld profiles (`FEB_OPENWORLD3`, `APRIL_OPENWORLD3`) have no training
scenarios, so `training_session=full` does not apply to them: their scores come
only from live eval sessions whose TA3 server can reach TA1. Inspect the saved
score diagnostics with:

```bash
python -m json.tool /path/to/run/scoring_status.json
```

The event journal also records `alignment_scoring` outcomes, including when
score artifact saving is disabled. Aborted episodes do not trigger additional
score API requests during cleanup.

## Failure and completion journal

Every driver run creates `agent_events.jsonl` before model resolution. Each event
has a run ID, episode ID where applicable, sequence number, and timestamp. Writes
are flushed and synchronized to disk. The journal records episode and decision
starts, model attempts, tool requests/results,
and attempted/successful/rejected/failed environment actions. Validation messages
are retained as tool results. Exceptions identify their type. Actor responses
include token-usage metadata when the backend provides it; toolkit token usage
is not currently measured.

Aborted episodes are recorded even when no action succeeds. A graceful run finish
does not imply every scenario completed: action-limit exits remain unfinished.
A killed process can leave missing terminal events or a truncated final JSONL
line. There is no automatic resume. Reusing an output directory containing a
journal is rejected so runs cannot be mixed.

With `save_input_output=true`, an empty action trace exists even if model setup
fails. Successful traces are replaced atomically. Their `choice_info.event_log`
identifiers connect each action to its journal.
