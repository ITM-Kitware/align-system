# Frozen classifier transfer to active defense

This comparison applies the previously selected P4 prompt with three Monitor
examples per class to recorded active hybrid PPO episodes. The prompt and example
contents stay frozen. It evaluates classification, not LLM-driven policy switching.

`active_defender_transfer.py prepare` verifies the existing Monitor and active
capture audits, hashes, common scenario and simulator, and paired episode seeds.
It excludes all four example seed groups. The remaining 96 episodes cover sixteen
seed groups and five PPO training checkpoints, with unequal checkpoint counts.

The primary reserved cohort contains 48 episodes from eight seed groups. It covers
four checkpoints. The broader cohort includes another eight groups previously used
for Monitor development; it covers all five checkpoints. Reports keep these scopes
separate and provide checkpoint and prediction-time breakdowns.

At five prediction times, each episode supplies three conditions: frozen P4 on
Monitor, frozen P4 on active defense, and the historical P0 prompt on active defense.
Both templates keep their exact system prompt and available-label formatting.
Output caps are uniformly 256 with greedy structured classification. Inputs contain
only Blue's selected actions and resulting tables, excluding Red actions, rewards,
hidden metrics, checkpoint IDs, episode filenames, and evaluation labels.
P4's demonstrations are Blue-only. P0 retains its historical Red-action
demonstrations for baseline continuity; no evaluation-episode Red information
is supplied to either condition.

The default comparison has 1,440 requests. `run_active_transfer.py` requires a
preflight token audit and a class-balanced pilot of twelve requests per arm.
Admission uses per-arm latency, a thirty percent margin, and cleanup reserve.
If the full broader comparison cannot fit, it falls back to the complete reserved
cohort, 720 requests, before any main accuracy is examined. It fails closed if
that smaller complete comparison cannot fit. It does not truncate inputs.

The Slurm entrypoint reserves one GPU in `community`, keeps Slurm's GPU assignment,
and enforces both a resource limit and a run-specific absolute timeout. A daytime
request must specify a new deadline rather than reuse an expired overnight cutoff.
CPU report/archive jobs can run after inference ends.

Reports include paired P4 active-minus-Monitor and P4-minus-P0 active accuracy,
with uncertainty grouped by shared rollout seed. Different Blue actions change
Red's subsequent behavior and evidence, so the paired conditions share starts
and attacker class, not necessarily identical attack traces.
