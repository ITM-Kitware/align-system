# Blue observation prompt comparisons

This workflow compares defender-specific instructions, compact chronological
tables, one, two, or three examples per class, and longer histories. It reuses verified
Monitor captures and completed matched-baseline predictions. It never supplies
Red actions, hidden metric scores, episode filenames, or true evaluation labels
in model messages.

Run `scripts/prepare_blue_prompt_screen.py` with the matched Blue request file,
its completed prediction file, the verified Monitor capture directory, and a
new output directory. The preparer requires twenty shared rollout seeds across
six classes. It partitions those seeds into four example, eight development,
and eight reserved confirmation seeds, keeping classes from the same rollout
seed together. The confirmation episodes were included in earlier aggregate
baselines, so this is exploratory confirmation rather than an unseen final test.

The prepared variants are:

| ID | Instructions and representation | Examples | Observed steps |
| --- | --- | ---: | ---: |
| P1 | Blue-specific instructions and full tables | 0 | 7 |
| P2 | Blue-specific instructions and initial rows plus chronological changes | 0 | 7 |
| P3 | Same as P2 | 1 per class | 7 |
| P4 | Same as P2 | 3 per class | 7 |
| P5 | Same as P3, with additional past history | 1 per class | Up to 21 |
| P6 | Same as P2 | 2 per class | 7 |

Compact histories preserve every initial row, subsequent changes, each current
non-None activity including repetitions, and every Blue action and result. P5
ends at the same prediction time; an early window with only seventeen available
steps uses those seventeen and records its actual length. Examples are nested,
class balanced, and selected from distinct example seeds without consulting model
accuracy. P0 reuses unchanged-prompt predictions on the corresponding windows.

Use `scripts/run_blue_prompt_screen.py --preflight-only` with the prepared input
directory, a separate output directory, the cached model, and a context limit.
It tokenizes every development and confirmation request with thinking disabled.
An over-context optional variant is excluded entirely; evidence is not truncated.
Core variants P1, P2, and P3 must fit. The default runtime cap is 32,768 tokens;
this is a configurable memory budget, not the model's declared context capacity.

The inference workflow requires a Slurm allocation and `--stop-at` as an absolute
epoch timestamp. Its twenty-four-request pilot contains four requests per class
chosen for large contexts. Only pilot latency determines the budget. The recorded
budget can omit P4, then P6, then P5, reduce confirmation to one candidate, and finally
reduce development to one balanced window per episode. If the smallest complete
protocol cannot fit, inference stops before development rather than scheduling
an incomplete class-ordered comparison.

The workflow evaluates eligible variants, freezes the strongest one or two using
development joint accuracy with shorter mean input length breaking ties, then
evaluates the frozen candidates on reserved episodes. Raw inputs, decisions,
predictions, and reports remain in separate files. A changed or repeated workflow
attempt requires a new output directory; retained selected input files can also
be replayed explicitly with the existing runner and its resume checks.

Use `scripts/blue_prompt_screen.slurm` with explicit immutable source, existing
inference Python, data-drive caches and temporary directories, and absolute cutoff
environment variables. Reserve one GPU through the community partition and set
any dependency and deadline at submission. The wrapper enforces the absolute
cutoff with a cleanup margin and preserves Slurm's GPU assignment.

`scripts/blue_prompt_screen_report.py` produces JSON and Markdown reports without
model dependencies. It reports joint, family, and objective accuracy, macro F1,
confusion matrices, output bias, latency, and position-specific results. Paired
uncertainty resamples shared rollout seeds across all classes. Partial conditions
remain marked partial. CPU report jobs can run after inference termination.
