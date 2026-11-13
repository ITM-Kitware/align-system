# Fresh episodes and legal-event history comparisons

These workflows extend the Blue prompt screen while keeping its executed source
and frozen selection separate. Both GPU jobs require an absolute cutoff and an
allocation time budget. CPU capture, reporting, and evidence collection can run
after the GPU deadline.

`capture_fresh_monitor.py` collects fixed-horizon Monitor episodes across the six
attacker classes and a declared range of shared seeds. Every episode is collected
twice in the same Python process and compared exactly before retention. It saves
only Blue actions, tables, and raw Blue observations. The summary records seed
groups, source and scenario fingerprints, environment versions, and hashes.
It makes no historical-reproduction claim. Seed disjointness is checked against
the supplied example/development/confirmation protocol; it does not establish
exclusion from every historical training run.

`prepare_overnight_followups.py raw` prepares a two-by-two experiment with table
events or tables plus raw legal events, each using seven steps or the whole
observed prefix. Both representations use the same example episode identities,
one per class. Seven-step examples are fixed across the history factor, so only
query history changes. Prediction times and rollout-seed partitions follow the
original screen. Raw fields are drawn exclusively from recorded Blue observations.

The raw representation preserves connection order and repetitions, additional
fields, and a presence mask for missing keys. Known addresses may be abbreviated
using the visible table. Host attribute changes and attribute presence are
recorded separately. The renderer rejects unreviewed host fields and never
supplies a true Red action or Red action outcome. Round-trip checks verify the
retained host evidence, including repeated and partially specified connections.

`prepare_overnight_followups.py fresh` reads the original development selection,
freezes its first candidate, and preserves that candidate's exact system prompt,
greedy sampling, and output schema. Every new condition uses a uniform 256-token
output cap instead of the historical mixture of 32 and 256. It pairs this prompt with the historical Blue baseline
on new captures, at the same prediction times. The fresh labels and episode
identities remain outside model messages.

`run_overnight_followups.py raw` uses a large-context, class-balanced timing pilot,
then evaluates 96 development requests per condition. It confirms the strongest
development condition on 240 reserved requests. Separately, it freezes the best
raw condition and its table control at the same history length for a secondary
fresh-seed comparison. Neither reserved nor fresh scores enter either selection.

`run_overnight_followups.py fresh` compares the original screen winner against the
historical Blue baseline. Its timing pilot uses only old development episodes,
before fresh scoring. The two workflows may select different fresh sample sizes
because their prompts have different costs; both retain their exact selections.

Each fresh block starts with up to 100 available shared seed groups. From pilot
latency and its remaining allocation, the budget chooses between 50 and 100
complete groups, with five prediction times where feasible or three otherwise.
Class balance, paired conditions, and entire seed groups are retained. A 30 percent
runtime margin and cleanup allowance are included. If fewer than fifty groups
and three times can fit, the block is not admitted. No accuracy is used to size
the sample. Requests exceeding context capacity fail without truncation.

`overnight_followups.slurm` requires explicit source, Python, model, cache,
temporary, source-screen, and owned result locations. It preserves Slurm's GPU
assignment, runs one GPU in the community partition, and places the workflow in
an absolute process-group timeout. Submit a scheduler deadline and a compatible
allocation time limit as independent resource controls. Serialize each lane and
retain scheduler job IDs; terminal evidence jobs require CPU only.

`report_overnight_followups.py` reports class confusion, joint/family/objective
accuracy, prediction times, and paired comparisons. Intervals resample shared
rollout-seed groups across classes and times. Fresh provenance is distinguished
from exploratory historical confirmation. Partial conditions remain partial.
