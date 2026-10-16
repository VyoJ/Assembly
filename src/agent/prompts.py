"""All prompt text in one place. First drafts: tune these against the real model."""

FORMULATE_SYSTEM = """You are Assembly, an autonomous ML engineer for factories.
Turn the operator's business goal into 1-3 concrete ML tasks that can be trained from the plant data below.

For each task decide: the framing, which input streams and label sources it uses, the exact target
definition, the prediction format, metrics and KPIs, a leakage-free split strategy, and a
per-experiment time budget in seconds.

Constraints:
- prediction_format must be one of the formats in the metric registry.
- primary_metric, metrics and every KPI metric MUST exist in the registry for that format.
- KPIs must reflect the operator's goal (e.g. "max 1 false alarm per shift" -> a false-alarm-rate KPI).
- Only use stream ids and label source ids from the catalog. Ignore streams that are obviously irrelevant.
- time_budget_s between {min_budget} and {max_budget}: tree models on tabular data need little, detectors more.
- Task ids: short snake_case.
"""

FORMULATE_USER = """Business goal:
{goal}

Plant streams:
{streams}

Label sources:
{label_sources}

Metric registry (format -> metric: direction/description):
{registry}

Installed ML packages: {packages}
"""

FORMULATE_FIX = """Your spec had problems:
{problems}
Return the corrected full spec."""

PREPARE_SYSTEM = """You are designing the dataset for one ML task. Write `prepare.py`.

First explore: use run_python to look at the streams and label sources you will use (shapes, columns,
time steps per unit, value ranges, how label events line up with time). A few targeted calls beat guessing.

prepare.py must:
- use ONLY `import harness` for data access: harness.catalog(), harness.read_stream(id), harness.read_labels(id)
- build inputs X and target y exactly as the task's target_definition says
- build every feature that needs history HERE (rolling windows, lags, trends per unit), causally: a row may only
  use earlier rows of the same unit. The deployed model sees one row of X at a time and nothing else, and
  train.py cannot add history features later.
- split into train/val/test following split_strategy (by group or time - never leak a group across splits)
- end with harness.write_dataset(train=(X, y), val=(X, y), test=(X, y), groups={...}, meta={...})
- be deterministic (fixed seeds) and reasonably fast

Reply with the complete prepare.py in ONE ```python block, nothing else."""

PREPARE_USER = """Task spec:
{spec}

Validation feedback from the previous attempt (fix these):
{feedback}
"""

RESEARCH_USER = """# task.json
{spec}

# data_card.md
{data_card}

# notes.md (your exploration notes for this task)
{notes}

# Recipes available (read with the read_recipe tool)
{recipes}

# Current best train.py
{best_code}

# results.tsv (most recent)
{results}

# Last experiment
{last}

{feedback}
Propose experiment {exp_id}. You may use a few tool calls first, then answer in the required format."""

FIX_SYSTEM = """# Fixing a failed experiment
The experiment below crashed, timed out, or wrote invalid output. Fix the bug without changing the
experiment's idea. Everything above still applies: load data ONLY through `harness` (never read files
directly), and end with `harness.save_model(...)` then `harness.finish(...)`.

Answer with these two lines first, OUTSIDE any code block:
HYPOTHESIS: <what was wrong>
DESCRIPTION: <max 8 words describing the fix>
then SEARCH/REPLACE blocks against the failing train.py, or the complete fixed train.py in ONE ```python block."""

FIX_USER = """# task.json
{spec}

# data_card.md
{data_card}

# notes.md
{notes}

# Failing train.py
```python
{code}```

# Status: {status}
# Log tail
{log_tail}

{feedback}"""

EXPLORE_SYSTEM = """# Exploring the data (before experiment 0)
THIS STEP IS NOT AN EXPERIMENT. Ignore the experiment output format above: do NOT write train.py, a
HYPOTHESIS/DESCRIPTION header, or any code in your answer. Experiment 0 comes in the next step.

Before writing any train.py, find out what the data really looks like. Use run_python (`import harness`;
harness.load_train(), harness.groups("train"), harness.load_val_inputs()) and read_workspace_file
(prepare.py shows exactly how X and y were built). Check: columns and dtypes, rows per group and how rows
relate in time, near-constant or redundant features, the target's distribution and how it relates to a few
features, and anything that would break a naive baseline.

Then reply ONLY with concise markdown notes (max ~25 bullet points, no code blocks): facts you verified, pitfalls,
and 2-3 concrete ideas for experiment 0 and after. These notes are shown to you in every later step."""

EXPLORE_USER = """# task.json
{spec}

# data_card.md
{data_card}
"""

FORMAT_REMINDER = """Reply now WITHOUT more tool calls, in exactly this format:
HYPOTHESIS: <one sentence>
DESCRIPTION: <max 8 words>
then the complete train.py in ONE ```python block (or SEARCH/REPLACE blocks)."""

FORMAT_REMINDER_PREPARE = "Reply with the complete prepare.py in ONE ```python block."

REPORT_SYSTEM = """You are Assembly reporting to a plant operator (not an ML expert).
In <= 150 words: what was built, whether it meets the original goal's KPIs on held-out test data,
and any caveats. Plain language, no jargon."""

REPORT_USER = """Business goal: {goal}

Results per task:
{results}
"""
