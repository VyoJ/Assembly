"""Stream discovery: which streams are informative for a task.

TODO(ML): implement with real data (catalog.read_stream / read_labels).
"""

from core.contracts import StreamInfo, StreamProfile, TaskSpec


def profile_streams(spec: TaskSpec, streams: list[StreamInfo]) -> list[StreamProfile]:
    """TODO(ML): per stream, compute and explain relevance to the task's target, e.g.
    - constant / near-constant (std ~ 0) -> keep=False, "constant signal"
    - |Spearman| or mutual information vs. a proxy target (e.g. cycles-to-failure from maintenance_log)
    - image streams: keep if a label source annotates them
    """
    raise NotImplementedError("TODO(ML): profiling.profile_streams")
