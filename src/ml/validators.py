"""Checks on prepare.py output. This is what makes agent-designed tasks trustworthy.

TODO(ML): implement. Return a ValidationReport with one entry per check and human-readable
messages the agent can act on (they're fed back to the LLM to fix prepare.py).
"""

from core.contracts import ValidationReport, Workspace

CHECKS = [
    "files_present",  # data/ has train inputs+labels, val/test inputs; hidden dir has val/test labels
    "format",  # label shape/type matches spec.prediction_format
    "group_disjoint",  # no group (engine unit, image id...) appears in more than one split
    "time_ordered",  # if split_strategy is by time: max(train time) < min(val time) < min(test time)
    "min_sizes",  # each split above a minimum size
    "label_sanity",  # val/test not single-class; enough positives for binary/boxes
    "no_label_leak",  # no input column identical/near-identical to the label
]


def validate(ws: Workspace) -> ValidationReport:
    """TODO(ML): run CHECKS, build a data card (shapes, columns, label stats, split sizes) for data_card.md."""
    raise NotImplementedError("TODO(ML): validators.validate")
