import pytest

from agent.research.edits import EditError, apply_edits, parse_proposal
from core.contracts import KPI, TaskSpec
from ml import scoring

BASE = "import harness\n\nLR = 0.1\nWINDOW = 30\n"


def test_parse_search_replace():
    p = parse_proposal("HYPOTHESIS: smaller lr\nDESCRIPTION: lr 0.05\n<<<<<<< SEARCH\nLR = 0.1\n=======\nLR = 0.05\n>>>>>>> REPLACE")
    assert p.description == "lr 0.05"
    assert apply_edits(BASE, p.edits) == BASE.replace("0.1", "0.05")


def test_parse_full_rewrite():
    p = parse_proposal("HYPOTHESIS: x\nDESCRIPTION: y\n```python\nprint('hi')\n```")
    assert apply_edits(BASE, p.edits) == "print('hi')\n"


def test_search_whitespace_tolerant_and_missing():
    assert "WINDOW = 50" in apply_edits(BASE, [{"old": "WINDOW = 30   ", "new": "WINDOW = 50"}])
    with pytest.raises(EditError):
        apply_edits(BASE, [{"old": "NOT THERE", "new": "x"}])
    with pytest.raises(EditError):
        parse_proposal("no code here")


def _spec(**kw) -> TaskSpec:
    base = {
        "id": "t", "name": "t", "rationale": "", "framing": "", "input_streams": [], "label_sources": [],
        "target_definition": "", "prediction_format": "binary_score", "primary_metric": "pr_auc",
        "metrics": ["pr_auc"], "split_strategy": "", "time_budget_s": 60,
        "kpis": [KPI(metric="recall", op=">=", threshold=0.9), KPI(metric="false_alarm_rate", op="<=", threshold=0.05)],
    }
    return TaskSpec(**{**base, **kw})


def test_score_meeting_kpis_always_wins():
    spec = _spec()
    passing = {"recall": 0.91, "false_alarm_rate": 0.04, "pr_auc": 0.5}
    failing_but_better_primary = {"recall": 0.89, "false_alarm_rate": 0.04, "pr_auc": 0.99}
    assert scoring.compute_score(passing, spec) < scoring.compute_score(failing_but_better_primary, spec)
    assert scoring.kpi_status(passing, spec.kpis) == {"recall >= 0.9": True, "false_alarm_rate <= 0.05": True}


def test_validate_spec_metrics():
    assert scoring.validate_spec_metrics(_spec()) == []
    assert scoring.validate_spec_metrics(_spec(primary_metric="rmse"))  # rmse isn't a binary_score metric
