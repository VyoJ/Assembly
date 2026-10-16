import asyncio

from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool

from agent import llm
from agent.research import graph
from core.config import settings
from core.contracts import KPI, TaskSpec
from ml import workspace


def _ws(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "runs_dir", tmp_path / "runs")
    spec = TaskSpec(id="t", name="t", rationale="", framing="", input_streams=[], label_sources=[],
                    target_definition="", prediction_format="scalar", primary_metric="rmse", metrics=["rmse"],
                    kpis=[KPI(metric="rmse", op="<=", threshold=1.0)], split_strategy="by unit", time_budget_s=60)
    return workspace.create_workspace("r", spec)


def test_duplicate_only_for_same_code(tmp_path, monkeypatch):
    ws = _ws(tmp_path, monkeypatch)
    workspace.new_exp_dir(ws, "import harness\nLR = 0.1  # baseline\n\nharness.finish(f)\n")
    assert graph._duplicate_of(ws, "import harness\nLR = 0.1\nharness.finish(f)  # same\n") == "exp_000"
    assert graph._duplicate_of(ws, "import harness\nLR = 0.05\nharness.finish(f)\n") is None


def test_clean_notes_drops_code_and_headers():
    raw = "HYPOTHESIS: x\nDESCRIPTION: y\n- y is capped at 125\n```python\nimport harness\n```\n- 70 units"
    assert graph._clean_notes(raw) == "- y is capped at 125\n\n- 70 units"
    assert graph._clean_notes("```python\nprint(1)\n```") == ""


class _LoopingModel:
    """Fake chat model that asks for run_python forever, then answers once tools are exhausted."""

    def __init__(self):
        self.calls = 0

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, msgs):
        self.calls += 1
        if "used all tool calls" in str(msgs[-1].content):
            return AIMessage("final answer")
        return AIMessage("", tool_calls=[{"name": "run_python", "args": {"code": "print(1)"}, "id": f"c{self.calls}"}])


def test_capped_tool_cannot_loop_forever(monkeypatch):
    model = _LoopingModel()
    monkeypatch.setattr(settings, "llm_mock", False)
    monkeypatch.setattr(llm, "get_llm", lambda: model)
    ran = []
    tool = StructuredTool.from_function(lambda code: ran.append(code) or "ok", name="run_python", description="run")
    traces = []
    out = asyncio.run(llm.text(llm.messages("s", "u"), mock=lambda: "", tools=[tool], max_tool_calls=4,
                               tool_limits={"run_python": 1}, trace=traces.append))
    assert out == "final answer"
    assert len(ran) == 1  # the cap held
    assert model.calls == 5  # 4 budgeted turns (1 run + 3 blocked) + the final answer
    assert traces[0]["reply"] == "final answer" and len(traces[0]["tool_calls"]) == 1


def test_split_notes_and_plan():
    raw = "## Notes\n- y capped at 125\n\n## Plan\n1. [ ] LightGBM baseline - fast\n2. [ ] average two models"
    notes, plan = graph._split_notes_plan(raw)
    assert notes == "- y capped at 125"
    assert plan.startswith("1. [ ] LightGBM") and "## Plan" not in plan
    assert graph._split_notes_plan("- just notes") == ("- just notes", "")


def test_clean_reply_detects_fake_tool_calls_and_think():
    text, fake = llm.clean_reply("Let me look first.</think>\n<｜DSML｜tool_calls><｜DSML｜invoke name=\"x\">")
    assert fake and text.startswith("<｜DSML｜tool_calls>")
    assert llm.clean_reply("reasoning...</think>\n1. [ ] plan item") == ("1. [ ] plan item", False)


def test_fix_history_shows_earlier_attempt_outcomes(tmp_path, monkeypatch):
    ws = _ws(tmp_path, monkeypatch)
    d = workspace.ws_dir(ws) / "exp_003"
    d.mkdir()
    (d / "result.json").write_text('{"log_tail": "ValueError: Must have at least 1 validation dataset"}')
    history = [{"of": "exp_002", "runs_as": "exp_003", "description": "drop callback", "diff": "-a\n+b\n"},
               {"of": "exp_003", "runs_as": "exp_004", "description": "drop early stopping", "diff": "-b\n+c\n"}]
    text = graph._render_fix_history(ws, history, current="exp_004")
    assert "Attempt 1: exp_002 -> exp_003: drop callback" in text and "Must have at least 1" in text
    assert "Attempt 2" in text and "the failure shown above" in text
    assert graph._render_fix_history(ws, [], "exp_001").startswith("(none")
