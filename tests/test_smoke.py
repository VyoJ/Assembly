"""End-to-end in mock mode: goal -> specs -> design -> research -> deploy."""

from agent.graph import run_pipeline
from core import registry
from core.events import bus


async def test_pipeline_mock_end_to_end():
    final = await run_pipeline("Prevent unplanned stoppages on Press Line 3", run_id="test-run", auto_approve=True)

    assert final["status"] == "done"
    assert set(final["best"]) == {t.id for t in final["spec"].tasks}
    assert final["deployment"].id in registry.deployments
    types = [e["type"] for e in bus.history("test-run")]
    for expected in ("task_specs", "discovery", "validation", "experiment", "research_done", "final_test", "deployed"):
        assert expected in types, expected
