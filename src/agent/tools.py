"""Tools the LLM may call (bounded by settings.max_tool_calls per step). Reading tools are read-only;
run_python executes a short script in the same sandbox as train.py (never sees hidden labels)."""

import json
from pathlib import Path

from langchain_core.tools import BaseTool, StructuredTool

from core.contracts import Workspace
from ml import api as ml_api

RECIPES_DIR = Path(ml_api.__file__).resolve().parent / "recipes"
READABLE = ("prepare.py", "harness.py", "task.json", "data_card.md", "notes.md", "plan.md", "data/meta.json")


def read_recipe(name: str) -> str:
    """Read a reference script or INDEX.md from the recipes folder, e.g. 'xgb_baseline.py'."""
    p = (RECIPES_DIR / name).resolve()
    if not p.is_relative_to(RECIPES_DIR) or not p.is_file():
        return f"(no recipe '{name}'; available: {sorted(f.name for f in RECIPES_DIR.iterdir())})"
    return p.read_text()


def list_catalog() -> str:
    """List all plant streams and label sources (ids, modality, description)."""
    streams, labels = ml_api.list_catalog()
    lines = [f"stream {s.id} [{s.modality}] {s.name}" for s in streams]
    lines += [f"label_source {lab.id} [{lab.kind}] {lab.description}" for lab in labels]
    return "\n".join(lines)


def peek_stream(stream_id: str, n: int = 5) -> str:
    """Show the first n rows/items of a stream."""
    return ml_api.peek_stream(stream_id, n)


def _read_workspace_tool(ws: Workspace) -> StructuredTool:
    def read_workspace_file(name: str) -> str:
        """Read a workspace file: prepare.py (how X and y were built), harness.py (the data / submit API),
        task.json, data_card.md, notes.md (your exploration notes), plan.md (your research plan) or data/meta.json."""
        if name not in READABLE:
            return f"(not readable: {name}; choose one of {list(READABLE)})"
        text = ml_api.read_workspace_file(ws, name)
        if name == "data/meta.json":  # per-sample group lists are huge; summarise them
            try:
                meta = json.loads(text)
                meta["groups"] = {k: f"<{len(v)} ids, {len(set(map(str, v)))} distinct>"
                                  for k, v in (meta.get("groups") or {}).items()}
                text = json.dumps(meta, indent=2)
            except ValueError:
                pass
        return text

    return StructuredTool.from_function(read_workspace_file)


def _run_python_tool(ws: Workspace, phase: str) -> StructuredTool:
    def run_python(code: str) -> str:
        """Run a short Python script (60 s limit) in the workspace and return what it prints. `import harness`
        works: in research use harness.load_train(), harness.groups('train'), harness.load_val_inputs();
        in design use harness.catalog(), harness.read_stream(id), harness.read_labels(id).
        Print only what you need (shapes, dtypes, head(), describe(), value_counts()). Output is capped."""
        return ml_api.run_snippet(ws, code, phase)

    return StructuredTool.from_function(run_python)


def peek_labels(label_source_id: str, n: int = 5) -> str:
    """Show a label source's columns, dtypes, first n rows and the values of its categorical columns."""
    return ml_api.peek_labels(label_source_id, n)


def design_tools(ws: Workspace) -> list[BaseTool]:
    def read_harness() -> str:
        """Read harness.py: the exact API prepare.py uses (catalog, read_stream(s), read_labels, write_dataset)."""
        return ml_api.read_workspace_file(ws, "harness.py")

    return [*(StructuredTool.from_function(f) for f in (list_catalog, peek_stream, peek_labels, read_recipe,
                                                         read_harness)),
            _run_python_tool(ws, "design")]


def research_tools(ws: Workspace, read_only: bool = False) -> list[BaseTool]:
    def read_experiment_file(exp_id: str, filename: str) -> str:
        """Read a file from a past experiment, e.g. ('exp_003', 'train.py') or ('exp_003', 'run.log')."""
        return ml_api.read_workspace_file(ws, f"{exp_id}/{filename}")

    tools = [StructuredTool.from_function(read_recipe), StructuredTool.from_function(read_experiment_file),
             _read_workspace_tool(ws)]
    return tools if read_only else [*tools, _run_python_tool(ws, "research")]
