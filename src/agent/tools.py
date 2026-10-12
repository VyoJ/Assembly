"""Read-only tools the LLM may call (bounded by settings.max_tool_calls per step)."""

from pathlib import Path

from langchain_core.tools import BaseTool, StructuredTool

from core.contracts import Workspace
from ml import api as ml_api

RECIPES_DIR = Path(ml_api.__file__).resolve().parent / "recipes"


def read_recipe(name: str) -> str:
    """Read a reference script or INDEX.md from the recipes folder, e.g. 'xgb_window_features.py'."""
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


def design_tools() -> list[BaseTool]:
    return [StructuredTool.from_function(f) for f in (list_catalog, peek_stream, read_recipe)]


def research_tools(ws: Workspace) -> list[BaseTool]:
    def read_experiment_file(exp_id: str, filename: str) -> str:
        """Read a file from a past experiment, e.g. ('exp_003', 'train.py') or ('exp_003', 'run.log')."""
        return ml_api.read_workspace_file(ws, f"{exp_id}/{filename}")

    return [StructuredTool.from_function(read_recipe), StructuredTool.from_function(read_experiment_file)]
