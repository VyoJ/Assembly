"""Parse the LLM's experiment proposals and apply them to train.py.

Format (see ml/program.md): HYPOTHESIS / DESCRIPTION lines, then either a full ```python block
or SEARCH/REPLACE blocks against the current best train.py.
"""

import re

from core.contracts import Proposal

_BLOCK = re.compile(r"<{5,9} SEARCH\n(.*?)\n?={5,9}\n(.*?)\n?>{5,9} REPLACE", re.DOTALL)
_CODE = re.compile(r"```(?:python|py)?\n(.*?)```", re.DOTALL)


class EditError(ValueError):
    pass


def _field(text: str, name: str) -> str:
    """'HYPOTHESIS: x', '**Hypothesis:** x', or a '## Hypothesis' heading followed by the text on the next line."""
    m = re.search(rf"^\s*[#>*\s]*{name}[*\s]*[:\-][*\s]*(.+)$", text, re.MULTILINE | re.IGNORECASE)
    if m and m.group(1).strip(" *"):
        return m.group(1).strip(" *")
    m = re.search(rf"^\s*#+\s*{name}\s*\n+\s*(\S.*)$", text, re.MULTILINE | re.IGNORECASE)
    return m.group(1).strip(" *") if m else ""


def _short(text: str, words: int = 8) -> str:
    parts = text.split()
    return " ".join(parts[:words]) + ("..." if len(parts) > words else "")


_HEADER = re.compile(r"^\s*[#>*\s]*(HYPOTHESIS|DESCRIPTION)[*\s]*[:\-]", re.IGNORECASE)


def _strip_header(code: str) -> str:
    """Drop HYPOTHESIS / DESCRIPTION lines a model put inside the code block instead of above it."""
    lines = code.splitlines()
    while lines and (not lines[0].strip() or _HEADER.match(lines[0])):
        lines.pop(0)
    return "\n".join(lines)


def extract_code(text: str) -> str | None:
    blocks = _CODE.findall(text)
    return _strip_header(max(blocks, key=len)).strip() + "\n" if blocks else None


def parse_proposal(text: str) -> Proposal:
    hypothesis, description = _field(text, "HYPOTHESIS"), _field(text, "DESCRIPTION")
    edits = [{"old": old, "new": new} for old, new in _BLOCK.findall(text)]
    if not edits:
        code = extract_code(text)
        if code is None:
            raise EditError("No ```python block and no SEARCH/REPLACE blocks found.")
        edits = [{"rewrite": code}]
    description = description or (_short(hypothesis) if hypothesis else "(no description)")
    return Proposal(hypothesis=hypothesis or "(none given)", description=description, edits=edits)


def check_train_py(code: str) -> None:
    """Cheap static checks before spending an experiment on code that can't work."""
    try:
        compile(code, "train.py", "exec")
    except SyntaxError as e:
        raise EditError(f"train.py does not compile: {e.msg} (line {e.lineno}): {(e.text or '').strip()}") from e
    if "harness.finish(" not in code:
        raise EditError("train.py never calls harness.finish(predict_fn): no predictions would be written.")


def _norm(s: str) -> str:
    return "\n".join(line.rstrip() for line in s.strip("\n").splitlines())


def apply_edits(base: str | None, edits: list[dict]) -> str:
    code = base or ""
    for e in edits:
        if "rewrite" in e:
            code = e["rewrite"]
            continue
        old, new = e["old"], e["new"]
        if old and old in code:
            code = code.replace(old, new, 1)
            continue
        # whitespace-tolerant fallback: compare with trailing spaces stripped
        lines, old_lines = code.splitlines(), _norm(old).splitlines()
        for i in range(len(lines) - len(old_lines) + 1):
            if [ln.rstrip() for ln in lines[i : i + len(old_lines)]] == old_lines:
                lines[i : i + len(old_lines)] = new.strip("\n").splitlines()
                code = "\n".join(lines) + "\n"
                break
        else:
            raise EditError(f"SEARCH block not found in train.py:\n{old[:400]}")
    if not code.strip():
        raise EditError("Resulting train.py is empty.")
    return code
