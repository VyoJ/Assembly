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
    m = re.search(rf"^\s*\**{name}\**\s*:\s*(.+)$", text, re.MULTILINE | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def extract_code(text: str) -> str | None:
    blocks = _CODE.findall(text)
    return max(blocks, key=len).strip() + "\n" if blocks else None


def parse_proposal(text: str) -> Proposal:
    hypothesis, description = _field(text, "HYPOTHESIS"), _field(text, "DESCRIPTION")
    edits = [{"old": old, "new": new} for old, new in _BLOCK.findall(text)]
    if not edits:
        code = extract_code(text)
        if code is None:
            raise EditError("No ```python block and no SEARCH/REPLACE blocks found.")
        edits = [{"rewrite": code}]
    return Proposal(hypothesis=hypothesis or "(none given)", description=description or "(no description)", edits=edits)


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
