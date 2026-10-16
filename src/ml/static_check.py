"""Pre-run static check of agent code against the INSTALLED library versions (no execution of the code).

Catches the commonest crash class from LLM-written code: keyword arguments that don't exist in the installed
version (e.g. `fit(early_stopping_rounds=...)` on XGBoost >= 2). For every call whose target can be resolved from
the file's imports (`xgb.XGBRegressor(...)`, `lgb.early_stopping(...)`, `from sklearn.x import Y; Y(...)`), and
for method calls on variables assigned from such a constructor (`m = xgb.XGBRegressor(); m.fit(...)`), each
keyword is checked against inspect.signature. Callables taking **kwargs are skipped. Plus ruff's undefined-name
check when ruff is installed.

Run as a subprocess (imports the libraries the code uses):  python -m ml.static_check < train.py
Prints a JSON list of problems.
"""

import ast
import importlib
import inspect
import json
import subprocess
import sys


def _imports(tree: ast.Module) -> dict[str, str]:
    """local name -> dotted path, e.g. {'xgb': 'xgboost', 'Pipeline': 'sklearn.pipeline.Pipeline'}"""
    names = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                names[a.asname or a.name.split(".")[0]] = a.name if a.asname else a.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            for a in node.names:
                names[a.asname or a.name] = f"{node.module}.{a.name}"
    return names


def _dotted(node: ast.AST) -> str | None:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        return ".".join([node.id, *reversed(parts)])
    return None


def _resolve(path: str):
    """'xgboost.callback.EarlyStopping' -> the object, importing the longest importable module prefix."""
    parts = path.split(".")
    for i in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:i]))
        except Exception:  # noqa: BLE001, S112 - not a module at this depth
            continue
        try:
            for p in parts[i:]:
                obj = getattr(obj, p)
        except AttributeError:
            return None
        return obj
    return None


def _params(obj) -> tuple[set[str], bool] | None:
    """(keyword names, accepts **kwargs) or None if the signature can't be read."""
    target = obj.__init__ if inspect.isclass(obj) else obj
    try:
        sig = inspect.signature(target)
    except (TypeError, ValueError):
        return None
    var_kw = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
    names = {n for n, p in sig.parameters.items() if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
    return names, var_kw


def check_kwargs(code: str) -> list[str]:
    tree = ast.parse(code)
    imports = _imports(tree)
    var_types: dict[str, object] = {}  # variable -> class it was constructed from
    problems = []

    def full_path(dotted: str) -> str | None:
        head, _, rest = dotted.partition(".")
        if head not in imports:
            return None
        return imports[head] + (f".{rest}" if rest else "")

    # Pass 1: `x = <resolvable class>(...)` so later `x.method(...)` can be checked
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            dotted = _dotted(node.value.func)
            path = full_path(dotted) if dotted else None
            obj = _resolve(path) if path else None
            if inspect.isclass(obj):
                var_types[node.targets[0].id] = obj

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        dotted = _dotted(node.func)
        if not dotted:
            continue
        head, _, attr = dotted.partition(".")
        if head in var_types and attr and "." not in attr:
            obj, label = getattr(var_types[head], attr, None), f"{var_types[head].__name__}.{attr}"
        else:
            path = full_path(dotted)
            obj, label = (_resolve(path), dotted) if path else (None, dotted)
        if obj is None or not callable(obj):
            continue
        info = _params(obj)
        if info is None or info[1]:
            continue
        valid, _ = info
        for kw in node.keywords:
            if kw.arg is not None and kw.arg not in valid:
                problems.append(f"line {node.lineno}: {label}() has no argument '{kw.arg}' in the installed version. "
                                f"Valid: {', '.join(sorted(v for v in valid if v != 'self'))}")
    return problems


def check_undefined_names(code: str) -> list[str]:
    try:
        out = subprocess.run(
            [sys.executable, "-m", "ruff", "check", "--select", "F821,F822,F823", "--output-format", "concise",
             "--no-cache", "--stdin-filename", "train.py", "-"],
            input=code, capture_output=True, text=True, timeout=20, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if out.returncode not in (0, 1) or "No module named ruff" in out.stderr:
        return []
    return [ln.split(":", 1)[1].strip() for ln in out.stdout.splitlines()
            if ln.startswith("train.py:") and "invalid-syntax" not in ln]  # compile() reports syntax errors


def main() -> None:
    code = sys.stdin.read()
    try:
        problems = check_kwargs(code)
    except SyntaxError:
        problems = []  # compile() reports syntax errors separately
    problems += check_undefined_names(code)
    print(json.dumps(problems))


if __name__ == "__main__":
    main()
