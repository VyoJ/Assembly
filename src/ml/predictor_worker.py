"""Worker process for ml.package.IsolatedPredictor: loads ONE package and answers predictor calls.

Started as `python -m ml.predictor_worker <package> <port> <authkey hex>` (never via multiprocessing spawn, which
would re-run the parent's main module and its imports here, defeating the isolation).
"""

import sys
from multiprocessing.connection import Client
from pathlib import Path


def main() -> None:
    package_path, port, authkey = sys.argv[1], int(sys.argv[2]), bytes.fromhex(sys.argv[3])
    conn = Client(("127.0.0.1", port), authkey=authkey)
    from ml.package import PROXY_ATTRS, Predictor

    try:
        predictor = Predictor(package_path)
    except Exception as e:  # noqa: BLE001 - report load failures to the parent
        conn.send((False, f"{type(e).__name__}: {e}"))
        return
    attrs = {k: getattr(predictor, k, None) for k in PROXY_ATTRS}
    conn.send((True, {k: str(v) if isinstance(v, Path) else v for k, v in attrs.items()}))
    while True:
        try:
            name, args = conn.recv()
        except (EOFError, OSError):
            return
        try:
            conn.send((True, getattr(predictor, name)(*args)))
        except Exception as e:  # noqa: BLE001 - errors go back to the caller
            conn.send((False, f"{type(e).__name__}: {e}"))


if __name__ == "__main__":
    main()
