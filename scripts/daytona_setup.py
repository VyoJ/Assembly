"""Build (once) the Daytona snapshot with the pinned ML stack, then smoke-test a sandbox from it.

    uv run python scripts/daytona_setup.py            # build if missing + smoke test
    uv run python scripts/daytona_setup.py --list     # list Assembly sandboxes (e.g. to clean up leftovers)
    uv run python scripts/daytona_setup.py --cleanup  # delete all Assembly sandboxes
"""

import argparse
import time

from ml.executors.daytona import DaytonaExecutor, snapshot_name
from ml.executors.remote import ML_PACKAGES, TORCH_PACKAGES, pinned


def main(args) -> None:
    ex = DaytonaExecutor()
    if args.list or args.cleanup:
        boxes = [s for s in ex.client.list() if (getattr(s, "labels", None) or {}).get("app") == "assembly"]
        for s in boxes:
            print(s.id, getattr(s, "state", "?"), s.labels)
            if args.cleanup:
                s.delete()
        print(f"{len(boxes)} sandbox(es){' deleted' if args.cleanup else ''}")
        return

    print(f"snapshot: {snapshot_name()}")
    print("torch:", pinned(TORCH_PACKAGES))
    print("packages:", len(pinned(ML_PACKAGES)))
    t0 = time.monotonic()
    ex.ensure_snapshot(on_logs=lambda line: print("  build>", line.rstrip()))
    print(f"snapshot ready ({time.monotonic() - t0:.0f}s)")

    from daytona import CreateSandboxFromSnapshotParams

    t0 = time.monotonic()
    box = ex.client.create(CreateSandboxFromSnapshotParams(
        snapshot=ex.ensure_snapshot(), labels={"app": "assembly", "run": "smoke"}, auto_stop_interval=10,
        auto_delete_interval=0, network_block_all=True), timeout=300)
    print(f"sandbox {box.id} up ({time.monotonic() - t0:.0f}s), home={box.get_user_home_dir()}")
    try:
        check = ("import sys, torch, sklearn, xgboost, lightgbm, pandas, pyarrow, pydantic_settings; "
                 "print(sys.version.split()[0], 'torch', torch.__version__, 'cuda', torch.cuda.is_available(), "
                 "'xgb', xgboost.__version__, 'lgb', lightgbm.__version__, 'sklearn', sklearn.__version__)")
        r = box.process.exec(f"python -c \"{check}\"", timeout=120)
        print("versions:", r.exit_code, r.result.strip())
        r = box.process.exec("nproc; cat /sys/fs/cgroup/cpu.max 2>/dev/null; cat /sys/fs/cgroup/memory.max 2>/dev/null; df -h ~ | tail -1", timeout=60)
        print("resources:\n", r.result.strip())
        r = box.process.exec("python -c \"import urllib.request; urllib.request.urlopen('https://pypi.org', timeout=5)\"",
                             timeout=60)
        print("network blocked:", r.exit_code != 0)
    finally:
        box.delete()
        print("sandbox deleted")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--list", action="store_true")
    p.add_argument("--cleanup", action="store_true")
    main(p.parse_args())
