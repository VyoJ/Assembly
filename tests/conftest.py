import os
import sys
import tempfile

# Must run before anything imports core.config.
os.environ["ML_MOCK"] = "1"
os.environ["LLM_MOCK"] = "1"
os.environ["MOCK_DELAY_S"] = "0"
os.environ["RUNS_DIR"] = tempfile.mkdtemp(prefix="assembly-test-runs-")
# macOS dev machines only: xgboost (Homebrew libomp) and torch (its own libomp) in ONE process deadlock or segfault
# unless OpenMP is single-threaded. Linux wheels don't clash; the server and Daytona boxes run Linux.
if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    # ...and torch must load before xgboost (the reverse order segfaults); same rule as ml.package for deployments
    try:
        import torch  # noqa: F401
    except ImportError:
        pass
