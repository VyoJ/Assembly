import os
import tempfile

# Must run before anything imports core.config.
os.environ["ML_MOCK"] = "1"
os.environ["LLM_MOCK"] = "1"
os.environ["MOCK_DELAY_S"] = "0"
os.environ["RUNS_DIR"] = tempfile.mkdtemp(prefix="assembly-test-runs-")
