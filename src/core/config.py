from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    # LLM (OpenAI-compatible: vLLM on the AMD box, or hosted for laptop dev)
    llm_base_url: str = "http://localhost:8001/v1"
    llm_model: str = "your-served-model-name"
    llm_api_key: str = "sk-local"
    llm_structured_method: Literal["json_schema", "function_calling", "json_mode"] = "json_schema"
    llm_temperature: float = 0.3

    # Mocks
    ml_mock: bool = False
    llm_mock: bool = False
    mock_delay_s: float = 0.2

    # Paths
    data_dir: Path = ROOT / "data"
    runs_dir: Path = ROOT / "runs"

    device: Literal["auto", "cuda", "mps", "cpu"] = "auto"

    # Budgets
    budget_profile: Literal["research", "demo"] = "research"
    min_budget_s: int = 60
    max_budget_s: int = 900
    demo_max_budget_s: int = 120
    full_train_multiplier: float = 5.0
    max_experiments: int = 20
    research_budget_s: int = 7200
    plateau_n: int = 6
    extra_after_pass: int = 0

    # Agent loop limits
    max_fix_attempts: int = 2
    max_design_attempts: int = 3
    max_tool_calls: int = 5
    results_context_rows: int = 20

    def clamp_budget(self, seconds: int) -> int:
        hi = self.demo_max_budget_s if self.budget_profile == "demo" else self.max_budget_s
        lo = min(self.min_budget_s, hi)
        return max(lo, min(hi, int(seconds)))


settings = Settings()
