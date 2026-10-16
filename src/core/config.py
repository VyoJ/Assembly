from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    # LLM (OpenAI-compatible: AMD Token Factory by default, see docs/Token-Factory.md)
    llm_base_url: str = "https://developer.amd.com.cn/radeon/api/v1"
    llm_model: str = "DeepSeek-V4-Flash"
    llm_api_key: str = "rc-missing"
    # Token Factory: only MiMo accepts json_schema; every chat model supports tool calling
    llm_structured_method: Literal["json_schema", "function_calling", "json_mode"] = "function_calling"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 16384  # thinking tokens count toward this; too low -> empty content
    llm_reasoning_effort: str | None = None  # e.g. "low"; omitted = model default (DeepSeek: no thinking)
    llm_max_retries: int = 6  # 429 / 5xx retried with backoff (honours Retry-After)
    llm_timeout_s: float = 600.0
    llm_max_concurrency: int = 3  # the team shares one key: 8 concurrent, ~20 RPM

    # Comma-separated stream / label-source ids hidden from the agent (e.g. data with no working ML path yet)
    catalog_exclude: str = ""

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
    max_repairs: int = 2  # targeted repairs of code that fails the pre-run checks
    max_fix_tool_calls: int = 3  # incl. run_python to check a signature before fixing an API error
    propose_run_python_calls: int = 1  # exploration happens once up front; proposals mostly reuse notes.md
    plan_revise_every: int = 4  # rewrite plan.md after this many experiments
    explore_tool_calls: int = 8  # one exploration pass per task (design: before prepare.py; research: before exp_000)
    results_context_rows: int = 20

    @field_validator("data_dir", "runs_dir")
    @classmethod
    def _absolute(cls, p: Path) -> Path:
        """Relative paths (e.g. ./runs in .env) are relative to the repo root, not the cwd: agent code runs with
        cwd = its workspace, so a relative path would resolve differently there."""
        return p if p.is_absolute() else (ROOT / p).resolve()

    def clamp_budget(self, seconds: int) -> int:
        hi = self.demo_max_budget_s if self.budget_profile == "demo" else self.max_budget_s
        lo = min(self.min_budget_s, hi)
        return max(lo, min(hi, int(seconds)))


settings = Settings()
