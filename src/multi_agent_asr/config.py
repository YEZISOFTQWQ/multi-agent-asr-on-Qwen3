from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="MASR_",
        extra="ignore",
    )

    environment: str = "development"
    host: str = "127.0.0.1"
    port: int = 8000

    asr_model_path: str = "Qwen/Qwen3-ASR-0.6B"
    forced_aligner_model_path: str | None = None
    device_map: str = "cuda:0"
    dtype: str = "bfloat16"
    max_inference_batch_size: int = 1
    max_new_tokens: int = 256

    data_root: Path = Path.home() / "data" / "multi-agent-asr"
    run_root: Path = Path.home() / "runs" / "multi-agent-asr"
    database_path: Path = Path.home() / "data" / "multi-agent-asr" / "state" / "memory.sqlite3"
    recent_utterance_limit: int = 5
    max_context_chars: int = 2000

    def ensure_runtime_directories(self) -> None:
        for path in (
            self.data_root / "raw",
            self.data_root / "processed",
            self.data_root / "annotations",
            self.data_root / "manifests",
            self.database_path.parent,
            self.run_root / "logs",
            self.run_root / "outputs",
            self.run_root / "checkpoints",
            self.run_root / "evaluations",
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
