"""集中定义环境变量、模型参数和运行时路径。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """从 MASR_ 环境变量读取应用、模型和持久化配置。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="MASR_",
        extra="ignore",
    )

    environment: str = "development"
    host: str = "127.0.0.1"
    port: int = 8000

    # 模型参数只描述如何加载 Qwen；服务启动阶段不会立即加载权重。
    asr_model_path: str = "Qwen/Qwen3-ASR-0.6B"
    forced_aligner_model_path: str | None = None
    device_map: str = "cuda:0"
    dtype: str = "bfloat16"
    max_inference_batch_size: int = 1
    max_new_tokens: int = 256

    # 业务记忆和 LangGraph Checkpoint 分库，便于独立备份与排障。
    data_root: Path = Path.home() / "data" / "multi-agent-asr"
    run_root: Path = Path.home() / "runs" / "multi-agent-asr"
    database_path: Path = Path.home() / "data" / "multi-agent-asr" / "state" / "memory.sqlite3"
    checkpoint_database_path: Path = (
        Path.home() / "data" / "multi-agent-asr" / "state" / "checkpoints.sqlite3"
    )
    recent_utterance_limit: int = 5
    max_context_chars: int = 2000
    max_asr_retries: int = 1

    def ensure_runtime_directories(self) -> None:
        """创建应用运行时需要的所有目录，重复调用是安全的。"""
        for path in (
            self.data_root / "raw",
            self.data_root / "processed",
            self.data_root / "annotations",
            self.data_root / "manifests",
            self.database_path.parent,
            self.checkpoint_database_path.parent,
            self.run_root / "logs",
            self.run_root / "outputs",
            self.run_root / "checkpoints",
            self.run_root / "evaluations",
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """返回进程内缓存的配置对象，避免重复解析环境变量。"""
    return Settings()
