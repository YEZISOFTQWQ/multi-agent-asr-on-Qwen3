"""将最终转写结果写入会话记忆。"""

from __future__ import annotations

from multi_agent_asr.memory import SqliteMemoryRepository
from multi_agent_asr.schemas import ASRResult


class ProfileUpdateAgent:
    """负责把最终结果追加到会话历史。"""

    def __init__(self, repository: SqliteMemoryRepository) -> None:
        """保存负责写入会话历史的共享仓库。"""
        self.repository = repository

    async def record(self, result: ASRResult) -> None:
        """把最终结果追加到短期会话历史。"""
        await self.repository.append_utterance(result)
