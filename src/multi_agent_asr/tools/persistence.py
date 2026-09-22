"""提供最终结果持久化工具。"""

from __future__ import annotations

from multi_agent_asr.memory import SqliteMemoryRepository
from multi_agent_asr.schemas import ASRResult


class HistoryRecorder:
    """把最终结果写入会话历史，不自行做长期记忆决策。"""

    def __init__(self, repository: SqliteMemoryRepository) -> None:
        self.repository = repository

    async def record(self, result: ASRResult) -> None:
        """追加最终结果和校验状态。"""
        await self.repository.append_utterance(result)
