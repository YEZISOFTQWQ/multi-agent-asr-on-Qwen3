from __future__ import annotations

from multi_agent_asr.memory import SqliteMemoryRepository
from multi_agent_asr.schemas import ASRResult


class ProfileUpdateAgent:
    def __init__(self, repository: SqliteMemoryRepository) -> None:
        self.repository = repository

    async def record(self, result: ASRResult) -> None:
        await self.repository.append_utterance(result)
