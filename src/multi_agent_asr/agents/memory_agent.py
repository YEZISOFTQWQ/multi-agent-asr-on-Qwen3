from __future__ import annotations

from multi_agent_asr.memory import ContextBuilder, SqliteMemoryRepository
from multi_agent_asr.schemas import SceneObservation, SpeakerProfile


class MemoryAgent:
    def __init__(
        self,
        repository: SqliteMemoryRepository,
        context_builder: ContextBuilder,
        recent_limit: int,
    ) -> None:
        self.repository = repository
        self.context_builder = context_builder
        self.recent_limit = recent_limit

    async def initialize(self) -> None:
        await self.repository.initialize()

    async def get_profile(self, speaker_id: str | None) -> SpeakerProfile | None:
        return await self.repository.get_profile(speaker_id)

    async def upsert_profile(self, profile: SpeakerProfile) -> SpeakerProfile:
        return await self.repository.upsert_profile(profile)

    async def build_context(
        self,
        *,
        session_id: str,
        speaker_id: str | None,
        scene: SceneObservation,
        explicit_context: str | None,
    ) -> str:
        profile = await self.repository.get_profile(speaker_id)
        recent = await self.repository.recent_utterances(session_id, self.recent_limit)
        return self.context_builder.build(
            profile=profile,
            scene=scene,
            recent_utterances=recent,
            explicit_context=explicit_context,
        )
