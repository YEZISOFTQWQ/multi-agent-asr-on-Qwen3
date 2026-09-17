"""连接 Agent 编排层与持久化记忆层。"""

from __future__ import annotations

from multi_agent_asr.memory import ContextBuilder, SqliteMemoryRepository
from multi_agent_asr.schemas import SceneObservation, SpeakerProfile


class MemoryAgent:
    """为编排层提供画像读写和上下文构建接口。"""

    def __init__(
        self,
        repository: SqliteMemoryRepository,
        context_builder: ContextBuilder,
        recent_limit: int,
    ) -> None:
        """保存共享记忆仓库、上下文构建器和历史条数上限。"""
        self.repository = repository
        self.context_builder = context_builder
        self.recent_limit = recent_limit

    async def initialize(self) -> None:
        """初始化底层记忆表。"""
        await self.repository.initialize()

    async def get_profile(self, speaker_id: str | None) -> SpeakerProfile | None:
        """按说话人 ID 读取画像；未知说话人返回 None。"""
        return await self.repository.get_profile(speaker_id)

    async def upsert_profile(self, profile: SpeakerProfile) -> SpeakerProfile:
        """新增或更新一份说话人画像。"""
        return await self.repository.upsert_profile(profile)

    async def build_context(
        self,
        *,
        session_id: str,
        speaker_id: str | None,
        scene: SceneObservation,
        explicit_context: str | None,
    ) -> str:
        """组合画像、场景、显式提示和最近可信转写。"""
        profile = await self.repository.get_profile(speaker_id)
        recent = await self.repository.recent_utterances(session_id, self.recent_limit)
        return self.context_builder.build(
            profile=profile,
            scene=scene,
            recent_utterances=recent,
            explicit_context=explicit_context,
        )
