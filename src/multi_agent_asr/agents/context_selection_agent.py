"""实现上下文证据观察、选择、渲染和预算复评。"""

from __future__ import annotations

import asyncio

from multi_agent_asr.memory import ContextBuilder, SqliteMemoryRepository
from multi_agent_asr.schemas import (
    AgentTraceEntry,
    ContextSelectionResult,
    SpeakerProfile,
    TranscriptionInput,
)
from multi_agent_asr.tools import SceneResolver, SpeakerResolver


class ContextSelectionAgent:
    """在可信度和字符预算约束下选择 ASR 消歧上下文。

    Agent 将当前音频观察、长期说话人画像、已验证会话历史和用户显式
    上下文视为不同证据源。说话人身份置信度决定画像能否进入上下文，
    字符预算决定是否需要移除优先级较低的会话历史。
    """

    def __init__(
        self,
        repository: SqliteMemoryRepository,
        context_builder: ContextBuilder,
        speaker_resolver: SpeakerResolver,
        scene_resolver: SceneResolver,
        recent_limit: int,
        min_speaker_confidence: float = 0.5,
        max_steps: int = 2,
    ) -> None:
        """配置上下文所需的仓库、工具和证据选择阈值。

        Args:
            repository: 保存说话人画像和已验证会话历史的仓库。
            context_builder: 将选中证据渲染为模型上下文的构建器。
            speaker_resolver: 从请求和音频中解析说话人线索的工具。
            scene_resolver: 从请求和音频中解析录音场景的工具。
            recent_limit: 最多读取的近期已验证转写数量。
            min_speaker_confidence: 允许使用长期画像的最低身份置信度。
            max_steps: 预算触顶时允许重新选择证据的最大次数。
        """
        self.repository = repository
        self.context_builder = context_builder
        self.speaker_resolver = speaker_resolver
        self.scene_resolver = scene_resolver
        # 负数限制没有业务含义，统一收敛为“不读取历史”。
        self.recent_limit = max(0, recent_limit)
        self.min_speaker_confidence = min_speaker_confidence
        self.max_steps = max(1, max_steps)

    async def initialize(self) -> None:
        """初始化底层记忆表。"""
        await self.repository.initialize()

    async def get_profile(self, speaker_id: str | None) -> SpeakerProfile | None:
        """为画像 API 提供读取入口。"""
        return await self.repository.get_profile(speaker_id)

    async def upsert_profile(self, profile: SpeakerProfile) -> SpeakerProfile:
        """为画像 API 提供显式写入入口。"""
        return await self.repository.upsert_profile(profile)

    async def run(
        self,
        request: TranscriptionInput,
        prepared_audio_path: str,
    ) -> ContextSelectionResult:
        """收集可信证据，并在预算触顶时调整策略后重新渲染。

        Args:
            request: 当前转写请求及用户显式提供的线索。
            prepared_audio_path: Audio Preparation Agent 接受的音频路径。

        Returns:
            渲染后的上下文、证据取舍、观察结果和审计轨迹。
        """
        # 两个解析器彼此独立，可以并行观察同一份已准备音频。
        speaker, scene = await asyncio.gather(
            self.speaker_resolver.resolve(prepared_audio_path, request.speaker_hint),
            self.scene_resolver.resolve(prepared_audio_path, request.scene_hint),
        )
        profile, recent = await asyncio.gather(
            self.repository.get_profile(speaker.speaker_id),
            self.repository.recent_utterances(request.session_id, self.recent_limit),
        )

        # 画像与 speaker_id 绑定。身份置信度不足时继续保留会话历史，
        # 但不把可能属于其他人的长期画像发送给模型。
        trusted_profile = profile if speaker.confidence >= self.min_speaker_confidence else None
        strategy = "personalized" if trusted_profile is not None else "session_only"
        selected_recent = recent
        included = ["safety_instruction"]
        excluded: list[str] = []
        if request.explicit_context and request.explicit_context.strip():
            included.append("explicit_context")
        if trusted_profile is not None:
            included.append("speaker_profile")
        elif profile is not None:
            excluded.append("speaker_profile_low_identity_confidence")
        if scene.label and scene.label != "unknown":
            included.append("scene")
        if recent:
            included.append("verified_session_history")

        trace: list[AgentTraceEntry] = []
        context = ""
        for step in range(1, self.max_steps + 1):
            # ContextBuilder 会执行最终硬截断；等于上限意味着内容可能已经
            # 被截断，因此仍视为预算触顶并尝试一次更保守的证据组合。
            context = self.context_builder.build(
                profile=trusted_profile,
                scene=scene,
                recent_utterances=selected_recent,
                explicit_context=request.explicit_context,
            )
            at_budget = len(context) >= self.context_builder.max_chars
            trace.append(
                AgentTraceEntry(
                    step=step,
                    observation={
                        "speaker_confidence": speaker.confidence,
                        "has_profile": profile is not None,
                        "recent_count": len(selected_recent),
                        "rendered_chars": len(context),
                        "at_budget": at_budget,
                    },
                    action=f"render_{strategy}",
                    reason="select_only_trusted_context_sources",
                    outcome="budget_reached" if at_budget else "context_ready",
                )
            )
            if not at_budget or not selected_recent or step == self.max_steps:
                break
            # 会话历史优先级低于安全指令、显式上下文和可信画像。
            # 移除后重新渲染，避免简单截断破坏高优先级信息的完整性。
            selected_recent = []
            strategy = "budget_reduced"
            included = [source for source in included if source != "verified_session_history"]
            excluded.append("verified_session_history_budget_pressure")

        return ContextSelectionResult(
            context=context,
            strategy=strategy,
            speaker=speaker,
            scene=scene,
            profile=trusted_profile,
            included_sources=included,
            excluded_sources=excluded,
            trace=trace,
        )
