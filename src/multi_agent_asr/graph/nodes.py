"""将现有 Agent 方法包装为可观测的 LangGraph 节点。"""

from __future__ import annotations

import asyncio

from multi_agent_asr.agents.audio_agent import AudioAgent
from multi_agent_asr.agents.memory_agent import MemoryAgent
from multi_agent_asr.agents.profile_update_agent import ProfileUpdateAgent
from multi_agent_asr.agents.qwen_asr_agent import QwenASRAgent
from multi_agent_asr.agents.scene_agent import SceneAgent
from multi_agent_asr.agents.speaker_agent import SpeakerAgent
from multi_agent_asr.agents.terminology_agent import TerminologyAgent
from multi_agent_asr.agents.verifier_agent import VerifierAgent
from multi_agent_asr.observability import SqliteRunRepository
from multi_agent_asr.schemas import ASRResult

from .state import ASRGraphState


class ASRGraphNodes:
    """把专职 Agent 组合成读写 ASRGraphState 的异步节点。"""

    def __init__(
        self,
        *,
        audio_agent: AudioAgent,
        speaker_agent: SpeakerAgent,
        scene_agent: SceneAgent,
        memory_agent: MemoryAgent,
        asr_agent: QwenASRAgent,
        terminology_agent: TerminologyAgent,
        verifier_agent: VerifierAgent,
        profile_update_agent: ProfileUpdateAgent,
        run_repository: SqliteRunRepository,
    ) -> None:
        """保存所有节点依赖和统一的运行记录仓库。"""
        self.audio_agent = audio_agent
        self.speaker_agent = speaker_agent
        self.scene_agent = scene_agent
        self.memory_agent = memory_agent
        self.asr_agent = asr_agent
        self.terminology_agent = terminology_agent
        self.verifier_agent = verifier_agent
        self.profile_update_agent = profile_update_agent
        self.run_repository = run_repository

    def _tracking(self, state: ASRGraphState, node_name: str, attempt: int = 1):
        """为节点创建统一的成功、失败和耗时记录上下文。"""
        return self.run_repository.track(
            run_id=state["run_id"],
            thread_id=state["thread_id"],
            node_name=node_name,
            attempt=attempt,
        )

    async def audio(self, state: ASRGraphState) -> dict[str, object]:
        """运行音频检查节点。"""
        async with self._tracking(state, "audio"):
            audio = await self.audio_agent.inspect(state["request"].audio_path)
        return {"audio": audio}

    async def speaker(self, state: ASRGraphState) -> dict[str, object]:
        """运行说话人识别节点。"""
        request = state["request"]
        async with self._tracking(state, "speaker"):
            speaker = await self.speaker_agent.identify(
                request.audio_path,
                request.speaker_hint,
            )
        return {"speaker": speaker}

    async def scene(self, state: ASRGraphState) -> dict[str, object]:
        """运行录音场景分析节点。"""
        request = state["request"]
        async with self._tracking(state, "scene"):
            scene = await self.scene_agent.analyze(request.audio_path, request.scene_hint)
        return {"scene": scene}

    async def context(self, state: ASRGraphState) -> dict[str, object]:
        """并行读取画像并构建当前请求的模型上下文。"""
        request = state["request"]
        speaker = state["speaker"]
        scene = state["scene"]
        async with self._tracking(state, "context"):
            # profile 需继续传给术语节点，因此与渲染后的 context 一并读取。
            context, profile = await asyncio.gather(
                self.memory_agent.build_context(
                    session_id=request.session_id,
                    speaker_id=speaker.speaker_id,
                    scene=scene,
                    explicit_context=request.explicit_context,
                ),
                self.memory_agent.get_profile(speaker.speaker_id),
            )
        return {"context": context, "profile": profile}

    async def asr(self, state: ASRGraphState) -> dict[str, object]:
        """按当前 attempt 调用 Qwen3-ASR Agent。"""
        request = state["request"]
        attempt = state.get("retry_count", 0) + 1
        async with self._tracking(state, "asr", attempt):
            candidate = await self.asr_agent.transcribe(
                audio_path=state["audio"].path,
                context=state["context"],
                language=request.language,
                return_time_stamps=request.return_time_stamps,
            )
        return {"candidate": candidate}

    async def terminology(self, state: ASRGraphState) -> dict[str, object]:
        """对当前候选文本应用确定性术语纠错。"""
        attempt = state.get("retry_count", 0) + 1
        async with self._tracking(state, "terminology", attempt):
            candidate = await self.terminology_agent.apply(
                state["candidate"],
                state.get("profile"),
            )
        return {"candidate": candidate}

    async def verify(self, state: ASRGraphState) -> dict[str, object]:
        """校验当前候选并产生条件路由所需结果。"""
        attempt = state.get("retry_count", 0) + 1
        async with self._tracking(state, "verify", attempt):
            verification = await self.verifier_agent.verify(state["candidate"])
        return {"verification": verification}

    async def retry(self, state: ASRGraphState) -> dict[str, object]:
        """增加重试计数，并把失败原因加入下一轮上下文。"""
        retry_count = state.get("retry_count", 0) + 1
        verification = state["verification"]
        reason = verification.retry_context or ", ".join(verification.warnings)
        # 重试提示只指出失败类型，并再次声明音频证据优先，防止上下文臆测。
        instruction = (
            "上一轮转写未通过校验。请重新识别音频，重点避免以下问题："
            f"{reason or 'unknown_verification_failure'}。音频证据仍具有最高优先级。"
        )
        context = f"{state['context']}\n重试提示：{instruction}".strip()
        async with self._tracking(state, "retry", retry_count):
            return {
                "retry_count": retry_count,
                "retry_reason": reason,
                "context": context,
            }

    async def finalize(self, state: ASRGraphState) -> dict[str, object]:
        """把图状态转换为稳定的 API 返回模型。"""
        request = state["request"]
        candidate = state["candidate"]
        verification = state["verification"]
        async with self._tracking(state, "finalize"):
            result = ASRResult(
                text=candidate.text,
                language=candidate.language,
                session_id=request.session_id,
                run_id=state["run_id"],
                speaker_id=state["speaker"].speaker_id,
                scene=state["scene"].label,
                verified=verification.verified,
                warnings=verification.warnings,
                context_used=candidate.context_used,
                time_stamps=candidate.time_stamps,
                applied_corrections=candidate.applied_corrections,
            )
        return {"result": result}

    async def persist(self, state: ASRGraphState) -> dict[str, object]:
        """保存最终结果，使已校验文本可用于后续上下文。"""
        result = state["result"]
        async with self._tracking(state, "persist"):
            await self.profile_update_agent.record(result)
        return {"result": result}
