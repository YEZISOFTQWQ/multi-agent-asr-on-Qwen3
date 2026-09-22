"""把五个独立 Agent 接入共享状态并记录每次决策周期。"""

from __future__ import annotations

from multi_agent_asr.agents.audio_preparation_agent import AudioPreparationAgent
from multi_agent_asr.agents.context_selection_agent import ContextSelectionAgent
from multi_agent_asr.agents.recognition_agent import RecognitionAgent
from multi_agent_asr.agents.supervisor_agent import SupervisorAgent
from multi_agent_asr.agents.transcript_review_agent import TranscriptReviewAgent
from multi_agent_asr.observability import SqliteRunRepository
from multi_agent_asr.schemas import ASRResult, SupervisorObservation
from multi_agent_asr.tools import HistoryRecorder

from .state import ASRGraphState


class ASRGraphNodes:
    """完成组件输入输出与 LangGraph 黑板状态之间的适配。

    本类不做自主决策。它把共享状态转换为各 Agent 的强类型输入，把结果
    写回对应状态字段，并为 Agent 节点和确定性收尾节点记录统一审计信息。
    """

    def __init__(
        self,
        *,
        supervisor_agent: SupervisorAgent,
        audio_preparation_agent: AudioPreparationAgent,
        context_selection_agent: ContextSelectionAgent,
        recognition_agent: RecognitionAgent,
        transcript_review_agent: TranscriptReviewAgent,
        history_recorder: HistoryRecorder,
        run_repository: SqliteRunRepository,
    ) -> None:
        self.supervisor_agent = supervisor_agent
        self.audio_preparation_agent = audio_preparation_agent
        self.context_selection_agent = context_selection_agent
        self.recognition_agent = recognition_agent
        self.transcript_review_agent = transcript_review_agent
        self.history_recorder = history_recorder
        self.run_repository = run_repository

    def _tracking(
        self,
        state: ASRGraphState,
        node_name: str,
        attempt: int = 1,
        details: dict[str, object] | None = None,
    ):
        """创建统一的节点运行审计上下文。

        Args:
            state: 当前运行的 LangGraph 黑板状态。
            node_name: 写入审计表的稳定节点名称。
            attempt: 同一类型节点的执行序号。
            details: 便于排障的静态补充信息。
        """
        runtime_details: dict[str, object] = {"log_schema_version": 1}
        if details:
            runtime_details.update(details)
        return self.run_repository.track(
            run_id=state["run_id"],
            thread_id=state["thread_id"],
            node_name=node_name,
            attempt=attempt,
            details=runtime_details,
        )

    async def supervisor(self, state: ASRGraphState) -> dict[str, object]:
        """让 Supervisor 观察黑板并选择唯一下一动作。"""
        step = state.get("supervisor_step", 0) + 1
        review = state.get("review")
        observation = SupervisorObservation(
            has_audio=state.get("audio_preparation") is not None,
            has_context=state.get("context_selection") is not None,
            has_candidate=state.get("recognition") is not None,
            has_review=review is not None,
            review_decision=review.decision if review else None,
            review_feedback=review.feedback if review else state.get("retry_feedback"),
            retry_count=state.get("retry_count", 0),
            max_retries=state.get("max_retries", 0),
            step_count=state.get("supervisor_step", 0),
        )
        async with self._tracking(
            state,
            "supervisor",
            step,
            {"agent": "SupervisorAgent"},
        ) as log:
            decision = await self.supervisor_agent.decide(observation)
            log.update(
                {
                    "observation": observation.model_dump(mode="json"),
                    "decision": decision.model_dump(mode="json"),
                }
            )

        updates: dict[str, object] = {
            "supervisor_step": decision.step,
            "supervisor_decision": decision,
            "supervisor_history": [*state.get("supervisor_history", []), decision],
            "retry_count": decision.retry_count,
            "retry_feedback": decision.retry_feedback,
        }
        # Supervisor 请求重试时，旧候选及其审核必须原子清空；否则下一轮
        # 观察会把旧产物误认为新识别已经完成。
        if decision.clears_previous_candidate:
            updates.update(
                {
                    "recognition": None,
                    "candidate": None,
                    "review": None,
                    "verification": None,
                }
            )
        return updates

    async def audio_preparation(self, state: ASRGraphState) -> dict[str, object]:
        """执行 Audio Preparation Agent 的有界处理循环。"""
        async with self._tracking(
            state,
            "audio_preparation",
            details={"agent": "AudioPreparationAgent"},
        ) as log:
            result = await self.audio_preparation_agent.run(
                state["request"].audio_path,
                state["run_id"],
            )
            log.update(
                {
                    "accepted": result.accepted,
                    "failure_reason": result.failure_reason,
                    "actions": result.actions,
                    "trace": [entry.model_dump(mode="json") for entry in result.trace],
                }
            )
            if not result.accepted or result.audio is None:
                raise ValueError(result.failure_reason or "Audio preparation failed")
        return {"audio_preparation": result, "audio": result.audio}

    async def context_selection(self, state: ASRGraphState) -> dict[str, object]:
        """执行 Context Selection Agent 的证据选择与预算复评。"""
        audio = state.get("audio")
        if audio is None:
            raise RuntimeError("Prepared audio is required before context selection")
        async with self._tracking(
            state,
            "context_selection",
            details={"agent": "ContextSelectionAgent"},
        ) as log:
            result = await self.context_selection_agent.run(state["request"], audio.path)
            log.update(
                {
                    "strategy": result.strategy,
                    "included_sources": result.included_sources,
                    "excluded_sources": result.excluded_sources,
                    "speaker": result.speaker.model_dump(mode="json"),
                    "scene": result.scene.model_dump(mode="json"),
                    "trace": [entry.model_dump(mode="json") for entry in result.trace],
                }
            )
        return {
            "context_selection": result,
            "speaker": result.speaker,
            "scene": result.scene,
            "profile": result.profile,
            "context": result.context,
        }

    async def recognition(self, state: ASRGraphState) -> dict[str, object]:
        """执行 Recognition Agent 的策略选择和模型调用。"""
        audio = state.get("audio")
        context = state.get("context")
        if audio is None or context is None:
            raise RuntimeError("Audio and context are required before recognition")
        attempt = state.get("retry_count", 0) + 1
        async with self._tracking(
            state,
            "recognition",
            attempt,
            {"agent": "RecognitionAgent"},
        ) as log:
            result = await self.recognition_agent.run(
                request=state["request"],
                audio=audio,
                context=context,
                profile=state.get("profile"),
                attempt=attempt,
                review_feedback=state.get("retry_feedback"),
            )
            log.update(
                {
                    "action": result.action,
                    "assessment": result.assessment,
                    "candidate_chars": len(result.candidate.text),
                    "applied_corrections": [
                        correction.model_dump(mode="json")
                        for correction in result.candidate.applied_corrections
                    ],
                    "trace": [entry.model_dump(mode="json") for entry in result.trace],
                }
            )
        return {"recognition": result, "candidate": result.candidate}

    async def transcript_review(self, state: ASRGraphState) -> dict[str, object]:
        """执行 Transcript Review Agent 并输出 Supervisor 可执行的反馈。"""
        audio = state.get("audio")
        candidate = state.get("candidate")
        if audio is None or candidate is None:
            raise RuntimeError("Audio and candidate are required before review")
        attempt = state.get("retry_count", 0) + 1
        async with self._tracking(
            state,
            "transcript_review",
            attempt,
            {"agent": "TranscriptReviewAgent"},
        ) as log:
            result = await self.transcript_review_agent.run(
                candidate=candidate,
                audio=audio,
                attempt=attempt,
                retry_count=state.get("retry_count", 0),
                max_retries=state.get("max_retries", 0),
            )
            log.update(
                {
                    "decision": result.decision,
                    "feedback": result.feedback,
                    "checks_run": result.checks_run,
                    "verification": result.verification.model_dump(mode="json"),
                    "trace": [entry.model_dump(mode="json") for entry in result.trace],
                }
            )
        return {"review": result, "verification": result.verification}

    async def finalize(self, state: ASRGraphState) -> dict[str, object]:
        """把审核后的黑板状态组装成稳定 API 结果。

        该节点仅映射数据，不进行新的识别或审核决策。
        """
        request = state["request"]
        candidate = state.get("candidate")
        verification = state.get("verification")
        if candidate is None or verification is None:
            raise RuntimeError("Reviewed candidate is required before finalization")
        speaker = state.get("speaker")
        scene = state.get("scene")
        async with self._tracking(state, "finalize"):
            result = ASRResult(
                text=candidate.text,
                language=candidate.language,
                session_id=request.session_id,
                run_id=state["run_id"],
                speaker_id=speaker.speaker_id if speaker else None,
                scene=scene.label if scene else None,
                verified=verification.verified,
                warnings=verification.warnings,
                context_used=candidate.context_used,
                time_stamps=candidate.time_stamps,
                applied_corrections=candidate.applied_corrections,
            )
        return {"result": result}

    async def persist(self, state: ASRGraphState) -> dict[str, object]:
        """调用确定性仓库工具保存结果，并保持结果内容不变。"""
        result = state["result"]
        async with self._tracking(state, "persist"):
            await self.history_recorder.record(result)
        return {"result": result}
