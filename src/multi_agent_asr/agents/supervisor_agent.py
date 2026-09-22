"""实现全局目标驱动的观察、路由、反馈处理和终止决策。"""

from __future__ import annotations

from multi_agent_asr.schemas import SupervisorDecision, SupervisorObservation


class SupervisorAgent:
    """根据全局状态选择下一名 Agent，并管理重试反馈。

    Supervisor 不执行音频处理或模型推理。它观察共享状态是否已经具备
    各阶段产物，选择唯一下一动作，并通过步数和重试预算限制反馈闭环。
    """

    def __init__(self, max_steps: int = 16) -> None:
        """设置一次工作流允许的最大 Supervisor 决策次数。"""
        # 完整成功路径至少需要五次决策：四个 Agent 加一次 finalize。
        self.max_steps = max(5, max_steps)

    async def decide(self, observation: SupervisorObservation) -> SupervisorDecision:
        """根据当前观察选择唯一下一动作。

        前置产物按照音频、上下文、候选、审核的顺序检查。审核要求重试时，
        Supervisor 增加重试计数并要求图清除上一轮候选及审核结果。
        """
        step = observation.step_count + 1
        # 步数上限是最后一道保险，用于阻止未来新增路由造成意外死循环。
        if step > self.max_steps:
            if observation.has_candidate and observation.has_review:
                return SupervisorDecision(
                    action="finalize",
                    reason="supervisor_step_limit_reached_with_candidate",
                    step=step,
                    retry_count=observation.retry_count,
                    retry_feedback=observation.review_feedback,
                )
            raise RuntimeError("Supervisor step limit exceeded before a candidate was reviewed")

        # 按数据依赖顺序补齐缺失产物。每个 Agent 完成后都会回到
        # Supervisor，因此这里每轮只选择一个动作。
        if not observation.has_audio:
            return SupervisorDecision(
                action="audio_preparation",
                reason="prepared_audio_is_missing",
                step=step,
                retry_count=observation.retry_count,
            )
        if not observation.has_context:
            return SupervisorDecision(
                action="context_selection",
                reason="trusted_context_is_missing",
                step=step,
                retry_count=observation.retry_count,
            )
        if not observation.has_candidate:
            return SupervisorDecision(
                action="recognition",
                reason="transcript_candidate_is_missing",
                step=step,
                retry_count=observation.retry_count,
                retry_feedback=observation.review_feedback,
            )
        if not observation.has_review:
            return SupervisorDecision(
                action="transcript_review",
                reason="candidate_has_not_been_reviewed",
                step=step,
                retry_count=observation.retry_count,
            )
        if (
            observation.review_decision == "retry_recognition"
            and observation.retry_count < observation.max_retries
        ):
            # 清除标志由图适配层执行，防止旧候选让下一轮误判为已经完成
            # Recognition 或 Review。
            return SupervisorDecision(
                action="recognition",
                reason="review_requested_retry_with_available_budget",
                step=step,
                retry_count=observation.retry_count + 1,
                retry_feedback=observation.review_feedback,
                clears_previous_candidate=True,
            )
        return SupervisorDecision(
            action="finalize",
            reason="review_accepted_or_retry_budget_exhausted",
            step=step,
            retry_count=observation.retry_count,
            retry_feedback=observation.review_feedback,
        )
