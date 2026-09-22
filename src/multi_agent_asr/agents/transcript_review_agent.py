"""实现检查选择、结果评估和结构化反馈决策。"""

from __future__ import annotations

from multi_agent_asr.schemas import (
    AgentTraceEntry,
    AudioInfo,
    TranscriptCandidate,
    TranscriptReviewResult,
    VerificationResult,
)
from multi_agent_asr.tools import TranscriptValidator


class TranscriptReviewAgent:
    """选择确定性检查并决定接受、重试或带告警结束。

    Agent 根据候选是否包含时间戳选择检查集合，再结合剩余重试预算生成
    Supervisor 可以直接执行的决策。Validator 只报告事实，是否重试由本
    Agent 决定。
    """

    def __init__(self, validator: TranscriptValidator) -> None:
        """注入可复用的确定性文本和时间戳检查工具。"""
        self.validator = validator

    async def run(
        self,
        *,
        candidate: TranscriptCandidate,
        audio: AudioInfo,
        attempt: int,
        retry_count: int,
        max_retries: int,
    ) -> TranscriptReviewResult:
        """按候选能力选择检查工具，并生成可执行的审核结论。

        Args:
            candidate: Recognition Agent 生成并完成术语纠错的候选。
            audio: 已检查音频，用于验证时间戳边界。
            attempt: 候选对应的识别尝试序号。
            retry_count: 已经消耗的重试次数。
            max_retries: 本次运行允许消耗的最大重试次数。

        Returns:
            校验结果、下一步决策、反馈内容和检查轨迹。
        """
        checks_run = ["text_quality"]
        warnings = await self.validator.validate_text(candidate)
        # 未请求或模型未返回时间戳时跳过对应检查，避免凭空生成告警。
        if candidate.time_stamps:
            checks_run.append("timestamp_consistency")
            warnings.extend(
                await self.validator.validate_timestamps(candidate, audio.duration_seconds)
            )
        # 多个检查可能报告同一问题；保序去重让反馈稳定且便于测试。
        warnings = list(dict.fromkeys(warnings))

        if not warnings:
            decision = "accept"
            feedback = None
            reason = "all_selected_checks_passed"
        elif retry_count < max_retries:
            decision = "retry_recognition"
            feedback = ", ".join(warnings)
            reason = "review_failed_and_retry_budget_is_available"
        else:
            decision = "finalize_unverified"
            feedback = ", ".join(warnings)
            reason = "review_failed_and_retry_budget_is_exhausted"

        verification = VerificationResult(
            verified=not warnings,
            warnings=warnings,
            retry_context=feedback if decision == "retry_recognition" else None,
        )
        trace = [
            AgentTraceEntry(
                step=1,
                observation={
                    "candidate_chars": len(candidate.text),
                    "timestamp_count": len(candidate.time_stamps),
                    "retry_count": retry_count,
                    "max_retries": max_retries,
                },
                action=decision,
                reason=reason,
                outcome="verified" if verification.verified else feedback,
            )
        ]
        return TranscriptReviewResult(
            verification=verification,
            decision=decision,
            feedback=feedback,
            checks_run=checks_run,
            attempt=attempt,
            trace=trace,
        )
