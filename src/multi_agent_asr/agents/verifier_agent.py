"""对候选转写执行轻量、确定性的质量检查。"""

from __future__ import annotations

import re

from multi_agent_asr.schemas import TranscriptCandidate, VerificationResult


class VerifierAgent:
    """检查空结果、控制字符和明显的异常重复。"""

    async def verify(self, candidate: TranscriptCandidate) -> VerificationResult:
        """返回确定性的校验结果和可用于重试的告警代码。"""
        warnings: list[str] = []
        text = candidate.text.strip()

        if not text:
            warnings.append("empty_transcript")
        if any(ord(character) < 32 and character not in "\n\t" for character in text):
            warnings.append("control_character_detected")
        if re.search(r"(.{2,12})\1\1", text):
            warnings.append("suspicious_repetition")

        return VerificationResult(verified=not warnings, warnings=warnings)
