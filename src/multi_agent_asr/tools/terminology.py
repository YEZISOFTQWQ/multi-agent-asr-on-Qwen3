"""应用已经确认的确定性术语纠错。"""

from __future__ import annotations

import re
from collections import Counter

from multi_agent_asr.schemas import AppliedCorrection, SpeakerProfile, TranscriptCandidate


class TerminologyCorrector:
    """使用单次最长匹配应用说话人画像中的确认纠错。"""

    async def apply(
        self,
        candidate: TranscriptCandidate,
        profile: SpeakerProfile | None,
    ) -> TranscriptCandidate:
        """执行非级联替换并保留命中次数审计。"""
        if profile is None or not candidate.text or not profile.corrections:
            return candidate

        replacements = {
            original: replacement
            for original, replacement in profile.corrections.items()
            if original and original != replacement
        }
        if not replacements:
            return candidate

        # 长词优先防止短词先命中并破坏更具体的术语。例如同时配置
        # “千问”和“千问语音”时，后者应作为一个整体匹配。
        originals = sorted(replacements, key=len, reverse=True)
        pattern = re.compile("|".join(re.escape(original) for original in originals))
        counts: Counter[str] = Counter()

        # re.sub 单次扫描原文本，replacement 不会再次进入匹配，实现非级联替换。
        def replace(match: re.Match[str]) -> str:
            original = match.group(0)
            counts[original] += 1
            return replacements[original]

        corrected_text = pattern.sub(replace, candidate.text)
        applied = [
            AppliedCorrection(
                original=original,
                replacement=replacements[original],
                occurrences=counts[original],
            )
            for original in originals
            if counts[original]
        ]
        if not applied:
            return candidate

        return candidate.model_copy(
            update={
                "text": corrected_text,
                "applied_corrections": [*candidate.applied_corrections, *applied],
            }
        )
