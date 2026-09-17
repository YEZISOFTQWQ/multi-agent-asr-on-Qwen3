"""应用说话人画像中已经确认的术语纠错。"""

from __future__ import annotations

import re
from collections import Counter

from multi_agent_asr.schemas import AppliedCorrection, SpeakerProfile, TranscriptCandidate


class TerminologyAgent:
    """只应用记忆中已经明确确认的说话人专属纠错。"""

    async def apply(
        self,
        candidate: TranscriptCandidate,
        profile: SpeakerProfile | None,
    ) -> TranscriptCandidate:
        """使用单次最长匹配应用画像中已经确认的纠错。"""
        if profile is None or not candidate.text or not profile.corrections:
            return candidate

        replacements = {
            original: replacement
            for original, replacement in profile.corrections.items()
            if original and original != replacement
        }
        if not replacements:
            return candidate

        # 长词优先避免较短前缀抢先命中；单次正则替换也阻止替换结果级联。
        originals = sorted(replacements, key=len, reverse=True)
        pattern = re.compile("|".join(re.escape(original) for original in originals))
        counts: Counter[str] = Counter()

        def replace(match: re.Match[str]) -> str:
            """记录命中次数并返回当前原词对应的替换文本。"""
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
