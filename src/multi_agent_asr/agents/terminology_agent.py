from __future__ import annotations

import re
from collections import Counter

from multi_agent_asr.schemas import AppliedCorrection, SpeakerProfile, TranscriptCandidate


class TerminologyAgent:
    """Apply speaker-specific corrections that were explicitly confirmed in memory."""

    async def apply(
        self,
        candidate: TranscriptCandidate,
        profile: SpeakerProfile | None,
    ) -> TranscriptCandidate:
        if profile is None or not candidate.text or not profile.corrections:
            return candidate

        replacements = {
            original: replacement
            for original, replacement in profile.corrections.items()
            if original and original != replacement
        }
        if not replacements:
            return candidate

        originals = sorted(replacements, key=len, reverse=True)
        pattern = re.compile("|".join(re.escape(original) for original in originals))
        counts: Counter[str] = Counter()

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
