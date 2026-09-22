"""提供可复用的确定性转写检查工具。"""

from __future__ import annotations

import re

from multi_agent_asr.schemas import TranscriptCandidate


class TranscriptValidator:
    """返回候选文本和时间戳中的确定性告警代码。"""

    async def validate_text(self, candidate: TranscriptCandidate) -> list[str]:
        """检查空文本、控制字符和异常重复。"""
        warnings: list[str] = []
        text = candidate.text.strip()
        if not text:
            warnings.append("empty_transcript")
        if any(ord(character) < 32 and character not in "\n\t" for character in text):
            warnings.append("control_character_detected")
        # 捕获长度 2 到 12 的片段连续出现三次，过滤常见单字叠词的同时
        # 识别模型卡住后产生的短句循环。
        if re.search(r"(.{2,12})\1\1", text):
            warnings.append("suspicious_repetition")
        return warnings

    async def validate_timestamps(
        self,
        candidate: TranscriptCandidate,
        audio_duration: float,
    ) -> list[str]:
        """检查时间戳的范围和顺序。"""
        warnings: list[str] = []
        previous_end = 0.0
        for stamp in candidate.time_stamps:
            if stamp.start_time < previous_end or stamp.end_time < stamp.start_time:
                warnings.append("invalid_timestamp_order")
                break
            if stamp.start_time < 0 or stamp.end_time > audio_duration + 0.25:
                # 允许 250ms 浮点和对齐误差，超出后才报告越过音频边界。
                warnings.append("timestamp_out_of_audio_range")
                break
            previous_end = stamp.end_time
        return warnings
