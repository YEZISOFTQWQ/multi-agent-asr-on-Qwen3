"""把 Qwen3-ASR 模型服务适配为图中的转写 Agent。"""

from __future__ import annotations

from multi_agent_asr.schemas import TranscriptCandidate
from multi_agent_asr.services import QwenASRService


class QwenASRAgent:
    """向图节点暴露与模型实现无关的异步转写接口。"""

    def __init__(self, service: QwenASRService) -> None:
        """保存共享的 Qwen3-ASR 模型服务。"""
        self.service = service

    async def transcribe(
        self,
        *,
        audio_path: str,
        context: str,
        language: str | None,
        return_time_stamps: bool,
    ) -> TranscriptCandidate:
        """把结构化转写参数原样委托给模型服务。"""
        return await self.service.transcribe(
            audio_path=audio_path,
            context=context,
            language=language,
            return_time_stamps=return_time_stamps,
        )
