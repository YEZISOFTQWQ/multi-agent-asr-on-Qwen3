"""封装 Qwen3-ASR 的懒加载、并发控制和结果转换。"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

from multi_agent_asr.config import Settings
from multi_agent_asr.schemas import TimeStamp, TranscriptCandidate


class QwenASRService:
    """按需加载单个 Qwen3-ASR 实例并串行执行 GPU 推理。"""

    def __init__(self, settings: Settings) -> None:
        """保存模型配置，并创建加载锁和推理锁。"""
        self.settings = settings
        self._model: Any | None = None
        # 加载锁保护首次构造；推理锁限制单个 GPU 模型的并发入口。
        self._load_lock = threading.Lock()
        self._inference_lock = asyncio.Lock()

    @property
    def is_loaded(self) -> bool:
        """指示模型权重是否已经加载到当前进程。"""
        return self._model is not None

    def _torch_dtype(self):
        """把字符串配置映射为 PyTorch dtype。"""
        import torch

        mapping = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        try:
            return mapping[self.settings.dtype]
        except KeyError as error:
            raise ValueError(f"Unsupported dtype: {self.settings.dtype}") from error

    def _load_model(self):
        """线程安全地懒加载并缓存唯一的模型实例。"""
        if self._model is not None:
            return self._model
        with self._load_lock:
            # 获取锁后再次检查，避免两个线程先后重复加载权重。
            if self._model is not None:
                return self._model

            # 延迟导入保证健康检查和普通单元测试不触发 CUDA 初始化。
            from qwen_asr import Qwen3ASRModel

            kwargs: dict[str, Any] = {
                "dtype": self._torch_dtype(),
                "device_map": self.settings.device_map,
                "max_inference_batch_size": self.settings.max_inference_batch_size,
                "max_new_tokens": self.settings.max_new_tokens,
            }
            if self.settings.forced_aligner_model_path:
                kwargs["forced_aligner"] = self.settings.forced_aligner_model_path
                kwargs["forced_aligner_kwargs"] = {
                    "dtype": self._torch_dtype(),
                    "device_map": self.settings.device_map,
                }

            self._model = Qwen3ASRModel.from_pretrained(
                self.settings.asr_model_path,
                **kwargs,
            )
        return self._model

    async def transcribe(
        self,
        *,
        audio_path: str,
        context: str,
        language: str | None,
        return_time_stamps: bool,
    ) -> TranscriptCandidate:
        """串行进入模型，并把阻塞推理移到工作线程。"""
        # 官方 transcribe 是阻塞调用；锁内转到线程可保持事件循环可响应。
        async with self._inference_lock:
            return await asyncio.to_thread(
                self._transcribe_sync,
                audio_path,
                context,
                language,
                return_time_stamps,
            )

    def _transcribe_sync(
        self,
        audio_path: str,
        context: str,
        language: str | None,
        return_time_stamps: bool,
    ) -> TranscriptCandidate:
        """同步调用官方模型并转换第一条返回结果。"""
        model = self._load_model()
        results = model.transcribe(
            audio=audio_path,
            context=context,
            language=language,
            return_time_stamps=return_time_stamps,
        )
        if not results:
            return TranscriptCandidate(text="", language=language, context_used=context)

        result = results[0]
        stamps = [
            TimeStamp(
                text=item.text,
                start_time=item.start_time,
                end_time=item.end_time,
            )
            for item in (result.time_stamps or [])
        ]
        return TranscriptCandidate(
            text=result.text,
            language=result.language,
            context_used=context,
            time_stamps=stamps,
        )
