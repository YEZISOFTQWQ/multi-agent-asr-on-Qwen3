from __future__ import annotations

import asyncio
import threading
from typing import Any

from multi_agent_asr.config import Settings
from multi_agent_asr.schemas import TimeStamp, TranscriptCandidate


class QwenASRService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model: Any | None = None
        self._load_lock = threading.Lock()
        self._inference_lock = asyncio.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _torch_dtype(self):
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
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is not None:
                return self._model

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
