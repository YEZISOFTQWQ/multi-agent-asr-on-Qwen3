"""读取并验证输入音频的基础元数据。"""

from __future__ import annotations

import asyncio
from pathlib import Path

import soundfile as sf

from multi_agent_asr.schemas import AudioInfo


class AudioAgent:
    """验证音频路径并提取后续节点需要的基础元数据。"""

    async def inspect(self, audio_path: str) -> AudioInfo:
        """在线程池中读取音频信息，避免阻塞事件循环。"""
        return await asyncio.to_thread(self._inspect_sync, audio_path)

    def _inspect_sync(self, audio_path: str) -> AudioInfo:
        """校验路径并通过 SoundFile 同步读取音频头信息。"""
        path = Path(audio_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")

        info = sf.info(path)
        return AudioInfo(
            path=str(path),
            duration_seconds=float(info.duration),
            sample_rate=int(info.samplerate),
            channels=int(info.channels),
        )
