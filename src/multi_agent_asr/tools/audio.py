"""提供确定性的音频检查与变换工具。"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Protocol

import soundfile as sf

from multi_agent_asr.schemas import AudioInfo


class AudioPreparationTools(Protocol):
    """Audio Preparation Agent 可调用的工具协议。"""

    async def inspect(self, audio_path: str) -> AudioInfo:
        """检查音频并返回元数据。"""

    async def downmix_to_mono(self, audio_path: str, output_path: str) -> AudioInfo:
        """把多声道音频合成为单声道并返回新文件信息。"""


class SoundFileAudioTools:
    """使用 SoundFile 实现音频元数据读取和单声道转换。"""

    async def inspect(self, audio_path: str) -> AudioInfo:
        """在线程中读取文件头，避免阻塞事件循环。"""
        return await asyncio.to_thread(self._inspect_sync, audio_path)

    def _inspect_sync(self, audio_path: str) -> AudioInfo:
        """验证路径并读取音频基础信息。"""
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

    async def downmix_to_mono(self, audio_path: str, output_path: str) -> AudioInfo:
        """在线程中把所有声道取平均值并写成单声道 WAV。"""
        await asyncio.to_thread(self._downmix_sync, audio_path, output_path)
        return await self.inspect(output_path)

    def _downmix_sync(self, audio_path: str, output_path: str) -> None:
        """执行无损于时长的确定性声道平均。"""
        source = Path(audio_path).expanduser().resolve()
        target = Path(output_path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        samples, sample_rate = sf.read(source, dtype="float32", always_2d=True)
        mono = samples.mean(axis=1)
        sf.write(target, mono, sample_rate)
