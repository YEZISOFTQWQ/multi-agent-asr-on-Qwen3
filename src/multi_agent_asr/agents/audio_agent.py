from __future__ import annotations

import asyncio
from pathlib import Path

import soundfile as sf

from multi_agent_asr.schemas import AudioInfo


class AudioAgent:
    async def inspect(self, audio_path: str) -> AudioInfo:
        return await asyncio.to_thread(self._inspect_sync, audio_path)

    def _inspect_sync(self, audio_path: str) -> AudioInfo:
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
