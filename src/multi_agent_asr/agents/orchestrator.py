from __future__ import annotations

import asyncio

from multi_agent_asr.agents.audio_agent import AudioAgent
from multi_agent_asr.agents.memory_agent import MemoryAgent
from multi_agent_asr.agents.profile_update_agent import ProfileUpdateAgent
from multi_agent_asr.agents.qwen_asr_agent import QwenASRAgent
from multi_agent_asr.agents.scene_agent import SceneAgent
from multi_agent_asr.agents.speaker_agent import SpeakerAgent
from multi_agent_asr.agents.verifier_agent import VerifierAgent
from multi_agent_asr.schemas import ASRResult, TranscriptionInput


class ASROrchestrator:
    def __init__(
        self,
        *,
        audio_agent: AudioAgent,
        speaker_agent: SpeakerAgent,
        scene_agent: SceneAgent,
        memory_agent: MemoryAgent,
        asr_agent: QwenASRAgent,
        verifier_agent: VerifierAgent,
        profile_update_agent: ProfileUpdateAgent,
    ) -> None:
        self.audio_agent = audio_agent
        self.speaker_agent = speaker_agent
        self.scene_agent = scene_agent
        self.memory_agent = memory_agent
        self.asr_agent = asr_agent
        self.verifier_agent = verifier_agent
        self.profile_update_agent = profile_update_agent

    async def initialize(self) -> None:
        await self.memory_agent.initialize()

    async def transcribe(self, request: TranscriptionInput) -> ASRResult:
        audio, speaker, scene = await asyncio.gather(
            self.audio_agent.inspect(request.audio_path),
            self.speaker_agent.identify(request.audio_path, request.speaker_hint),
            self.scene_agent.analyze(request.audio_path, request.scene_hint),
        )

        context = await self.memory_agent.build_context(
            session_id=request.session_id,
            speaker_id=speaker.speaker_id,
            scene=scene,
            explicit_context=request.explicit_context,
        )
        candidate = await self.asr_agent.transcribe(
            audio_path=audio.path,
            context=context,
            language=request.language,
            return_time_stamps=request.return_time_stamps,
        )
        verification = await self.verifier_agent.verify(candidate)

        result = ASRResult(
            text=candidate.text,
            language=candidate.language,
            session_id=request.session_id,
            speaker_id=speaker.speaker_id,
            scene=scene.label,
            verified=verification.verified,
            warnings=verification.warnings,
            context_used=candidate.context_used,
            time_stamps=candidate.time_stamps,
        )
        await self.profile_update_agent.record(result)
        return result
