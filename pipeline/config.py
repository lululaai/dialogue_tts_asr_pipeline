from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PipelineConfig:
    input_json: str
    output_dir: str

    chunk_ms: int = 160
    sample_rate: int = 24000

    user_agent: str = "agent_1"
    assistant_agent: str = "agent_2"

    user_voice_name: str = "user_voice"
    assistant_voice_name: str = "assistant_voice"

    user_tts_voice: str = "alloy"
    assistant_tts_voice: str = "coral"

    tts_model: str = "gpt-4o-mini-tts"
    asr_model: str = "whisper-1"
    asr_mode: str = "turn"

    tts_response_format: str = "wav"

    inter_turn_silence_ms: int = 240

    preserve_original_agent_ids: bool = True

    transcribe_each_chunk: bool = True

    max_retries: int = 5

    fallback_turn_overlap: bool = True

    tts_concurrency: int = 3
    asr_concurrency: int = 6

    @property
    def agent_to_stream(self) -> dict[str, str]:
        return {
            self.user_agent: self.user_voice_name,
            self.assistant_agent: self.assistant_voice_name,
        }

    @property
    def stream_to_role(self) -> dict[str, str]:
        return {
            self.user_voice_name: "input",
            self.assistant_voice_name: "output",
        }
