from __future__ import annotations

from dataclasses import dataclass, field


GEMINI_TTS_VOICES: tuple[str, ...] = (
    "Zephyr",
    "Puck",
    "Charon",
    "Kore",
    "Fenrir",
    "Leda",
    "Orus",
    "Aoede",
    "Callirrhoe",
    "Autonoe",
    "Enceladus",
    "Iapetus",
    "Umbriel",
    "Algieba",
    "Despina",
    "Erinome",
    "Algenib",
    "Rasalgethi",
    "Laomedeia",
    "Achernar",
    "Alnilam",
    "Schedar",
    "Gacrux",
    "Pulcherrima",
    "Achird",
    "Zubenelgenubi",
    "Vindemiatrix",
    "Sadachbia",
    "Sadaltager",
    "Sulafat",
)


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
    stereo_audio_name: str = "duplex_stereo"

    user_tts_voice: str = "Kore"
    assistant_tts_voice: str = "Puck"
    tts_speed: float = 1.1

    tts_provider: str = "google"
    tts_model: str = "gemini-2.5-flash-tts"
    tts_random_voice: bool = True
    google_tts_voices: tuple[str, ...] = GEMINI_TTS_VOICES
    asr_model: str = "whisper-1"
    asr_mode: str = "turn"

    tts_response_format: str = "wav"

    inter_turn_silence_ms: int = 240

    preserve_original_agent_ids: bool = True

    generate_chunk_targets: bool = False
    transcribe_each_chunk: bool = False

    max_retries: int = 5

    fallback_turn_overlap: bool = True

    turn_overlap_enabled: bool = True
    turn_overlap_min_count: int = 2
    turn_overlap_max_count: int = 3
    turn_overlap_start_ratio: float = 0.5

    backchannel_enabled: bool = True
    backchannel_max_count: int = 3
    backchannel_min_turn_duration_ms: int = 3000
    backchannel_max_duration_ms: int = 650
    backchannel_min_start_offset_ms: int = 900
    backchannel_min_end_margin_ms: int = 650
    backchannel_gain_db: float = -5.0
    backchannel_phrases: tuple[str, ...] = field(
        default_factory=lambda: (
            "yes",
            "yeah",
            "yep",
            "right",
            "correct",
            "exactly",
            "true",
            "that's right",
            "you are right",
            "I agree",
            "sure",
            "of course",
            "totally",
            "absolutely",
        )
    )

    tts_concurrency: int = 0
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
