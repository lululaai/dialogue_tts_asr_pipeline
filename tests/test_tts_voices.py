from __future__ import annotations

from pathlib import Path

from pipeline.build_dataset import _select_dialogue_tts_voices
from pipeline.config import PipelineConfig


def test_random_tts_voices_are_stable_per_dialogue(tmp_path: Path) -> None:
    config = PipelineConfig(input_json="input.json", output_dir=str(tmp_path))

    first = _select_dialogue_tts_voices(config, "dialogue_000001")
    second = _select_dialogue_tts_voices(config, "dialogue_000001")

    assert first == second
    assert first[config.user_voice_name] in config.google_tts_voices
    assert first[config.assistant_voice_name] in config.google_tts_voices
    assert first[config.user_voice_name] != first[config.assistant_voice_name]


def test_non_random_tts_voices_use_configured_voices(tmp_path: Path) -> None:
    config = PipelineConfig(
        input_json="input.json",
        output_dir=str(tmp_path),
        tts_random_voice=False,
        user_tts_voice="Kore",
        assistant_tts_voice="Puck",
    )

    voices = _select_dialogue_tts_voices(config, "dialogue_000001")

    assert voices == {
        config.user_voice_name: "Kore",
        config.assistant_voice_name: "Puck",
    }
