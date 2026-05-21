from __future__ import annotations

from pathlib import Path

from pydub import AudioSegment

from pipeline.build_dataset import build_sample
from pipeline.load_dialogues import load_dialogues

from conftest import fake_tts


def test_timeline_tracks_have_same_duration(config):
    dialogue = load_dialogues(config.input_json, config)[0]
    sample = build_sample(dialogue, config, Path(config.output_dir) / "samples" / "dialogue_000001", tts_fn=fake_tts, skip_asr=True)

    user = AudioSegment.from_file(Path(config.output_dir) / "samples/dialogue_000001/audio/user_voice.wav")
    assistant = AudioSegment.from_file(Path(config.output_dir) / "samples/dialogue_000001/audio/assistant_voice.wav")

    assert len(user) == len(assistant) == sample["duration_ms"]
    assert sample["turns"][0]["start_ms"] == 0
    assert sample["turns"][0]["end_ms"] == 320
    assert sample["turns"][1]["start_ms"] == 560
    assert sample["turns"][1]["end_ms"] == 1040
