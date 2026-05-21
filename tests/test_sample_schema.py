from __future__ import annotations

from pathlib import Path

from pipeline.build_dataset import build_sample
from pipeline.load_dialogues import load_dialogues
from pipeline.schemas import validate_sample_json

from conftest import fake_tts


def test_sample_json_validates_and_marks_silence(config):
    dialogue = load_dialogues(config.input_json, config)[0]
    sample_dir = Path(config.output_dir) / "samples" / "dialogue_000001"
    sample = build_sample(dialogue, config, sample_dir, tts_fn=fake_tts, skip_asr=True)

    validate_sample_json(sample, sample_dir)
    first = sample["chunk_targets"][0]
    assert first["input"]["user_voice"]["state"] == "present"
    assert first["output"]["assistant_voice"]["state"] == "silence"

    assistant_chunk = sample["chunk_targets"][4]
    assert assistant_chunk["input"]["user_voice"]["state"] == "silence"
    assert assistant_chunk["output"]["assistant_voice"]["state"] == "present"
