from __future__ import annotations

from pathlib import Path

from pipeline.build_dataset import build_sample
from pipeline.config import PipelineConfig
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


def test_turn_asr_words_are_mapped_to_chunks(tmp_path, tiny_input):
    config = PipelineConfig(
        input_json=str(tiny_input),
        output_dir=str(tmp_path / "out"),
        transcribe_each_chunk=True,
        asr_mode="turn",
        asr_model="whisper-1",
        tts_concurrency=1,
        asr_concurrency=1,
    )

    def fake_turn_asr(turn_wav_path, model, prompt=None, **kwargs):
        if prompt == "hello":
            return {"text": "hello", "words": [{"word": "hello", "start": 0.0, "end": 0.32}]}
        return {
            "text": "hi there",
            "words": [
                {"word": "hi", "start": 0.0, "end": 0.24},
                {"word": "there", "start": 0.24, "end": 0.48},
            ],
        }

    dialogue = load_dialogues(config.input_json, config)[0]
    sample_dir = Path(config.output_dir) / "samples" / "dialogue_000001"
    sample = build_sample(dialogue, config, sample_dir, tts_fn=fake_tts, asr_fn=fake_turn_asr)

    first = sample["chunk_targets"][0]["input"]["user_voice"]
    assistant = sample["chunk_targets"][5]["output"]["assistant_voice"]

    assert first["text"] == "hello"
    assert first["text_source"] == "asr_turn_words"
    assert first["words"][0]["source"] == "asr"
    assert assistant["text"] == "there"
    assert assistant["text_source"] == "asr_turn_words"
