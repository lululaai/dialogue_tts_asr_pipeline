from __future__ import annotations

import json
from pathlib import Path

from pydub import AudioSegment
from pydub.generators import Sine

from pipeline.build_dataset import build_sample
from pipeline.config import PipelineConfig
from pipeline.load_dialogues import load_dialogues

from conftest import fake_tts


def fake_long_tts(
    text: str,
    output_wav_path: str,
    voice: str,
    model: str,
    sample_rate: int,
    response_format: str = "wav",
    instructions: str | None = None,
    **_: object,
) -> None:
    if text.startswith("very long"):
        duration = 6500
    elif text.startswith("long"):
        duration = 3600
    else:
        duration = 300
    audio = Sine(440).to_audio_segment(duration=duration).set_frame_rate(sample_rate).set_channels(1).set_sample_width(2)
    Path(output_wav_path).parent.mkdir(parents=True, exist_ok=True)
    audio.export(output_wav_path, format="wav")


def test_timeline_tracks_have_same_duration(config):
    dialogue = load_dialogues(config.input_json, config)[0]
    sample = build_sample(dialogue, config, Path(config.output_dir) / "samples" / "dialogue_000001", tts_fn=fake_tts, skip_asr=True)

    user = AudioSegment.from_file(Path(config.output_dir) / "samples/dialogue_000001/audio/user_voice.wav")
    assistant = AudioSegment.from_file(Path(config.output_dir) / "samples/dialogue_000001/audio/assistant_voice.wav")
    stereo = AudioSegment.from_file(Path(config.output_dir) / "samples/dialogue_000001/audio/duplex_stereo.wav")

    assert len(user) == len(assistant) == sample["duration_ms"]
    assert len(stereo) == sample["duration_ms"]
    assert stereo.channels == 2
    assert sample["audio_files"]["duplex_stereo"] == "audio/duplex_stereo.wav"
    assert sample["turns"][0]["start_ms"] == 0
    assert sample["turns"][0]["end_ms"] == 320
    assert sample["turns"][1]["start_ms"] == 160
    assert sample["turns"][1]["end_ms"] == 640
    assert sample["duration_ms"] == 640
    assert sample["overlap_events"] == [
        {
            "turn_id": "a1",
            "overlaps_turn_id": "u1",
            "start_ms": 160,
            "overlap_start_ms": 160,
            "overlap_end_ms": 320,
            "overlap_ms": 160,
        }
    ]


def test_timeline_selects_two_or_three_non_adjacent_overlaps(tmp_path):
    input_path = tmp_path / "input.json"
    input_path.write_text(
        json.dumps(
            {
                "d1": {
                    "content": [
                        {"message": f"turn {index}", "agent": "agent_1" if index % 2 == 0 else "agent_2"}
                        for index in range(8)
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    config = PipelineConfig(
        input_json=str(input_path),
        output_dir=str(tmp_path / "out"),
        tts_concurrency=1,
        asr_concurrency=1,
    )

    dialogue = load_dialogues(config.input_json, config)[0]
    sample = build_sample(dialogue, config, Path(config.output_dir) / "samples" / "dialogue_000001", tts_fn=fake_tts, skip_asr=True)

    overlap_source_indices = [
        turn["source_turn_index"]
        for turn in sample["turns"]
        if turn["overlap_with_previous"]
    ]
    assert len(sample["overlap_events"]) == 3
    assert overlap_source_indices == [1, 4, 7]
    assert "chunk_targets" not in sample


def test_backchannel_events_are_generated_for_long_turns(tmp_path):
    input_path = tmp_path / "input.json"
    input_path.write_text(
        json.dumps(
            {
                "d1": {
                    "content": [
                        {"message": f"long turn {index}", "agent": "agent_1" if index % 2 == 0 else "agent_2"}
                        for index in range(4)
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    config = PipelineConfig(
        input_json=str(input_path),
        output_dir=str(tmp_path / "out"),
        turn_overlap_enabled=False,
        backchannel_max_count=3,
        tts_concurrency=1,
        asr_concurrency=1,
    )

    dialogue = load_dialogues(config.input_json, config)[0]
    sample_dir = Path(config.output_dir) / "samples" / "dialogue_000001"
    sample = build_sample(dialogue, config, sample_dir, tts_fn=fake_long_tts, skip_asr=True)

    assert len(sample["backchannel_events"]) == 3
    assert [event["text"] for event in sample["backchannel_events"]] == ["yes", "yeah", "yep"]
    for event in sample["backchannel_events"]:
        assert event["end_ms"] == event["start_ms"] + 300
        assert (sample_dir / event["audio_path"]).exists()


def test_backchannel_events_can_repeat_with_varied_phrases_in_one_long_turn(tmp_path):
    input_path = tmp_path / "input.json"
    input_path.write_text(
        json.dumps(
            {
                "d1": {
                    "content": [
                        {"message": "very long turn", "agent": "agent_1"},
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    config = PipelineConfig(
        input_json=str(input_path),
        output_dir=str(tmp_path / "out"),
        turn_overlap_enabled=False,
        backchannel_max_count=3,
        tts_concurrency=1,
        asr_concurrency=1,
    )

    dialogue = load_dialogues(config.input_json, config)[0]
    sample_dir = Path(config.output_dir) / "samples" / "dialogue_000001"
    sample = build_sample(dialogue, config, sample_dir, tts_fn=fake_long_tts, skip_asr=True)

    assert [event["text"] for event in sample["backchannel_events"]] == ["yes", "yeah", "yep"]
    assert len({event["start_ms"] for event in sample["backchannel_events"]}) == 3
