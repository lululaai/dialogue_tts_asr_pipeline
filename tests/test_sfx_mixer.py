from __future__ import annotations

import json
from pathlib import Path

from pydub import AudioSegment
from pydub.generators import Sine

from pipeline.config import PipelineConfig
from pipeline.sfx_mixer import SfxAsset, _build_prompt, load_sfx_catalog, mix_sample_sfx


def test_sfx_mixer_selects_only_uploaded_map_assets(tmp_path, monkeypatch):
    map_path = tmp_path / "uploaded_segments_map_to_file.json"
    map_path.write_text(
        json.dumps(
            {
                "_prefix": "tos://model-data-segment/audio_segments/",
                "Human sounds": {
                    "laughter": ["laugh_asset/laugh_asset__part001.wav"],
                    "breathing": ["missing/missing__part001.wav"],
                },
                "Music": {
                    "choir": ["not_downloaded/not_downloaded__part001.wav"],
                },
            }
        ),
        encoding="utf-8",
    )

    sfx_root = tmp_path / "sfx"
    sfx_path = sfx_root / "audio_segments/Human sounds/laughter/laugh_asset/laugh_asset__part001.wav"
    sfx_path.parent.mkdir(parents=True)
    Sine(880).to_audio_segment(duration=400).set_frame_rate(24000).set_channels(1).set_sample_width(2).export(
        sfx_path,
        format="wav",
    )

    sample_dir = tmp_path / "out/samples/dialogue_000001"
    audio_dir = sample_dir / "audio"
    audio_dir.mkdir(parents=True)
    stereo = AudioSegment.from_mono_audiosegments(
        AudioSegment.silent(duration=2000, frame_rate=24000),
        AudioSegment.silent(duration=2000, frame_rate=24000),
    )
    stereo.export(audio_dir / "duplex_stereo.wav", format="wav")

    sample = {
        "sample_id": "dialogue_000001",
        "source_dialogue_id": "d1",
        "sample_rate": 24000,
        "duration_ms": 2000,
        "streams": {"input": ["user_voice"], "output": ["assistant_voice"]},
        "audio_files": {"duplex_stereo": "audio/duplex_stereo.wav"},
        "turns": [
            {
                "turn_id": "u1",
                "stream": "user_voice",
                "start_ms": 0,
                "end_ms": 500,
                "text": "that was funny",
            }
        ],
        "overlap_events": [],
        "backchannel_events": [],
    }
    config = PipelineConfig(
        input_json="input.json",
        output_dir=str(tmp_path / "out"),
        sfx_enabled=True,
        sfx_map_path=str(map_path),
        sfx_root=str(sfx_root),
        sfx_max_events=1,
    )

    def fake_generate_json(*_, **__):
        return {
            "events": [
                {
                    "category": "Human sounds",
                    "label": "laughter",
                    "start_ms": 700,
                    "duration_ms": 400,
                    "gain_db": -12,
                    "ducking_db": -1,
                    "reason": "laughter after funny line",
                },
                {
                    "category": "Music",
                    "label": "choir",
                    "start_ms": 1200,
                    "duration_ms": 400,
                },
            ]
        }

    monkeypatch.setattr("pipeline.sfx_mixer.generate_json", fake_generate_json)

    catalog = load_sfx_catalog(config)
    assert sorted(catalog) == [("Human sounds", "laughter")]

    mixed_sample = mix_sample_sfx(sample, sample_dir, config)
    mixed_path = sample_dir / "audio/duplex_stereo_sfx.wav"

    assert mixed_path.exists()
    assert mixed_sample["audio_files"]["duplex_stereo_sfx"] == "audio/duplex_stereo_sfx.wav"
    assert len(mixed_sample["sfx_events"]) == 1
    assert mixed_sample["sfx_events"][0]["category"] == "Human sounds"
    assert mixed_sample["sfx_events"][0]["label"] == "laughter"
    assert mixed_sample["sfx_events"][0]["asset_path"].endswith("laugh_asset__part001.wav")
    assert AudioSegment.from_file(mixed_path).channels == 2


def test_sfx_prompt_matches_dialogue_cues_without_category_bias(tmp_path):
    catalog = {
        ("Human sounds", "laughter"): [
            SfxAsset("Human sounds", "laughter", "laugh/laugh.wav", tmp_path / "laugh.wav")
        ],
        ("Sounds of things", "doors_windows_locks"): [
            SfxAsset("Sounds of things", "doors_windows_locks", "door/door.wav", tmp_path / "door.wav")
        ],
    }
    sample = {
        "duration_ms": 2000,
        "turns": [
            {
                "turn_id": "u1",
                "stream": "user_voice",
                "start_ms": 0,
                "end_ms": 500,
                "text": "That was funny, then someone opened the door.",
            }
        ],
    }
    config = PipelineConfig(input_json="input.json", output_dir=str(tmp_path / "out"))

    prompt = _build_prompt(sample, catalog, config)

    assert "Do not prefer any category by default" in prompt
    assert "Choose each event by the strongest cue in the dialogue text" in prompt
    assert "Use Human sounds only when the cue is human" in prompt
    assert "doors_windows_locks" in prompt
