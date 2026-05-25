from __future__ import annotations

import json
from pathlib import Path

from pydub import AudioSegment
from pydub.generators import Sine

from pipeline.build_dataset import run_pipeline
from pipeline.config import PipelineConfig

from conftest import fake_tts


def _write_multi_input(path: Path, count: int = 4) -> None:
    path.write_text(
        json.dumps(
            {
                f"d{index}": {
                    "content": [
                        {"message": f"hello {index}", "agent": "agent_1"},
                        {"message": f"reply {index}", "agent": "agent_2"},
                    ]
                }
                for index in range(count)
            }
        ),
        encoding="utf-8",
    )


def test_run_pipeline_processes_samples_concurrently_and_writes_manifest(tmp_path):
    input_path = tmp_path / "input.json"
    _write_multi_input(input_path, count=4)
    config = PipelineConfig(
        input_json=str(input_path),
        output_dir=str(tmp_path / "out"),
        sample_concurrency=2,
        tts_concurrency=1,
        asr_concurrency=1,
        turn_overlap_enabled=False,
        backchannel_enabled=False,
    )

    rows = run_pipeline(config, tts_fn=fake_tts, skip_asr=True)

    assert len(rows) == 4
    manifest_lines = (Path(config.output_dir) / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(manifest_lines) == 4
    assert not (Path(config.output_dir) / "failed.jsonl").exists()
    for index in range(1, 5):
        sample_dir = Path(config.output_dir) / "samples" / f"dialogue_{index:06d}"
        assert (sample_dir / "sample.json").exists()
        assert (sample_dir / "audio/duplex_stereo.wav").exists()


def test_resume_with_sfx_only_fills_missing_sfx_output(tmp_path, monkeypatch):
    input_path = tmp_path / "input.json"
    _write_multi_input(input_path, count=1)
    output_dir = tmp_path / "out"
    base_config = PipelineConfig(
        input_json=str(input_path),
        output_dir=str(output_dir),
        sample_concurrency=1,
        tts_concurrency=1,
        asr_concurrency=1,
        turn_overlap_enabled=False,
        backchannel_enabled=False,
    )
    run_pipeline(base_config, tts_fn=fake_tts, skip_asr=True)

    sfx_root = tmp_path / "sfx"
    sfx_path = sfx_root / "audio_segments/Human sounds/laughter/laugh_asset/laugh_asset__part001.wav"
    sfx_path.parent.mkdir(parents=True)
    Sine(880).to_audio_segment(duration=300).set_frame_rate(24000).set_channels(1).set_sample_width(2).export(
        sfx_path,
        format="wav",
    )
    map_path = tmp_path / "uploaded_segments_map_to_file.json"
    map_path.write_text(
        json.dumps(
            {
                "_prefix": "tos://model-data-segment/audio_segments/",
                "Human sounds": {
                    "laughter": ["laugh_asset/laugh_asset__part001.wav"],
                },
            }
        ),
        encoding="utf-8",
    )

    def fake_generate_json(*_, **__):
        return {
            "events": [
                {
                    "category": "Human sounds",
                    "label": "laughter",
                    "start_ms": 100,
                    "duration_ms": 300,
                    "gain_db": -12,
                    "ducking_db": -1,
                    "reason": "test",
                }
            ]
        }

    monkeypatch.setattr("pipeline.sfx_mixer.generate_json", fake_generate_json)
    sfx_config = PipelineConfig(
        input_json=str(input_path),
        output_dir=str(output_dir),
        sample_concurrency=2,
        tts_concurrency=1,
        asr_concurrency=1,
        sfx_enabled=True,
        sfx_root=str(sfx_root),
        sfx_map_path=str(map_path),
    )

    rows = run_pipeline(sfx_config, resume=True, tts_fn=fake_tts, skip_asr=True)
    sample_dir = output_dir / "samples/dialogue_000001"
    sample = json.loads((sample_dir / "sample.json").read_text(encoding="utf-8"))

    assert len(rows) == 1
    assert sample["audio_files"]["duplex_stereo_sfx"] == "audio/duplex_stereo_sfx.wav"
    assert len(sample["sfx_events"]) == 1
    assert (sample_dir / "audio/duplex_stereo_sfx.wav").exists()
    assert AudioSegment.from_file(sample_dir / "audio/duplex_stereo_sfx.wav").channels == 2
