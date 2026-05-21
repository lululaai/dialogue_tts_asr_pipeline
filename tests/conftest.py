from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydub import AudioSegment

from pipeline.config import PipelineConfig


@pytest.fixture
def tiny_input(tmp_path: Path) -> Path:
    path = tmp_path / "input.json"
    path.write_text(
        json.dumps(
            {
                "d1": {
                    "content": [
                        {"message": "hello", "agent": "agent_1"},
                        {"message": "hi there", "agent": "agent_2"},
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def config(tmp_path: Path, tiny_input: Path) -> PipelineConfig:
    return PipelineConfig(
        input_json=str(tiny_input),
        output_dir=str(tmp_path / "out"),
        inter_turn_silence_ms=240,
        transcribe_each_chunk=False,
        tts_concurrency=1,
        asr_concurrency=1,
    )


def fake_tts(
    text: str,
    output_wav_path: str,
    voice: str,
    model: str,
    sample_rate: int,
    response_format: str = "wav",
    instructions: str | None = None,
    **_: object,
) -> None:
    duration = 320 if text == "hello" else 480
    audio = AudioSegment.silent(duration=duration, frame_rate=sample_rate).set_channels(1).set_sample_width(2)
    Path(output_wav_path).parent.mkdir(parents=True, exist_ok=True)
    audio.export(output_wav_path, format="wav")

