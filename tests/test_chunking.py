from __future__ import annotations

from pathlib import Path

from pydub import AudioSegment

from pipeline.align_chunks import split_audio_to_chunks


def test_split_audio_to_chunks_pads_last_chunk(tmp_path):
    audio_path = tmp_path / "stream.wav"
    AudioSegment.silent(duration=250, frame_rate=24000).set_channels(1).set_sample_width(2).export(audio_path, format="wav")

    chunks = split_audio_to_chunks(audio_path, tmp_path / "chunks" / "user_voice", 160, 24000)

    assert len(chunks) == 2
    assert chunks[1]["start_ms"] == 160
    assert chunks[1]["end_ms"] == 320
    assert len(AudioSegment.from_file(tmp_path / "chunks/user_voice/chunk_000001.wav")) == 160

