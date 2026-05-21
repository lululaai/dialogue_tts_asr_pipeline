from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from pydub import AudioSegment
from pydub.utils import which

from .config import PipelineConfig


def ensure_ffmpeg_available() -> None:
    if not shutil.which("ffmpeg") and not which("ffmpeg"):
        raise RuntimeError("ffmpeg is required but was not found on PATH")


def load_wav(path: str | Path, sample_rate: int) -> AudioSegment:
    audio = AudioSegment.from_file(path)
    return normalize_audio_segment(audio, sample_rate)


def normalize_audio_segment(audio: AudioSegment, sample_rate: int) -> AudioSegment:
    return audio.set_frame_rate(sample_rate).set_channels(1).set_sample_width(2)


def normalize_wav_file(input_path: str | Path, output_path: str | Path, sample_rate: int) -> None:
    normalized = load_wav(input_path, sample_rate)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    normalized.export(output_path, format="wav")


def build_stream_tracks(
    turns: list[dict[str, Any]],
    turn_audio_dir: str | Path,
    audio_dir: str | Path,
    config: PipelineConfig,
) -> int:
    ensure_ffmpeg_available()
    turn_audio_dir = Path(turn_audio_dir)
    audio_dir = Path(audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)

    placements: list[tuple[dict[str, Any], AudioSegment]] = []
    current_ms = 0
    for idx, turn in enumerate(turns):
        if idx > 0:
            current_ms += config.inter_turn_silence_ms
        wav_path = turn_audio_dir / f"{turn['turn_id']}.wav"
        turn_audio = load_wav(wav_path, config.sample_rate)
        start_ms = current_ms
        end_ms = start_ms + len(turn_audio)
        turn["start_ms"] = start_ms
        turn["end_ms"] = end_ms
        placements.append((turn, turn_audio))
        current_ms = end_ms

    duration_ms = current_ms
    user_track = AudioSegment.silent(duration=duration_ms, frame_rate=config.sample_rate).set_channels(1).set_sample_width(2)
    assistant_track = AudioSegment.silent(duration=duration_ms, frame_rate=config.sample_rate).set_channels(1).set_sample_width(2)
    tracks = {
        config.user_voice_name: user_track,
        config.assistant_voice_name: assistant_track,
    }

    for turn, turn_audio in placements:
        stream = turn["stream"]
        tracks[stream] = tracks[stream].overlay(turn_audio, position=int(turn["start_ms"]))

    for stream_name, track in tracks.items():
        track = normalize_audio_segment(track, config.sample_rate)
        track.export(audio_dir / f"{stream_name}.wav", format="wav")

    return duration_ms


def is_silence(audio_path: str | Path, dbfs_threshold: float = -45.0, rms_threshold: int = 80) -> bool:
    audio = AudioSegment.from_file(audio_path)
    if audio.rms <= rms_threshold:
        return True
    return audio.dBFS == float("-inf") or audio.dBFS < dbfs_threshold

