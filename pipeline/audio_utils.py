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


def build_stereo_audio(
    left_track: AudioSegment,
    right_track: AudioSegment,
    sample_rate: int,
) -> AudioSegment:
    left_track = normalize_audio_segment(left_track, sample_rate)
    right_track = normalize_audio_segment(right_track, sample_rate)
    duration_ms = max(len(left_track), len(right_track))
    if len(left_track) < duration_ms:
        left_track += AudioSegment.silent(duration=duration_ms - len(left_track), frame_rate=sample_rate)
    if len(right_track) < duration_ms:
        right_track += AudioSegment.silent(duration=duration_ms - len(right_track), frame_rate=sample_rate)
    return AudioSegment.from_mono_audiosegments(left_track, right_track).set_sample_width(2)


def select_overlap_turn_indices(turns: list[dict[str, Any]], config: PipelineConfig) -> set[int]:
    if not config.turn_overlap_enabled:
        return set()

    candidates = [
        idx
        for idx in range(1, len(turns))
        if turns[idx].get("stream") != turns[idx - 1].get("stream")
    ]
    if not candidates:
        return set()

    target_count = min(max(1, config.turn_overlap_max_count), len(candidates))
    if len(candidates) >= config.turn_overlap_min_count:
        target_count = min(config.turn_overlap_max_count, len(candidates))

    selected: list[int] = []
    if target_count == 1:
        selected = [candidates[len(candidates) // 2]]
    else:
        last_slot = len(candidates) - 1
        for slot in range(target_count):
            candidate = candidates[round(slot * last_slot / (target_count - 1))]
            if candidate not in selected:
                selected.append(candidate)

    non_adjacent: list[int] = []
    for idx in selected:
        if non_adjacent and idx - non_adjacent[-1] <= 1:
            replacement = next(
                (
                    candidate
                    for candidate in candidates
                    if candidate not in non_adjacent and candidate - non_adjacent[-1] > 1
                ),
                None,
            )
            if replacement is None:
                continue
            idx = replacement
        non_adjacent.append(idx)

    return set(non_adjacent)


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
    next_available_ms = 0
    overlap_indices = select_overlap_turn_indices(turns, config)
    for idx, turn in enumerate(turns):
        wav_path = turn_audio_dir / f"{turn['turn_id']}.wav"
        turn_audio = load_wav(wav_path, config.sample_rate)
        overlap_previous_turn_id = None
        overlap_ms = 0
        if idx == 0:
            start_ms = 0
        elif idx in overlap_indices:
            previous = turns[idx - 1]
            previous_start_ms = int(previous["start_ms"])
            previous_end_ms = int(previous["end_ms"])
            previous_duration_ms = max(1, previous_end_ms - previous_start_ms)
            ratio = min(0.95, max(0.05, config.turn_overlap_start_ratio))
            start_ms = previous_start_ms + round(previous_duration_ms * ratio)
            overlap_ms = max(0, previous_end_ms - start_ms)
            overlap_previous_turn_id = previous["turn_id"]
        else:
            start_ms = next_available_ms + config.inter_turn_silence_ms
        end_ms = start_ms + len(turn_audio)
        turn["start_ms"] = start_ms
        turn["end_ms"] = end_ms
        turn["overlap_with_previous"] = overlap_previous_turn_id is not None
        turn["overlap_previous_turn_id"] = overlap_previous_turn_id
        turn["overlap_ms"] = overlap_ms
        placements.append((turn, turn_audio))
        next_available_ms = max(next_available_ms, end_ms)

    duration_ms = next_available_ms
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
        tracks[stream_name] = track

    stereo_track = build_stereo_audio(
        tracks[config.user_voice_name],
        tracks[config.assistant_voice_name],
        config.sample_rate,
    )
    stereo_track.export(audio_dir / f"{config.stereo_audio_name}.wav", format="wav")

    return duration_ms


def overlay_backchannel_events(
    backchannel_events: list[dict[str, Any]],
    sample_dir: str | Path,
    audio_dir: str | Path,
    config: PipelineConfig,
) -> int:
    if not backchannel_events:
        return len(AudioSegment.from_file(Path(audio_dir) / f"{config.stereo_audio_name}.wav"))

    sample_dir = Path(sample_dir)
    audio_dir = Path(audio_dir)
    tracks = {
        config.user_voice_name: load_wav(audio_dir / f"{config.user_voice_name}.wav", config.sample_rate),
        config.assistant_voice_name: load_wav(audio_dir / f"{config.assistant_voice_name}.wav", config.sample_rate),
    }

    for event in backchannel_events:
        clip = load_wav(sample_dir / event["audio_path"], config.sample_rate)
        turn_start_ms = int(event["while_turn_start_ms"])
        turn_end_ms = int(event["while_turn_end_ms"])
        available_ms = turn_end_ms - turn_start_ms - config.backchannel_min_start_offset_ms - config.backchannel_min_end_margin_ms
        max_clip_ms = min(config.backchannel_max_duration_ms, available_ms)
        if max_clip_ms > 160 and len(clip) > max_clip_ms:
            clip = clip[:max_clip_ms].fade_out(min(80, max_clip_ms // 3))
        clip = clip.apply_gain(config.backchannel_gain_db)

        start_ms = int(event["start_ms"])
        earliest_ms = turn_start_ms + config.backchannel_min_start_offset_ms
        latest_ms = turn_end_ms - len(clip) - config.backchannel_min_end_margin_ms
        if latest_ms >= earliest_ms:
            start_ms = min(max(start_ms, earliest_ms), latest_ms)
        else:
            start_ms = max(turn_start_ms, min(start_ms, turn_end_ms))

        end_ms = start_ms + len(clip)
        duration_ms = max(max(len(track) for track in tracks.values()), end_ms)
        for stream_name, track in list(tracks.items()):
            if len(track) < duration_ms:
                track += AudioSegment.silent(duration=duration_ms - len(track), frame_rate=config.sample_rate)
                tracks[stream_name] = track

        stream = event["stream"]
        tracks[stream] = tracks[stream].overlay(clip, position=start_ms)
        event["start_ms"] = start_ms
        event["end_ms"] = end_ms
        event["duration_ms"] = len(clip)

    duration_ms = max(len(track) for track in tracks.values())
    for stream_name, track in tracks.items():
        track = normalize_audio_segment(track, config.sample_rate)
        track.export(audio_dir / f"{stream_name}.wav", format="wav")
        tracks[stream_name] = track

    stereo_track = build_stereo_audio(
        tracks[config.user_voice_name],
        tracks[config.assistant_voice_name],
        config.sample_rate,
    )
    stereo_track.export(audio_dir / f"{config.stereo_audio_name}.wav", format="wav")
    return duration_ms


def is_silence(audio_path: str | Path, dbfs_threshold: float = -45.0, rms_threshold: int = 80) -> bool:
    audio = AudioSegment.from_file(audio_path)
    if audio.rms <= rms_threshold:
        return True
    return audio.dBFS == float("-inf") or audio.dBFS < dbfs_threshold
