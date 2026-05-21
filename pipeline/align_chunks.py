from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from pydub import AudioSegment

from .audio_utils import normalize_audio_segment

CJK_RE = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def split_audio_to_chunks(
    audio_path: str | Path,
    output_chunk_dir: str | Path,
    chunk_ms: int,
    sample_rate: int,
) -> list[dict[str, Any]]:
    audio_path = Path(audio_path)
    output_chunk_dir = Path(output_chunk_dir)
    output_chunk_dir.mkdir(parents=True, exist_ok=True)

    audio = normalize_audio_segment(AudioSegment.from_file(audio_path), sample_rate)
    duration_ms = len(audio)
    num_chunks = math.ceil(duration_ms / chunk_ms) if duration_ms else 0
    stream_name = output_chunk_dir.name
    metadata: list[dict[str, Any]] = []

    for chunk_id in range(num_chunks):
        start_ms = chunk_id * chunk_ms
        end_ms = start_ms + chunk_ms
        chunk = audio[start_ms:min(end_ms, duration_ms)]
        if len(chunk) < chunk_ms:
            chunk += AudioSegment.silent(duration=chunk_ms - len(chunk), frame_rate=sample_rate)
        chunk = normalize_audio_segment(chunk, sample_rate)
        chunk_name = f"chunk_{chunk_id:06d}.wav"
        chunk.export(output_chunk_dir / chunk_name, format="wav")
        metadata.append(
            {
                "chunk_id": chunk_id,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "audio_path": f"audio/chunks/{stream_name}/{chunk_name}",
            }
        )

    return metadata


def find_overlapping_turns(
    turns: list[dict[str, Any]],
    stream: str,
    chunk_start_ms: int,
    chunk_end_ms: int,
) -> list[dict[str, Any]]:
    overlaps: list[dict[str, Any]] = []
    window_ms = chunk_end_ms - chunk_start_ms
    for turn in turns:
        if turn.get("stream") != stream:
            continue
        turn_start = turn.get("start_ms")
        turn_end = turn.get("end_ms")
        if turn_start is None or turn_end is None:
            continue
        overlap_ms = max(0, min(turn_end, chunk_end_ms) - max(turn_start, chunk_start_ms))
        if overlap_ms <= 0:
            continue
        enriched = dict(turn)
        enriched["overlap_ms"] = overlap_ms
        enriched["overlap_ratio"] = overlap_ms / window_ms
        overlaps.append(enriched)
    overlaps.sort(key=lambda item: item["overlap_ms"], reverse=True)
    return overlaps


def split_text_into_words_or_cjk_chars(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    if CJK_RE.search(stripped) and not re.search(r"\s", stripped):
        return [char for char in stripped if not char.isspace()]
    return TOKEN_RE.findall(stripped)


def estimate_word_timings(turn_text: str, turn_start_ms: int, turn_end_ms: int) -> list[dict[str, Any]]:
    tokens = split_text_into_words_or_cjk_chars(turn_text)
    if not tokens:
        return []

    duration = max(1, turn_end_ms - turn_start_ms)
    weights = [max(1, len(token)) for token in tokens]
    total_weight = sum(weights)
    cursor = turn_start_ms
    words: list[dict[str, Any]] = []

    for index, (token, weight) in enumerate(zip(tokens, weights)):
        if index == len(tokens) - 1:
            end_ms = turn_end_ms
        else:
            end_ms = turn_start_ms + round(duration * sum(weights[: index + 1]) / total_weight)
        words.append(
            {
                "word": token,
                "start_ms": cursor,
                "end_ms": max(cursor, end_ms),
                "overlap_ratio": 1.0,
                "source": "estimated",
            }
        )
        cursor = max(cursor, end_ms)

    return words


def words_overlapping_chunk(words: list[dict[str, Any]], chunk_start_ms: int, chunk_end_ms: int) -> list[dict[str, Any]]:
    window: list[dict[str, Any]] = []
    for word in words:
        start_ms = int(word.get("start_ms", 0))
        end_ms = int(word.get("end_ms", 0))
        overlap_ms = max(0, min(end_ms, chunk_end_ms) - max(start_ms, chunk_start_ms))
        if overlap_ms <= 0:
            continue
        word_duration = max(1, end_ms - start_ms)
        copied = dict(word)
        copied["overlap_ratio"] = overlap_ms / word_duration
        window.append(copied)
    return window


def normalize_asr_words(
    asr_result: dict[str, Any],
    chunk_start_ms: int,
    chunk_end_ms: int,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for raw_word in asr_result.get("words") or []:
        word = raw_word.get("word") or raw_word.get("text") or ""
        if not word:
            continue
        start = raw_word.get("start")
        end = raw_word.get("end")
        if start is None or end is None:
            continue
        local_start_ms = round(float(start) * 1000)
        local_end_ms = round(float(end) * 1000)
        global_start = chunk_start_ms + local_start_ms
        global_end = chunk_start_ms + local_end_ms
        overlap_ms = max(0, min(global_end, chunk_end_ms) - max(global_start, chunk_start_ms))
        duration = max(1, global_end - global_start)
        normalized.append(
            {
                "word": word,
                "start_ms": global_start,
                "end_ms": global_end,
                "overlap_ratio": overlap_ms / duration,
                "source": "asr",
            }
        )
    return normalized

