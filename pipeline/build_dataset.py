from __future__ import annotations

import json
import logging
import math
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .align_chunks import (
    estimate_word_timings,
    find_overlapping_turns,
    normalize_asr_words,
    split_audio_to_chunks,
    words_overlapping_chunk,
)
from .audio_utils import build_stream_tracks, ensure_ffmpeg_available, overlay_backchannel_events
from .config import PipelineConfig
from .google_limiter import configure_google_request_limit
from .load_dialogues import load_dialogues
from .openai_audio import (
    transcribe_audio_chunk,
    transcribe_audio_turn_with_word_timestamps,
)
from .schemas import validate_sample_json
from .sfx_mixer import SfxCatalog, load_sfx_catalog, mix_sample_sfx

LOGGER = logging.getLogger(__name__)

TTSFn = Callable[..., None]
ASRFn = Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class SampleResult:
    row: dict[str, Any]
    status: str


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(data, ensure_ascii=False) + "\n")


def _resolve_tts_fn(config: PipelineConfig, tts_fn: TTSFn | None) -> TTSFn:
    if tts_fn is not None:
        return tts_fn
    if config.tts_provider == "google":
        from .google_audio import synthesize_tts

        return synthesize_tts
    if config.tts_provider == "openai":
        from .openai_audio import synthesize_tts

        return synthesize_tts
    raise ValueError(f"Unsupported tts_provider: {config.tts_provider!r}")


def _stable_voice_index(config: PipelineConfig, key: str) -> int:
    digest = hashlib.sha256(f"{config.tts_model}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % len(config.google_tts_voices)


def _select_dialogue_tts_voices(config: PipelineConfig, sample_id: str) -> dict[str, str]:
    voice_by_stream = {
        config.user_voice_name: config.user_tts_voice,
        config.assistant_voice_name: config.assistant_tts_voice,
    }
    if config.tts_provider != "google" or not config.tts_random_voice:
        return voice_by_stream
    if not config.google_tts_voices:
        raise ValueError("--google-tts-voices must include at least one voice")

    user_index = _stable_voice_index(config, f"{sample_id}:{config.user_voice_name}")
    assistant_index = _stable_voice_index(config, f"{sample_id}:{config.assistant_voice_name}")
    if len(config.google_tts_voices) > 1 and assistant_index == user_index:
        assistant_index = (assistant_index + 1) % len(config.google_tts_voices)

    return {
        config.user_voice_name: config.google_tts_voices[user_index],
        config.assistant_voice_name: config.google_tts_voices[assistant_index],
    }


def _tts_worker_count(config: PipelineConfig, item_count: int) -> int:
    if item_count <= 0:
        return 1
    if config.tts_concurrency <= 0:
        return item_count
    return max(1, config.tts_concurrency)


def _format_elapsed(seconds: float) -> str:
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:d}:{seconds:02d}"


def run_pipeline(
    config: PipelineConfig,
    *,
    limit: int | None = None,
    resume: bool = False,
    skip_tts: bool = False,
    skip_asr: bool = False,
    force: bool = False,
    tts_fn: TTSFn | None = None,
    asr_fn: ASRFn | None = None,
) -> list[dict[str, Any]]:
    ensure_ffmpeg_available()
    configure_google_request_limit(config.google_request_concurrency)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    failed_path = output_dir / "failed.jsonl"

    if not resume:
        manifest_path.unlink(missing_ok=True)
        failed_path.unlink(missing_ok=True)

    tts_fn = _resolve_tts_fn(config, tts_fn)
    dialogues = load_dialogues(config.input_json, config)
    if limit is not None:
        dialogues = dialogues[:limit]
    sfx_catalog = load_sfx_catalog(config) if config.sfx_enabled else None

    manifest_rows: list[dict[str, Any]] = []
    worker_count = max(1, config.sample_concurrency)
    total = len(dialogues)
    started_at = time.perf_counter()
    stats = {
        "generated": 0,
        "resumed": 0,
        "resumed_sfx": 0,
        "failed": 0,
        "duration_ms": 0,
        "turns": 0,
        "chunks": 0,
        "overlap_events": 0,
        "backchannel_events": 0,
        "sfx_events": 0,
    }
    LOGGER.info(
        "Starting pipeline: samples=%s sample_concurrency=%s tts_concurrency=%s "
        "asr_concurrency=%s google_request_concurrency=%s sfx_enabled=%s resume=%s",
        total,
        worker_count,
        config.tts_concurrency,
        config.asr_concurrency,
        config.google_request_concurrency,
        config.sfx_enabled,
        resume,
    )

    def _process(dialogue: dict[str, Any]) -> SampleResult:
        sample_dir = output_dir / "samples" / dialogue["sample_id"]
        sample_json_path = sample_dir / "sample.json"
        if resume and not force and sample_json_path.exists():
            sample = json.loads(sample_json_path.read_text(encoding="utf-8"))
            status = "resumed"
            if config.sfx_enabled and config.sfx_audio_name not in sample.get("audio_files", {}):
                sample = mix_sample_sfx(sample, sample_dir, config, force=force, catalog=sfx_catalog)
                validate_sample_json(sample, sample_dir)
                write_json(sample_dir / "metadata.json", {k: v for k, v in sample.items() if k != "chunk_targets"})
                write_json(sample_json_path, sample)
                status = "resumed_sfx"
            LOGGER.info("Skipping completed sample %s", dialogue["sample_id"])
            return SampleResult(_manifest_row(sample), status)

        sample = build_sample(
            dialogue,
            config,
            sample_dir,
            skip_tts=skip_tts,
            skip_asr=skip_asr,
            force=force,
            tts_fn=tts_fn,
            asr_fn=asr_fn,
            sfx_catalog=sfx_catalog,
        )
        return SampleResult(_manifest_row(sample), "generated")

    def _record_success(result: SampleResult) -> None:
        row = result.row
        append_jsonl(manifest_path, row)
        manifest_rows.append(row)
        stats[result.status] += 1
        stats["duration_ms"] += int(row.get("duration_ms") or 0)
        stats["turns"] += int(row.get("num_turns") or 0)
        stats["chunks"] += int(row.get("num_chunks") or 0)
        stats["overlap_events"] += int(row.get("num_overlaps") or 0)
        stats["backchannel_events"] += int(row.get("num_backchannels") or 0)
        stats["sfx_events"] += int(row.get("num_sfx_events") or 0)
        _log_progress(row["sample_id"], result.status)

    def _record_failure(dialogue: dict[str, Any], exc: Exception) -> None:
        stats["failed"] += 1
        LOGGER.exception("Failed to process %s", dialogue["sample_id"])
        append_jsonl(
            failed_path,
            {
                "sample_id": dialogue["sample_id"],
                "source_dialogue_id": dialogue["source_dialogue_id"],
                "error": str(exc),
            },
        )
        _log_progress(dialogue["sample_id"], "failed")

    def _log_progress(sample_id: str, status: str) -> None:
        completed = len(manifest_rows) + stats["failed"]
        elapsed = time.perf_counter() - started_at
        processed_per_min = (completed / elapsed * 60.0) if elapsed > 0 else 0.0
        remaining = max(0, total - completed)
        eta_seconds = (remaining / processed_per_min * 60.0) if processed_per_min > 0 else 0.0
        LOGGER.info(
            "Pipeline progress %s/%s: sample=%s status=%s elapsed=%s eta=%s "
            "rate=%.2f samples/min audio_min=%.2f turns=%s chunks=%s "
            "overlap_events=%s backchannel_events=%s sfx_events=%s "
            "generated=%s resumed=%s resumed_sfx=%s failed=%s",
            completed,
            total,
            sample_id,
            status,
            _format_elapsed(elapsed),
            _format_elapsed(eta_seconds),
            processed_per_min,
            stats["duration_ms"] / 60000.0,
            stats["turns"],
            stats["chunks"],
            stats["overlap_events"],
            stats["backchannel_events"],
            stats["sfx_events"],
            stats["generated"],
            stats["resumed"],
            stats["resumed_sfx"],
            stats["failed"],
        )

    def _log_summary() -> None:
        elapsed = time.perf_counter() - started_at
        succeeded = len(manifest_rows)
        LOGGER.info(
            "Finished pipeline: samples=%s succeeded=%s failed=%s elapsed=%s "
            "rate=%.2f samples/min audio_min=%.2f turns=%s chunks=%s "
            "overlap_events=%s backchannel_events=%s sfx_events=%s "
            "generated=%s resumed=%s resumed_sfx=%s",
            total,
            succeeded,
            stats["failed"],
            _format_elapsed(elapsed),
            (succeeded / elapsed * 60.0) if elapsed > 0 else 0.0,
            stats["duration_ms"] / 60000.0,
            stats["turns"],
            stats["chunks"],
            stats["overlap_events"],
            stats["backchannel_events"],
            stats["sfx_events"],
            stats["generated"],
            stats["resumed"],
            stats["resumed_sfx"],
        )

    if worker_count == 1:
        for dialogue in dialogues:
            try:
                _record_success(_process(dialogue))
            except Exception as exc:
                _record_failure(dialogue, exc)
        _log_summary()
        return manifest_rows

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_dialogue = {executor.submit(_process, dialogue): dialogue for dialogue in dialogues}
        for future in as_completed(future_to_dialogue):
            dialogue = future_to_dialogue[future]
            try:
                _record_success(future.result())
            except Exception as exc:
                _record_failure(dialogue, exc)

    _log_summary()
    return manifest_rows


def build_sample(
    dialogue: dict[str, Any],
    config: PipelineConfig,
    sample_dir: str | Path,
    *,
    skip_tts: bool = False,
    skip_asr: bool = False,
    force: bool = False,
    tts_fn: TTSFn | None = None,
    asr_fn: ASRFn | None = None,
    sfx_catalog: SfxCatalog | None = None,
) -> dict[str, Any]:
    sample_dir = Path(sample_dir)
    turn_audio_dir = sample_dir / "audio" / "turns"
    audio_dir = sample_dir / "audio"
    backchannel_audio_dir = audio_dir / "backchannels"
    chunk_root = audio_dir / "chunks"
    chunk_asr_root = sample_dir / "asr" / "chunks"
    turn_asr_root = sample_dir / "asr" / "turns"
    cache_tts_dir = Path(config.output_dir) / "cache" / "tts"
    cache_asr_dir = Path(config.output_dir) / "cache" / "asr"
    turns = [dict(turn) for turn in dialogue["turns"]]
    tts_fn = _resolve_tts_fn(config, tts_fn)
    tts_voices = _select_dialogue_tts_voices(config, dialogue["sample_id"])

    _generate_turn_audio(
        turns,
        config,
        tts_voices,
        turn_audio_dir,
        cache_tts_dir,
        skip_tts=skip_tts,
        force=force,
        tts_fn=tts_fn,
    )

    duration_ms = build_stream_tracks(turns, turn_audio_dir, audio_dir, config)
    backchannel_events = _select_backchannel_events(turns, config)
    _generate_backchannel_audio(
        backchannel_events,
        config,
        tts_voices,
        backchannel_audio_dir,
        cache_tts_dir,
        skip_tts=skip_tts,
        force=force,
        tts_fn=tts_fn,
    )
    duration_ms = overlay_backchannel_events(backchannel_events, sample_dir, audio_dir, config)
    overlap_events = _build_overlap_events(turns)

    sample = {
        "sample_id": dialogue["sample_id"],
        "source_dialogue_id": dialogue["source_dialogue_id"],
        "sample_rate": config.sample_rate,
        "duration_ms": duration_ms,
        "streams": {
            "input": [config.user_voice_name],
            "output": [config.assistant_voice_name],
        },
        "audio_files": {
            config.user_voice_name: f"audio/{config.user_voice_name}.wav",
            config.assistant_voice_name: f"audio/{config.assistant_voice_name}.wav",
            config.stereo_audio_name: f"audio/{config.stereo_audio_name}.wav",
        },
        "turns": [_serialize_turn(turn) for turn in turns],
        "overlap_events": overlap_events,
        "backchannel_events": backchannel_events,
        "tts": {
            "provider": config.tts_provider,
            "model": config.tts_model,
            "speed": config.tts_speed,
            "random_voice": config.tts_random_voice,
            "voices": tts_voices,
        },
        "source_metadata": dialogue.get("source_metadata", {}),
    }

    if config.generate_chunk_targets:
        user_chunks = split_audio_to_chunks(
            audio_dir / f"{config.user_voice_name}.wav",
            chunk_root / config.user_voice_name,
            config.chunk_ms,
            config.sample_rate,
        )
        assistant_chunks = split_audio_to_chunks(
            audio_dir / f"{config.assistant_voice_name}.wav",
            chunk_root / config.assistant_voice_name,
            config.chunk_ms,
            config.sample_rate,
        )
        num_chunks = math.ceil(duration_ms / config.chunk_ms) if duration_ms else 0

        chunk_asr_results: dict[str, dict[int, dict[str, Any]]] = {
            config.user_voice_name: {},
            config.assistant_voice_name: {},
        }
        turn_word_timings: dict[str, list[dict[str, Any]]] = {}
        if config.transcribe_each_chunk and not skip_asr:
            if config.asr_mode == "turn":
                turn_asr_fn = asr_fn or transcribe_audio_turn_with_word_timestamps
                turn_word_timings = _transcribe_turns(
                    turns,
                    turn_audio_dir,
                    turn_asr_root,
                    cache_asr_dir,
                    config,
                    force=force,
                    asr_fn=turn_asr_fn,
                )
            elif config.asr_mode == "chunk":
                chunk_asr_fn = asr_fn or transcribe_audio_chunk
                chunk_asr_results = _transcribe_chunks(
                    {
                        config.user_voice_name: user_chunks,
                        config.assistant_voice_name: assistant_chunks,
                    },
                    sample_dir,
                    chunk_asr_root,
                    cache_asr_dir,
                    config,
                    force=force,
                    asr_fn=chunk_asr_fn,
                )
            else:
                raise ValueError(f"Unsupported asr_mode: {config.asr_mode!r}")

        sample["chunk_ms"] = config.chunk_ms
        sample["num_chunks"] = num_chunks
        sample["chunk_targets"] = build_chunk_targets(
            turns,
            {
                config.user_voice_name: user_chunks,
                config.assistant_voice_name: assistant_chunks,
            },
            chunk_asr_results,
            turn_word_timings,
            config,
            num_chunks,
        )

    if config.sfx_enabled:
        sample = mix_sample_sfx(sample, sample_dir, config, force=force, catalog=sfx_catalog)

    validate_sample_json(sample, sample_dir)
    write_json(sample_dir / "metadata.json", {k: v for k, v in sample.items() if k != "chunk_targets"})
    write_json(sample_dir / "sample.json", sample)
    return sample


def _generate_turn_audio(
    turns: list[dict[str, Any]],
    config: PipelineConfig,
    tts_voices: dict[str, str],
    turn_audio_dir: Path,
    cache_tts_dir: Path,
    *,
    skip_tts: bool,
    force: bool,
    tts_fn: TTSFn,
) -> None:
    turn_audio_dir.mkdir(parents=True, exist_ok=True)
    if skip_tts:
        missing = [turn["turn_id"] for turn in turns if not (turn_audio_dir / f"{turn['turn_id']}.wav").exists()]
        if missing:
            raise FileNotFoundError(f"--skip-tts requires existing turn wav files; missing: {missing}")
        return

    def _work(turn: dict[str, Any]) -> None:
        voice = tts_voices[turn["stream"]]
        tts_fn(
            turn["message"],
            str(turn_audio_dir / f"{turn['turn_id']}.wav"),
            voice,
            config.tts_model,
            config.sample_rate,
            config.tts_response_format,
            speed=config.tts_speed,
            cache_dir=cache_tts_dir,
            max_retries=config.max_retries,
            force=force,
        )

    with ThreadPoolExecutor(max_workers=_tts_worker_count(config, len(turns))) as executor:
        futures = [executor.submit(_work, turn) for turn in turns]
        for future in as_completed(futures):
            future.result()


def _generate_backchannel_audio(
    backchannel_events: list[dict[str, Any]],
    config: PipelineConfig,
    tts_voices: dict[str, str],
    backchannel_audio_dir: Path,
    cache_tts_dir: Path,
    *,
    skip_tts: bool,
    force: bool,
    tts_fn: TTSFn,
) -> None:
    if not backchannel_events:
        return

    backchannel_audio_dir.mkdir(parents=True, exist_ok=True)
    if skip_tts:
        missing = [
            event["event_id"]
            for event in backchannel_events
            if not (backchannel_audio_dir / f"{event['event_id']}.wav").exists()
        ]
        if missing:
            raise FileNotFoundError(f"--skip-tts requires existing backchannel wav files; missing: {missing}")
        return

    def _work(event: dict[str, Any]) -> None:
        voice = tts_voices[event["stream"]]
        tts_fn(
            event["text"],
            str(backchannel_audio_dir / f"{event['event_id']}.wav"),
            voice,
            config.tts_model,
            config.sample_rate,
            config.tts_response_format,
            instructions="Say this as a brief, natural, low-key listener backchannel.",
            speed=config.tts_speed,
            cache_dir=cache_tts_dir,
            max_retries=config.max_retries,
            force=force,
        )

    with ThreadPoolExecutor(max_workers=_tts_worker_count(config, len(backchannel_events))) as executor:
        futures = [executor.submit(_work, event) for event in backchannel_events]
        for future in as_completed(futures):
            future.result()


def _transcribe_chunks(
    chunks_by_stream: dict[str, list[dict[str, Any]]],
    sample_dir: Path,
    asr_root: Path,
    cache_asr_dir: Path,
    config: PipelineConfig,
    *,
    force: bool,
    asr_fn: ASRFn,
) -> dict[str, dict[int, dict[str, Any]]]:
    results: dict[str, dict[int, dict[str, Any]]] = {stream: {} for stream in chunks_by_stream}

    def _work(stream: str, chunk: dict[str, Any]) -> tuple[str, int, dict[str, Any]]:
        out_path = asr_root / stream / f"chunk_{chunk['chunk_id']:06d}.json"
        if out_path.exists() and not force:
            return stream, chunk["chunk_id"], json.loads(out_path.read_text(encoding="utf-8"))
        chunk_path = sample_dir / chunk["audio_path"]
        try:
            result = asr_fn(
                str(chunk_path),
                config.asr_model,
                cache_dir=cache_asr_dir,
                max_retries=config.max_retries,
                force=force,
            )
        except Exception as exc:
            result = {
                "text": "",
                "words": [],
                "error": str(exc),
                "text_source": "asr_failed",
            }
        write_json(out_path, result)
        return stream, chunk["chunk_id"], result

    with ThreadPoolExecutor(max_workers=max(1, config.asr_concurrency)) as executor:
        futures = [
            executor.submit(_work, stream, chunk)
            for stream, chunks in chunks_by_stream.items()
            for chunk in chunks
        ]
        for future in as_completed(futures):
            stream, chunk_id, result = future.result()
            results[stream][chunk_id] = result

    return results


def _transcribe_turns(
    turns: list[dict[str, Any]],
    turn_audio_dir: Path,
    asr_root: Path,
    cache_asr_dir: Path,
    config: PipelineConfig,
    *,
    force: bool,
    asr_fn: ASRFn,
) -> dict[str, list[dict[str, Any]]]:
    results: dict[str, list[dict[str, Any]]] = {}

    def _work(turn: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        out_path = asr_root / f"{turn['turn_id']}.json"
        if out_path.exists() and not force:
            return turn["turn_id"], json.loads(out_path.read_text(encoding="utf-8"))
        turn_path = turn_audio_dir / f"{turn['turn_id']}.wav"
        try:
            result = asr_fn(
                str(turn_path),
                config.asr_model,
                prompt=turn["message"],
                cache_dir=cache_asr_dir,
                max_retries=config.max_retries,
                force=force,
            )
        except Exception as exc:
            result = {
                "text": "",
                "words": [],
                "error": str(exc),
                "text_source": "asr_failed",
            }
        write_json(out_path, result)
        return turn["turn_id"], result

    with ThreadPoolExecutor(max_workers=max(1, config.asr_concurrency)) as executor:
        futures = [executor.submit(_work, turn) for turn in turns]
        raw_results = {turn_id: result for turn_id, result in (future.result() for future in as_completed(futures))}

    by_turn = {turn["turn_id"]: turn for turn in turns}
    for turn_id, result in raw_results.items():
        words = _turn_asr_words_to_global(result, by_turn[turn_id])
        if words:
            results[turn_id] = words

    return results


def build_chunk_targets(
    turns: list[dict[str, Any]],
    chunks_by_stream: dict[str, list[dict[str, Any]]],
    asr_results: dict[str, dict[int, dict[str, Any]]],
    turn_word_timings: dict[str, list[dict[str, Any]]] | None,
    config: PipelineConfig,
    num_chunks: int,
) -> list[dict[str, Any]]:
    word_timings = {
        turn["turn_id"]: estimate_word_timings(turn["message"], int(turn["start_ms"]), int(turn["end_ms"]))
        for turn in turns
    }
    if turn_word_timings:
        word_timings.update(turn_word_timings)
    chunk_targets: list[dict[str, Any]] = []
    chunk_meta = {
        stream: {chunk["chunk_id"]: chunk for chunk in chunks}
        for stream, chunks in chunks_by_stream.items()
    }

    for chunk_id in range(num_chunks):
        start_ms = chunk_id * config.chunk_ms
        end_ms = start_ms + config.chunk_ms
        user_obj = _stream_chunk_target(
            turns,
            word_timings,
            config.user_voice_name,
            chunk_meta[config.user_voice_name][chunk_id],
            asr_results.get(config.user_voice_name, {}).get(chunk_id),
            config,
        )
        assistant_obj = _stream_chunk_target(
            turns,
            word_timings,
            config.assistant_voice_name,
            chunk_meta[config.assistant_voice_name][chunk_id],
            asr_results.get(config.assistant_voice_name, {}).get(chunk_id),
            config,
        )
        chunk_targets.append(
            {
                "chunk_id": chunk_id,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "input": {config.user_voice_name: user_obj},
                "output": {config.assistant_voice_name: assistant_obj},
            }
        )

    return chunk_targets


def _stream_chunk_target(
    turns: list[dict[str, Any]],
    word_timings: dict[str, list[dict[str, Any]]],
    stream: str,
    chunk: dict[str, Any],
    asr_result: dict[str, Any] | None,
    config: PipelineConfig,
) -> dict[str, Any]:
    start_ms = int(chunk["start_ms"])
    end_ms = int(chunk["end_ms"])
    overlaps = find_overlapping_turns(turns, stream, start_ms, end_ms)
    base = {
        "audio_ref": f"audio/{stream}.wav",
        "audio_start_ms": start_ms,
        "audio_end_ms": end_ms,
        "chunk_audio_ref": chunk["audio_path"],
    }
    if not overlaps:
        return {
            **base,
            "state": "silence",
            "text": "",
            "text_source": "none",
            "words": [],
            "turn_id": None,
            "full_turn_text": "",
        }

    turn = overlaps[0]
    overlapping_words = words_overlapping_chunk(word_timings[turn["turn_id"]], start_ms, end_ms)
    fallback_words = estimate_word_timings(turn["message"], int(turn["start_ms"]), int(turn["end_ms"]))
    fallback_words = words_overlapping_chunk(fallback_words, start_ms, end_ms)
    fallback_text = _join_tokens([word["word"] for word in fallback_words])
    timing_text = _join_tokens([word["word"] for word in overlapping_words])
    has_turn_asr_words = any(word.get("source") == "asr" for word in overlapping_words)
    text = ""
    text_source = "turn_overlap_fallback"
    words = fallback_words

    if asr_result:
        asr_text = str(asr_result.get("text", "")).strip()
        asr_words = normalize_asr_words(asr_result, start_ms, end_ms)
        if asr_text:
            text = asr_text
            text_source = asr_result.get("text_source") or "asr_chunk"
            words = asr_words or fallback_words
        elif asr_result.get("text_source") == "asr_failed" and not config.fallback_turn_overlap:
            text_source = "asr_failed"
            words = []
        elif config.fallback_turn_overlap:
            text = fallback_text
            text_source = "turn_overlap_fallback"
            words = fallback_words
    elif has_turn_asr_words:
        text = timing_text
        text_source = "asr_turn_words"
        words = overlapping_words
    elif config.fallback_turn_overlap:
        text = fallback_text
        text_source = "turn_overlap_fallback"
        words = fallback_words

    return {
        **base,
        "state": "present",
        "text": text,
        "text_source": text_source,
        "words": words,
        "turn_id": turn["turn_id"],
        "full_turn_text": turn["message"],
    }


def _join_tokens(tokens: list[str]) -> str:
    if not tokens:
        return ""
    if all(len(token) == 1 for token in tokens):
        return "".join(tokens)
    output = ""
    for token in tokens:
        if not output:
            output = token
        elif len(token) == 1 and not token.isalnum():
            output += token
        else:
            output += " " + token
    return output


def _turn_asr_words_to_global(asr_result: dict[str, Any], turn: dict[str, Any]) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    turn_start_ms = int(turn["start_ms"])
    turn_end_ms = int(turn["end_ms"])
    for raw_word in asr_result.get("words") or []:
        word = raw_word.get("word") or raw_word.get("text") or ""
        if not word:
            continue
        start = raw_word.get("start")
        end = raw_word.get("end")
        if start is None or end is None:
            continue
        start_ms = turn_start_ms + round(float(start) * 1000)
        end_ms = turn_start_ms + round(float(end) * 1000)
        start_ms = max(turn_start_ms, min(start_ms, turn_end_ms))
        end_ms = max(start_ms, min(end_ms, turn_end_ms))
        words.append(
            {
                "word": word,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "overlap_ratio": 1.0,
                "source": "asr",
            }
        )
    return words


def _select_backchannel_events(turns: list[dict[str, Any]], config: PipelineConfig) -> list[dict[str, Any]]:
    if not config.backchannel_enabled or config.backchannel_max_count <= 0 or not config.backchannel_phrases:
        return []

    candidates = [
        turn
        for turn in turns
        if int(turn["end_ms"]) - int(turn["start_ms"]) >= config.backchannel_min_turn_duration_ms
    ]
    if not candidates:
        return []

    selected = _spread_items(candidates, len(candidates))
    events: list[dict[str, Any]] = []
    ratios = (0.45, 0.62, 0.78)
    estimated_duration_ms = config.backchannel_max_duration_ms

    while len(events) < config.backchannel_max_count:
        made_progress = False
        for turn in selected:
            if len(events) >= config.backchannel_max_count:
                break

            speaker_stream = turn["stream"]
            listener_stream = (
                config.assistant_voice_name
                if speaker_stream == config.user_voice_name
                else config.user_voice_name
            )
            listener_agent = (
                config.assistant_agent
                if listener_stream == config.assistant_voice_name
                else config.user_agent
            )
            turn_start_ms = int(turn["start_ms"])
            turn_end_ms = int(turn["end_ms"])
            turn_duration_ms = turn_end_ms - turn_start_ms
            start_ms = _find_backchannel_start_ms(
                turns,
                events,
                listener_stream,
                turn_start_ms,
                turn_duration_ms,
                ratios,
                estimated_duration_ms,
                config.backchannel_min_start_offset_ms,
                config.backchannel_min_end_margin_ms,
            )
            if start_ms is None:
                continue

            phrase = config.backchannel_phrases[len(events) % len(config.backchannel_phrases)]
            event_id = f"bc_{len(events) + 1:06d}"
            events.append(
                {
                    "event_id": event_id,
                    "text": phrase,
                    "agent": listener_agent,
                    "stream": listener_stream,
                    "while_turn_id": turn["turn_id"],
                    "while_turn_stream": speaker_stream,
                    "while_turn_start_ms": turn_start_ms,
                    "while_turn_end_ms": turn_end_ms,
                    "start_ms": start_ms,
                    "end_ms": None,
                    "duration_ms": None,
                    "audio_path": f"audio/backchannels/{event_id}.wav",
                }
            )
            made_progress = True

        if not made_progress:
            break

    return events


def _spread_items(items: list[dict[str, Any]], target_count: int) -> list[dict[str, Any]]:
    if target_count <= 0:
        return []
    if target_count == 1:
        return [items[len(items) // 2]]

    selected: list[dict[str, Any]] = []
    last_slot = len(items) - 1
    for slot in range(target_count):
        item = items[round(slot * last_slot / (target_count - 1))]
        if item not in selected:
            selected.append(item)
    return selected


def _find_backchannel_start_ms(
    turns: list[dict[str, Any]],
    existing_events: list[dict[str, Any]],
    listener_stream: str,
    turn_start_ms: int,
    turn_duration_ms: int,
    ratios: tuple[float, ...],
    estimated_duration_ms: int,
    min_start_offset_ms: int,
    min_end_margin_ms: int,
) -> int | None:
    earliest_ms = turn_start_ms + min_start_offset_ms
    latest_ms = turn_start_ms + turn_duration_ms - estimated_duration_ms - min_end_margin_ms
    if latest_ms < earliest_ms:
        return None

    for ratio in ratios:
        start_ms = turn_start_ms + round(turn_duration_ms * ratio)
        start_ms = min(max(start_ms, earliest_ms), latest_ms)
        end_ms = start_ms + estimated_duration_ms
        if _stream_has_formal_speech(turns, listener_stream, start_ms, end_ms):
            continue
        if _stream_has_backchannel(existing_events, listener_stream, start_ms, end_ms, estimated_duration_ms):
            continue
        return start_ms
    return None


def _stream_has_backchannel(
    events: list[dict[str, Any]],
    stream: str,
    start_ms: int,
    end_ms: int,
    estimated_duration_ms: int,
) -> bool:
    for event in events:
        if event.get("stream") != stream:
            continue
        event_start_ms = int(event["start_ms"])
        event_end_ms = int(event.get("end_ms") or event_start_ms + estimated_duration_ms)
        if max(0, min(event_end_ms + 120, end_ms) - max(event_start_ms - 120, start_ms)) > 0:
            return True
    return False


def _stream_has_formal_speech(
    turns: list[dict[str, Any]],
    stream: str,
    start_ms: int,
    end_ms: int,
) -> bool:
    for turn in turns:
        if turn.get("stream") != stream:
            continue
        if max(0, min(int(turn["end_ms"]), end_ms) - max(int(turn["start_ms"]), start_ms)) > 0:
            return True
    return False


def _build_overlap_events(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    by_id = {turn["turn_id"]: turn for turn in turns}
    for turn in turns:
        previous_turn_id = turn.get("overlap_previous_turn_id")
        if not previous_turn_id:
            continue
        previous = by_id[previous_turn_id]
        events.append(
            {
                "turn_id": turn["turn_id"],
                "overlaps_turn_id": previous_turn_id,
                "start_ms": turn["start_ms"],
                "overlap_start_ms": turn["start_ms"],
                "overlap_end_ms": previous["end_ms"],
                "overlap_ms": turn.get("overlap_ms", 0),
            }
        )
    return events


def _serialize_turn(turn: dict[str, Any]) -> dict[str, Any]:
    return {
        "turn_id": turn["turn_id"],
        "source_turn_index": turn["source_turn_index"],
        "agent": turn["agent"],
        "stream": turn["stream"],
        "start_ms": turn["start_ms"],
        "end_ms": turn["end_ms"],
        "text": turn["message"],
        "overlap_with_previous": turn.get("overlap_with_previous", False),
        "overlap_previous_turn_id": turn.get("overlap_previous_turn_id"),
        "overlap_ms": turn.get("overlap_ms", 0),
        "sentiment": turn.get("sentiment"),
        "source_metadata": turn.get("source_metadata", {}),
    }


def _manifest_row(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": sample["sample_id"],
        "source_dialogue_id": sample["source_dialogue_id"],
        "sample_json": f"samples/{sample['sample_id']}/sample.json",
        "duration_ms": sample["duration_ms"],
        "num_turns": len(sample.get("turns", [])),
        "num_chunks": sample.get("num_chunks", 0),
        "num_overlaps": len(sample.get("overlap_events", [])),
        "num_backchannels": len(sample.get("backchannel_events", [])),
        "num_sfx_events": len(sample.get("sfx_events", [])),
    }
