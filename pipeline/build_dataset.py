from __future__ import annotations

import json
import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from .align_chunks import (
    estimate_word_timings,
    find_overlapping_turns,
    normalize_asr_words,
    split_audio_to_chunks,
    words_overlapping_chunk,
)
from .audio_utils import build_stream_tracks, ensure_ffmpeg_available
from .config import PipelineConfig
from .load_dialogues import load_dialogues
from .openai_audio import (
    synthesize_tts,
    transcribe_audio_chunk,
    transcribe_audio_turn_with_word_timestamps,
)
from .schemas import validate_sample_json

LOGGER = logging.getLogger(__name__)

TTSFn = Callable[..., None]
ASRFn = Callable[..., dict[str, Any]]


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(data, ensure_ascii=False) + "\n")


def run_pipeline(
    config: PipelineConfig,
    *,
    limit: int | None = None,
    resume: bool = False,
    skip_tts: bool = False,
    skip_asr: bool = False,
    force: bool = False,
    tts_fn: TTSFn = synthesize_tts,
    asr_fn: ASRFn | None = None,
) -> list[dict[str, Any]]:
    ensure_ffmpeg_available()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    failed_path = output_dir / "failed.jsonl"

    if not resume:
        manifest_path.unlink(missing_ok=True)
        failed_path.unlink(missing_ok=True)

    dialogues = load_dialogues(config.input_json, config)
    if limit is not None:
        dialogues = dialogues[:limit]

    manifest_rows: list[dict[str, Any]] = []
    for dialogue in dialogues:
        sample_dir = output_dir / "samples" / dialogue["sample_id"]
        sample_json_path = sample_dir / "sample.json"
        if resume and not force and sample_json_path.exists():
            sample = json.loads(sample_json_path.read_text(encoding="utf-8"))
            row = _manifest_row(sample)
            append_jsonl(manifest_path, row)
            manifest_rows.append(row)
            LOGGER.info("Skipping completed sample %s", dialogue["sample_id"])
            continue

        try:
            sample = build_sample(
                dialogue,
                config,
                sample_dir,
                skip_tts=skip_tts,
                skip_asr=skip_asr,
                force=force,
                tts_fn=tts_fn,
                asr_fn=asr_fn,
            )
            row = _manifest_row(sample)
            append_jsonl(manifest_path, row)
            manifest_rows.append(row)
        except Exception as exc:
            LOGGER.exception("Failed to process %s", dialogue["sample_id"])
            append_jsonl(
                failed_path,
                {
                    "sample_id": dialogue["sample_id"],
                    "source_dialogue_id": dialogue["source_dialogue_id"],
                    "error": str(exc),
                },
            )

    return manifest_rows


def build_sample(
    dialogue: dict[str, Any],
    config: PipelineConfig,
    sample_dir: str | Path,
    *,
    skip_tts: bool = False,
    skip_asr: bool = False,
    force: bool = False,
    tts_fn: TTSFn = synthesize_tts,
    asr_fn: ASRFn | None = None,
) -> dict[str, Any]:
    sample_dir = Path(sample_dir)
    turn_audio_dir = sample_dir / "audio" / "turns"
    audio_dir = sample_dir / "audio"
    chunk_root = audio_dir / "chunks"
    chunk_asr_root = sample_dir / "asr" / "chunks"
    turn_asr_root = sample_dir / "asr" / "turns"
    cache_tts_dir = Path(config.output_dir) / "cache" / "tts"
    cache_asr_dir = Path(config.output_dir) / "cache" / "asr"
    turns = [dict(turn) for turn in dialogue["turns"]]

    _generate_turn_audio(
        turns,
        config,
        turn_audio_dir,
        cache_tts_dir,
        skip_tts=skip_tts,
        force=force,
        tts_fn=tts_fn,
    )

    duration_ms = build_stream_tracks(turns, turn_audio_dir, audio_dir, config)
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

    chunk_targets = build_chunk_targets(
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

    sample = {
        "sample_id": dialogue["sample_id"],
        "source_dialogue_id": dialogue["source_dialogue_id"],
        "chunk_ms": config.chunk_ms,
        "sample_rate": config.sample_rate,
        "duration_ms": duration_ms,
        "num_chunks": num_chunks,
        "streams": {
            "input": [config.user_voice_name],
            "output": [config.assistant_voice_name],
        },
        "audio_files": {
            config.user_voice_name: f"audio/{config.user_voice_name}.wav",
            config.assistant_voice_name: f"audio/{config.assistant_voice_name}.wav",
        },
        "turns": [_serialize_turn(turn) for turn in turns],
        "source_metadata": dialogue.get("source_metadata", {}),
        "chunk_targets": chunk_targets,
    }

    validate_sample_json(sample, sample_dir)
    write_json(sample_dir / "metadata.json", {k: v for k, v in sample.items() if k != "chunk_targets"})
    write_json(sample_dir / "sample.json", sample)
    return sample


def _generate_turn_audio(
    turns: list[dict[str, Any]],
    config: PipelineConfig,
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
        voice = config.user_tts_voice if turn["stream"] == config.user_voice_name else config.assistant_tts_voice
        tts_fn(
            turn["message"],
            str(turn_audio_dir / f"{turn['turn_id']}.wav"),
            voice,
            config.tts_model,
            config.sample_rate,
            config.tts_response_format,
            cache_dir=cache_tts_dir,
            max_retries=config.max_retries,
            force=force,
        )

    with ThreadPoolExecutor(max_workers=max(1, config.tts_concurrency)) as executor:
        futures = [executor.submit(_work, turn) for turn in turns]
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


def _serialize_turn(turn: dict[str, Any]) -> dict[str, Any]:
    return {
        "turn_id": turn["turn_id"],
        "source_turn_index": turn["source_turn_index"],
        "agent": turn["agent"],
        "stream": turn["stream"],
        "start_ms": turn["start_ms"],
        "end_ms": turn["end_ms"],
        "text": turn["message"],
        "sentiment": turn.get("sentiment"),
        "source_metadata": turn.get("source_metadata", {}),
    }


def _manifest_row(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": sample["sample_id"],
        "source_dialogue_id": sample["source_dialogue_id"],
        "sample_json": f"samples/{sample['sample_id']}/sample.json",
        "duration_ms": sample["duration_ms"],
        "num_chunks": sample["num_chunks"],
    }
