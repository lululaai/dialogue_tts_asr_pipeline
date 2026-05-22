from __future__ import annotations

from pathlib import Path
from typing import Any


def validate_sample_json(sample: dict[str, Any], sample_dir: str | Path | None = None) -> None:
    required = {
        "sample_id",
        "source_dialogue_id",
        "sample_rate",
        "duration_ms",
        "streams",
        "audio_files",
        "turns",
        "overlap_events",
        "backchannel_events",
    }
    missing = required - set(sample)
    if missing:
        raise ValueError(f"sample is missing required fields: {sorted(missing)}")
    validate_turns(sample)
    validate_overlap_events(sample)
    validate_backchannel_events(sample)
    if "chunk_targets" in sample:
        validate_chunk_targets(sample)
    if sample_dir is not None:
        base = Path(sample_dir)
        for key, rel_path in sample["audio_files"].items():
            if not (base / rel_path).exists():
                raise ValueError(f"audio_files.{key} does not exist: {rel_path}")


def validate_turns(sample: dict[str, Any]) -> None:
    for index, turn in enumerate(sample["turns"]):
        for key in ("turn_id", "stream", "start_ms", "end_ms", "text"):
            if key not in turn:
                raise ValueError(f"turn {index} missing {key}")
        if turn["end_ms"] < turn["start_ms"]:
            raise ValueError(f"turn {index} ends before it starts")


def validate_overlap_events(sample: dict[str, Any]) -> None:
    turns = {turn["turn_id"]: turn for turn in sample["turns"]}
    for index, event in enumerate(sample["overlap_events"]):
        for key in ("turn_id", "overlaps_turn_id", "overlap_start_ms", "overlap_end_ms", "overlap_ms"):
            if key not in event:
                raise ValueError(f"overlap event {index} missing {key}")
        if event["turn_id"] not in turns:
            raise ValueError(f"overlap event {index} references unknown turn_id")
        if event["overlaps_turn_id"] not in turns:
            raise ValueError(f"overlap event {index} references unknown overlaps_turn_id")
        if event["overlap_ms"] <= 0:
            raise ValueError(f"overlap event {index} must have positive overlap_ms")
        if event["overlap_end_ms"] <= event["overlap_start_ms"]:
            raise ValueError(f"overlap event {index} has invalid overlap range")


def validate_backchannel_events(sample: dict[str, Any]) -> None:
    turns = {turn["turn_id"]: turn for turn in sample["turns"]}
    for index, event in enumerate(sample["backchannel_events"]):
        for key in ("event_id", "text", "stream", "while_turn_id", "start_ms", "end_ms", "duration_ms", "audio_path"):
            if key not in event:
                raise ValueError(f"backchannel event {index} missing {key}")
        if event["while_turn_id"] not in turns:
            raise ValueError(f"backchannel event {index} references unknown while_turn_id")
        if not event["text"]:
            raise ValueError(f"backchannel event {index} has empty text")
        if event["duration_ms"] is None or event["duration_ms"] <= 0:
            raise ValueError(f"backchannel event {index} must have positive duration_ms")
        if event["end_ms"] <= event["start_ms"]:
            raise ValueError(f"backchannel event {index} has invalid time range")


def validate_chunk_targets(sample: dict[str, Any]) -> None:
    for key in ("chunk_ms", "num_chunks"):
        if key not in sample:
            raise ValueError(f"sample with chunk_targets is missing {key}")
    chunk_ms = sample["chunk_ms"]
    chunk_targets = sample["chunk_targets"]
    if sample["num_chunks"] != len(chunk_targets):
        raise ValueError("num_chunks must equal len(chunk_targets)")

    for expected_id, chunk in enumerate(chunk_targets):
        for key in ("chunk_id", "start_ms", "end_ms", "input", "output"):
            if key not in chunk:
                raise ValueError(f"chunk {expected_id} missing {key}")
        if chunk["chunk_id"] != expected_id:
            raise ValueError(f"chunk_id mismatch at {expected_id}")
        if chunk["start_ms"] != expected_id * chunk_ms:
            raise ValueError(f"chunk {expected_id} has invalid start_ms")
        if chunk["end_ms"] != chunk["start_ms"] + chunk_ms:
            raise ValueError(f"chunk {expected_id} has invalid end_ms")
        if "user_voice" not in chunk["input"]:
            raise ValueError(f"chunk {expected_id} missing input.user_voice")
        if "assistant_voice" not in chunk["output"]:
            raise ValueError(f"chunk {expected_id} missing output.assistant_voice")
        _validate_stream_obj(chunk["input"]["user_voice"], expected_id, "input.user_voice")
        _validate_stream_obj(chunk["output"]["assistant_voice"], expected_id, "output.assistant_voice")


def _validate_stream_obj(stream_obj: dict[str, Any], chunk_id: int, path: str) -> None:
    required = {
        "state",
        "audio_ref",
        "audio_start_ms",
        "audio_end_ms",
        "chunk_audio_ref",
        "text",
        "text_source",
        "words",
        "turn_id",
        "full_turn_text",
    }
    missing = required - set(stream_obj)
    if missing:
        raise ValueError(f"chunk {chunk_id} {path} missing {sorted(missing)}")
    if stream_obj["state"] not in {"present", "silence"}:
        raise ValueError(f"chunk {chunk_id} {path} has invalid state")
    if not isinstance(stream_obj["words"], list):
        raise ValueError(f"chunk {chunk_id} {path}.words must be a list")
    if stream_obj["state"] == "silence":
        if stream_obj["text"] != "" or stream_obj["words"] != [] or stream_obj["turn_id"] is not None:
            raise ValueError(f"chunk {chunk_id} {path} silence object contains speech data")
    if stream_obj["state"] == "present":
        if stream_obj["turn_id"] is None or not stream_obj["full_turn_text"]:
            raise ValueError(f"chunk {chunk_id} {path} present object missing turn metadata")
