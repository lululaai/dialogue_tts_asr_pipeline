from __future__ import annotations

from pathlib import Path
from typing import Any


def validate_sample_json(sample: dict[str, Any], sample_dir: str | Path | None = None) -> None:
    required = {
        "sample_id",
        "source_dialogue_id",
        "chunk_ms",
        "sample_rate",
        "duration_ms",
        "num_chunks",
        "streams",
        "audio_files",
        "turns",
        "chunk_targets",
    }
    missing = required - set(sample)
    if missing:
        raise ValueError(f"sample is missing required fields: {sorted(missing)}")
    validate_chunk_targets(sample)
    if sample_dir is not None:
        base = Path(sample_dir)
        for key, rel_path in sample["audio_files"].items():
            if not (base / rel_path).exists():
                raise ValueError(f"audio_files.{key} does not exist: {rel_path}")


def validate_chunk_targets(sample: dict[str, Any]) -> None:
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

