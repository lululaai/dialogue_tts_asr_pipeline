from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .audio_utils import is_silence, normalize_wav_file


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dump_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _tts_cache_key(
    text: str,
    voice: str,
    model: str,
    sample_rate: int,
    response_format: str,
    instructions: str | None,
) -> str:
    return sha256_text(
        json.dumps(
            {
                "model": model,
                "voice": voice,
                "response_format": response_format,
                "sample_rate": sample_rate,
                "instructions": instructions,
                "text": text,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def _asr_cache_key(chunk_wav_path: str | Path, model: str, prompt: str | None) -> str:
    digest = hashlib.sha256()
    digest.update(model.encode("utf-8"))
    digest.update((prompt or "").encode("utf-8"))
    with Path(chunk_wav_path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _response_to_bytes(response: Any) -> bytes:
    if isinstance(response, bytes):
        return response
    if hasattr(response, "read"):
        return response.read()
    if hasattr(response, "content"):
        return response.content
    raise TypeError(f"Unsupported speech response object: {type(response)!r}")


def _response_to_dict(response: Any) -> dict[str, Any]:
    if isinstance(response, dict):
        return response
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if hasattr(response, "dict"):
        return response.dict()
    text = getattr(response, "text", "")
    return {"text": text, "words": []}


def synthesize_tts(
    text: str,
    output_wav_path: str,
    voice: str,
    model: str,
    sample_rate: int,
    response_format: str = "wav",
    instructions: str | None = None,
    *,
    cache_dir: str | Path | None = None,
    max_retries: int = 5,
    force: bool = False,
) -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for TTS generation")

    output_path = Path(output_wav_path)
    metadata_path = output_path.with_suffix(output_path.suffix + ".json")
    cache_key = _tts_cache_key(text, voice, model, sample_rate, response_format, instructions)
    metadata = {
        "text": text,
        "voice": voice,
        "model": model,
        "sample_rate": sample_rate,
        "response_format": response_format,
        "instructions": instructions,
        "sha256_text": sha256_text(text),
        "cache_key": cache_key,
    }

    if not force and output_path.exists() and _load_json(metadata_path) == metadata:
        return

    cache_wav: Path | None = None
    cache_meta: Path | None = None
    if cache_dir is not None:
        cache_root = Path(cache_dir)
        cache_wav = cache_root / f"{cache_key}.wav"
        cache_meta = cache_root / f"{cache_key}.json"
        if not force and cache_wav.exists() and _load_json(cache_meta) == metadata:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cache_wav, output_path)
            _dump_json(metadata_path, metadata)
            return

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        stop=stop_after_attempt(max_retries),
        reraise=True,
    )
    def _call_openai() -> bytes:
        from openai import OpenAI

        client = OpenAI()
        kwargs: dict[str, Any] = {
            "model": model,
            "voice": voice,
            "input": text,
            "response_format": response_format,
        }
        if instructions:
            kwargs["instructions"] = instructions
        response = client.audio.speech.create(**kwargs)
        return _response_to_bytes(response)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(f".raw.{response_format}")
    temp_path.write_bytes(_call_openai())
    normalize_wav_file(temp_path, output_path, sample_rate)
    temp_path.unlink(missing_ok=True)
    _dump_json(metadata_path, metadata)

    if cache_wav is not None and cache_meta is not None:
        cache_wav.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output_path, cache_wav)
        _dump_json(cache_meta, metadata)


def transcribe_audio_chunk(
    chunk_wav_path: str,
    model: str,
    prompt: str | None = None,
    *,
    cache_dir: str | Path | None = None,
    max_retries: int = 5,
    force: bool = False,
) -> dict[str, Any]:
    chunk_path = Path(chunk_wav_path)
    if is_silence(chunk_path):
        return {"text": "", "words": [], "skipped": True, "reason": "silence", "text_source": "none"}

    cache_key = _asr_cache_key(chunk_path, model, prompt)
    cache_path = Path(cache_dir) / f"{cache_key}.json" if cache_dir is not None else None
    if cache_path is not None and not force:
        cached = _load_json(cache_path)
        if cached is not None:
            return cached

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for ASR")

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        stop=stop_after_attempt(max_retries),
        reraise=True,
    )
    def _call_openai() -> dict[str, Any]:
        from openai import OpenAI

        client = OpenAI()
        with chunk_path.open("rb") as audio_file:
            kwargs: dict[str, Any] = {
                "model": model,
                "file": audio_file,
                "response_format": "json",
            }
            if prompt:
                kwargs["prompt"] = prompt
            response = client.audio.transcriptions.create(**kwargs)
        data = _response_to_dict(response)
        data.setdefault("text", "")
        data.setdefault("words", [])
        data.setdefault("text_source", "asr_chunk")
        return data

    result = _call_openai()
    if cache_path is not None:
        _dump_json(cache_path, result)
    return result

