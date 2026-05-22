from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import wave
from base64 import b64decode
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import google.auth
import google.auth.transport.requests
import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .audio_utils import normalize_wav_file

_TOKEN_LOCK = threading.Lock()
_CACHED_TOKEN: str | None = None
_CACHED_TOKEN_EXPIRY: datetime | None = None


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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
    speed: float,
    model: str,
    sample_rate: int,
    response_format: str,
    instructions: str | None,
    location: str,
) -> str:
    return sha256_text(
        json.dumps(
            {
                "provider": "google",
                "model": model,
                "voice": voice,
                "speed": speed,
                "response_format": response_format,
                "sample_rate": sample_rate,
                "instructions": instructions,
                "location": location,
                "text": text,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def _resolve_project_id() -> str:
    project_id = (
        os.getenv("GOOGLE_CLOUD_PROJECT")
        or os.getenv("GOOGLE_PROJECT_ID")
        or os.getenv("GCLOUD_PROJECT")
    )
    if project_id:
        return project_id

    _, default_project_id = google.auth.default()
    if default_project_id:
        return default_project_id

    adc_path = Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
    adc = _load_json(adc_path)
    quota_project_id = adc.get("quota_project_id") if adc else None
    if isinstance(quota_project_id, str) and quota_project_id:
        return quota_project_id

    raise RuntimeError(
        "GOOGLE_CLOUD_PROJECT is required for Google TTS. "
        "Set it to your quota project, for example: "
        "export GOOGLE_CLOUD_PROJECT=YOUR_PROJECT_ID"
    )


def _get_access_token() -> str:
    global _CACHED_TOKEN, _CACHED_TOKEN_EXPIRY

    now = datetime.now(timezone.utc)
    if _CACHED_TOKEN and _CACHED_TOKEN_EXPIRY and _CACHED_TOKEN_EXPIRY - now > timedelta(minutes=5):
        return _CACHED_TOKEN

    with _TOKEN_LOCK:
        now = datetime.now(timezone.utc)
        if _CACHED_TOKEN and _CACHED_TOKEN_EXPIRY and _CACHED_TOKEN_EXPIRY - now > timedelta(minutes=5):
            return _CACHED_TOKEN

        credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        credentials.refresh(google.auth.transport.requests.Request())
        if not credentials.token:
            raise RuntimeError("Google ADC refresh did not return an access token")

        expiry = credentials.expiry
        if expiry is None:
            expiry = now + timedelta(minutes=30)
        elif expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=timezone.utc)

        _CACHED_TOKEN = credentials.token
        _CACHED_TOKEN_EXPIRY = expiry
        return _CACHED_TOKEN


def _build_contents(text: str, instructions: str | None, speed: float) -> str:
    style = instructions or "Say the following exactly."
    if speed != 1.0:
        style = f"{style} Use approximately {speed:g}x normal speaking pace."
    return f"{style}\n\n{text}"


def _write_pcm_wav(path: Path, pcm: bytes, *, sample_rate: int = 24000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)


def synthesize_tts(
    text: str,
    output_wav_path: str,
    voice: str,
    model: str,
    sample_rate: int,
    response_format: str = "wav",
    instructions: str | None = None,
    speed: float = 1.0,
    *,
    cache_dir: str | Path | None = None,
    max_retries: int = 5,
    force: bool = False,
) -> None:
    if response_format != "wav":
        raise ValueError("Google Gemini TTS currently writes wav output in this pipeline")

    output_path = Path(output_wav_path)
    metadata_path = output_path.with_suffix(output_path.suffix + ".json")
    location = os.getenv("GOOGLE_CLOUD_REGION", "us-central1")
    cache_key = _tts_cache_key(text, voice, speed, model, sample_rate, response_format, instructions, location)
    metadata = {
        "provider": "google",
        "text": text,
        "voice": voice,
        "speed": speed,
        "model": model,
        "sample_rate": sample_rate,
        "response_format": response_format,
        "instructions": instructions,
        "location": location,
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
    def _call_google() -> bytes:
        project_id = _resolve_project_id()
        access_token = _get_access_token()
        endpoint = (
            f"https://aiplatform.googleapis.com/v1beta1/projects/{project_id}"
            f"/locations/{location}/publishers/google/models/{model}:generateContent"
        )
        response = requests.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {access_token}",
                "x-goog-user-project": project_id,
                "Content-Type": "application/json",
            },
            json={
                "contents": {
                    "role": "user",
                    "parts": {
                        "text": _build_contents(text, instructions, speed),
                    },
                },
                "generation_config": {
                    "speech_config": {
                        "language_code": os.getenv("GOOGLE_TTS_LANGUAGE_CODE", "en-US"),
                        "voice_config": {
                            "prebuilt_voice_config": {
                                "voice_name": voice,
                            },
                        },
                    },
                    "temperature": 2.0,
                },
            },
            timeout=120,
        )
        if not response.ok:
            raise RuntimeError(f"Google TTS request failed: {response.status_code} {response.text}")
        data = response.json()
        encoded_audio = data["candidates"][0]["content"]["parts"][0]["inlineData"]["data"]
        return b64decode(encoded_audio)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(".raw_google.wav")
    _write_pcm_wav(temp_path, _call_google())
    normalize_wav_file(temp_path, output_path, sample_rate)
    temp_path.unlink(missing_ok=True)
    _dump_json(metadata_path, metadata)

    if cache_wav is not None and cache_meta is not None:
        cache_wav.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output_path, cache_wav)
        _dump_json(cache_meta, metadata)
