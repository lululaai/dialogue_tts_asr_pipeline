from __future__ import annotations

import json
import os
from typing import Any

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .google_audio import _get_access_token, _resolve_project_id
from .google_limiter import google_request_slot


def generate_json(
    prompt: str,
    *,
    model: str,
    temperature: float = 0.2,
    max_retries: int = 5,
) -> dict[str, Any]:
    location = os.getenv("GOOGLE_CLOUD_REGION", "us-central1")

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        stop=stop_after_attempt(max_retries),
        reraise=True,
    )
    def _call_google() -> dict[str, Any]:
        project_id = _resolve_project_id()
        access_token = _get_access_token()
        endpoint = (
            f"https://aiplatform.googleapis.com/v1beta1/projects/{project_id}"
            f"/locations/{location}/publishers/google/models/{model}:generateContent"
        )
        with google_request_slot():
            response = requests.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "x-goog-user-project": project_id,
                    "Content-Type": "application/json",
                },
                json={
                    "contents": [
                        {
                            "role": "user",
                            "parts": [{"text": prompt}],
                        }
                    ],
                    "generationConfig": {
                        "temperature": temperature,
                        "responseMimeType": "application/json",
                    },
                },
                timeout=120,
            )
        if not response.ok:
            raise RuntimeError(f"Google text request failed: {response.status_code} {response.text}")
        text = _extract_text(response.json())
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Google text response was not valid JSON: {text[:1000]}") from exc

    return _call_google()


def _extract_text(payload: dict[str, Any]) -> str:
    candidates = payload.get("candidates") or []
    if not candidates:
        raise RuntimeError(f"Google text response had no candidates: {payload}")
    parts = ((candidates[0].get("content") or {}).get("parts") or [])
    texts = [str(part.get("text") or "") for part in parts if part.get("text")]
    if not texts:
        raise RuntimeError(f"Google text response had no text parts: {payload}")
    return "\n".join(texts).strip()
