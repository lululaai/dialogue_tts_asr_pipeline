from __future__ import annotations

from contextlib import nullcontext

from pipeline import google_text


class FakeResponse:
    ok = True
    status_code = 200
    text = ""

    def json(self) -> dict:
        return {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": '{"ok": true}',
                            }
                        ]
                    }
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 12,
                "cachedContentTokenCount": 5,
            },
        }


def test_generate_json_sends_optional_system_instruction(monkeypatch):
    captured = {}

    def fake_post(endpoint, *, headers, json, timeout):
        captured["endpoint"] = endpoint
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(google_text, "_resolve_project_id", lambda: "project-1")
    monkeypatch.setattr(google_text, "_get_access_token", lambda: "token-1")
    monkeypatch.setattr(google_text, "google_request_slot", lambda: nullcontext())
    monkeypatch.setattr(google_text.requests, "post", fake_post)

    result = google_text.generate_json(
        "user payload",
        model="gemini-test",
        system_instruction="stable rules",
        temperature=0.1,
        max_retries=1,
    )

    assert result == {"ok": True}
    assert captured["json"]["contents"] == [
        {
            "role": "user",
            "parts": [{"text": "user payload"}],
        }
    ]
    assert captured["json"]["systemInstruction"] == {
        "parts": [{"text": "stable rules"}],
    }
    assert captured["json"]["generationConfig"] == {
        "temperature": 0.1,
        "responseMimeType": "application/json",
    }


def test_generate_json_omits_system_instruction_when_empty(monkeypatch):
    captured = {}

    def fake_post(endpoint, *, headers, json, timeout):
        captured["json"] = json
        return FakeResponse()

    monkeypatch.setattr(google_text, "_resolve_project_id", lambda: "project-1")
    monkeypatch.setattr(google_text, "_get_access_token", lambda: "token-1")
    monkeypatch.setattr(google_text, "google_request_slot", lambda: nullcontext())
    monkeypatch.setattr(google_text.requests, "post", fake_post)

    assert google_text.generate_json("user payload", model="gemini-test", max_retries=1) == {"ok": True}
    assert "systemInstruction" not in captured["json"]
