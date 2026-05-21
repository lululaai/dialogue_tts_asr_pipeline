from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .config import PipelineConfig

LOGGER = logging.getLogger(__name__)


SUPPORTED_TURN_KEYS = {"message", "agent"}


def load_dialogues(input_json: str | Path, config: PipelineConfig) -> list[dict[str, Any]]:
    path = Path(input_json)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"Unable to read input JSON {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError(f"Input JSON must be a top-level object: {path}")

    dialogues: list[dict[str, Any]] = []
    for sample_index, source_dialogue_id in enumerate(sorted(raw), start=1):
        dialogue = raw[source_dialogue_id]
        if not isinstance(dialogue, dict):
            LOGGER.warning("Skipping %s: dialogue value is not an object", source_dialogue_id)
            continue

        content = dialogue.get("content")
        if not isinstance(content, list):
            LOGGER.warning("Skipping %s: missing or invalid content list", source_dialogue_id)
            continue

        counters = {config.user_agent: 0, config.assistant_agent: 0}
        turns: list[dict[str, Any]] = []
        for source_turn_index, item in enumerate(content):
            if not isinstance(item, dict):
                LOGGER.warning(
                    "Skipping turn %s in %s: turn is not an object",
                    source_turn_index,
                    source_dialogue_id,
                )
                continue

            message = str(item.get("message", "")).strip()
            agent = item.get("agent")
            if not message:
                LOGGER.warning(
                    "Skipping turn %s in %s: empty message",
                    source_turn_index,
                    source_dialogue_id,
                )
                continue
            if not agent:
                raise ValueError(
                    f"Turn {source_turn_index} in {source_dialogue_id} is missing required field 'agent'"
                )
            if agent not in config.agent_to_stream:
                LOGGER.warning(
                    "Skipping turn %s in %s: unsupported agent %r",
                    source_turn_index,
                    source_dialogue_id,
                    agent,
                )
                continue

            counters[agent] += 1
            prefix = "u" if agent == config.user_agent else "a"
            metadata = {k: v for k, v in item.items() if k not in SUPPORTED_TURN_KEYS}
            turn: dict[str, Any] = {
                "turn_id": f"{prefix}{counters[agent]}",
                "source_turn_index": source_turn_index,
                "agent": agent,
                "stream": config.agent_to_stream[agent],
                "message": message,
                "sentiment": item.get("sentiment"),
                "start_ms": None,
                "end_ms": None,
                "source_metadata": metadata,
            }
            turns.append(turn)

        if not turns:
            LOGGER.warning("Skipping %s: no valid turns", source_dialogue_id)
            continue

        dialogues.append(
            {
                "sample_id": f"dialogue_{sample_index:06d}",
                "source_dialogue_id": source_dialogue_id,
                "source_metadata": {
                    key: value for key, value in dialogue.items() if key != "content"
                },
                "turns": turns,
            }
        )

    return dialogues

