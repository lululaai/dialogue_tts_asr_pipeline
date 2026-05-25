from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydub import AudioSegment

from .config import PipelineConfig
from .google_text import generate_json


@dataclass(frozen=True)
class SfxAsset:
    category: str
    label: str
    item: str
    path: Path


@dataclass
class SfxEvent:
    event_id: str
    category: str
    label: str
    start_ms: int
    duration_ms: int
    gain_db: float
    ducking_db: float
    reason: str
    asset_path: str | None = None
    end_ms: int | None = None


SfxCatalog = dict[tuple[str, str], list[SfxAsset]]


def mix_sample_sfx(
    sample: dict[str, Any],
    sample_dir: str | Path,
    config: PipelineConfig,
    *,
    force: bool = False,
    catalog: SfxCatalog | None = None,
) -> dict[str, Any]:
    sample_dir = Path(sample_dir)
    output_rel = f"audio/{config.sfx_audio_name}.wav"
    output_path = sample_dir / output_rel
    if not force and output_path.exists() and sample.get("sfx_events"):
        sample["audio_files"][config.sfx_audio_name] = output_rel
        return sample

    catalog = catalog or load_sfx_catalog(config)
    if not catalog:
        raise FileNotFoundError(
            f"No local SFX files from {config.sfx_map_path!r} were found under {config.sfx_root!r}."
        )

    plan = _plan_sfx_events(sample, catalog, config, force=force)
    events = _sanitize_plan(plan, sample, catalog, config)
    events = _select_assets(events, catalog, sample["sample_id"], config)
    _render_mix(sample, sample_dir, events, config, output_path)

    sample["audio_files"][config.sfx_audio_name] = output_rel
    sample["sfx"] = {
        "provider": "google",
        "model": config.sfx_planner_model,
        "map_path": config.sfx_map_path,
        "root": config.sfx_root,
    }
    sample["sfx_events"] = [_event_dict(event, sample_dir) for event in events]
    return sample


def load_sfx_catalog(config: PipelineConfig) -> SfxCatalog:
    map_path = Path(config.sfx_map_path)
    if not map_path.is_absolute():
        map_path = Path.cwd() / map_path
    data = json.loads(map_path.read_text(encoding="utf-8-sig"))
    local_prefix = _local_prefix_from_tos(data.get("_prefix", ""))
    sfx_root = Path(config.sfx_root)
    if not sfx_root.is_absolute():
        sfx_root = Path.cwd() / sfx_root
    audio_root = sfx_root / local_prefix

    catalog: SfxCatalog = {}
    for category, labels in data.items():
        if category.startswith("_") or not isinstance(labels, dict):
            continue
        for label, items in labels.items():
            if not isinstance(items, list):
                continue
            assets: list[SfxAsset] = []
            for item in items:
                if not isinstance(item, str):
                    continue
                path = audio_root / category / label / item
                if path.exists():
                    assets.append(SfxAsset(category=category, label=label, item=item, path=path))
            if assets:
                catalog[(category, label)] = assets
    return catalog


def _local_prefix_from_tos(prefix: str) -> Path:
    if prefix.startswith("tos://"):
        without_scheme = prefix[len("tos://") :].strip("/")
        parts = without_scheme.split("/", 1)
        if len(parts) == 2:
            return Path(parts[1])
    return Path(prefix.strip("/"))


def _plan_sfx_events(
    sample: dict[str, Any],
    catalog: SfxCatalog,
    config: PipelineConfig,
    *,
    force: bool = False,
) -> dict[str, Any]:
    plan_cache = Path(config.output_dir) / "cache" / "sfx_plans" / f"{sample['sample_id']}.json"
    if plan_cache.exists() and not force:
        return json.loads(plan_cache.read_text(encoding="utf-8"))
    prompt = _build_prompt(sample, catalog, config)
    plan = generate_json(prompt, model=config.sfx_planner_model, max_retries=config.max_retries)
    plan_cache.parent.mkdir(parents=True, exist_ok=True)
    plan_cache.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    return plan


def _build_prompt(
    sample: dict[str, Any],
    catalog: SfxCatalog,
    config: PipelineConfig,
) -> str:
    labels = [
        {"category": category, "label": label, "asset_count": len(assets)}
        for (category, label), assets in sorted(catalog.items())
    ]
    turns = [
        {
            "turn_id": turn["turn_id"],
            "speaker": turn["stream"],
            "start_ms": turn["start_ms"],
            "end_ms": turn["end_ms"],
            "text": turn["text"],
        }
        for turn in sample.get("turns", [])
    ]
    gaps = _dialogue_gaps(turns, int(sample["duration_ms"]))
    return json.dumps(
        {
            "task": "Create a sparse, dialogue-safe sound-effect plan for a stereo two-speaker dialogue mix.",
            "rules": [
                "Return only valid JSON.",
                "Use only category/label pairs from available_sfx_labels.",
                "Choose each event by the strongest cue in the dialogue text: named places, objects, activities, sports, vehicles, weather, animals, music, machines, doors, phones, crowds, or clear speaker reactions.",
                "Do not prefer any category by default. Human sounds are appropriate for explicit laughter, sighs, breaths, coughs, exertion, or other speaker reactions; non-human sounds are appropriate when the dialogue provides a concrete contextual cue.",
                "Prefer precise, localized effects over broad ambience. For example, a mentioned football game can justify a quiet sports crowd or whistle, while a mentioned door can justify a door sound.",
                "If multiple categories fit, pick the one that best matches the specific textual cue, not the safest or most generic category.",
                "Do not invent unrelated off-screen events just to make the mix busy; if there is no meaningful textual cue, return fewer events or no event.",
                "Use at most max_events events.",
                "Prefer placing events in dialogue gaps. If an event overlaps speech, keep gain_db quiet.",
                "Avoid loud impacts, horror, violence, or distracting sounds unless the dialogue explicitly calls for that kind of sound.",
                "Output schema: {\"events\":[{\"category\":\"Human sounds\",\"label\":\"laughter\",\"start_ms\":1234,\"duration_ms\":900,\"gain_db\":-20,\"ducking_db\":-2,\"reason\":\"short reason\"}]}",
            ],
            "selection_policy": {
                "priority": "Match the dialogue content first, then choose the quietest event that supports that content.",
                "category_balance": "Use Human sounds only when the cue is human; use non-human categories when the cue is an object, place, activity, environment, animal, vehicle, music, or crowd.",
                "silence_is_ok": "Return fewer than max_events if extra sounds would feel ungrounded or repetitive.",
            },
            "max_events": config.sfx_max_events,
            "duration_ms": sample["duration_ms"],
            "available_sfx_labels": labels,
            "dialogue_gaps": gaps[:24],
            "turns": turns[:80],
        },
        ensure_ascii=False,
    )


def _dialogue_gaps(turns: list[dict[str, Any]], duration_ms: int) -> list[dict[str, int]]:
    intervals = sorted((int(turn["start_ms"]), int(turn["end_ms"])) for turn in turns)
    merged: list[tuple[int, int]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    gaps: list[dict[str, int]] = []
    cursor = 0
    for start, end in merged:
        if start - cursor >= 400:
            gaps.append({"start_ms": cursor, "end_ms": start, "duration_ms": start - cursor})
        cursor = max(cursor, end)
    if duration_ms - cursor >= 400:
        gaps.append({"start_ms": cursor, "end_ms": duration_ms, "duration_ms": duration_ms - cursor})
    return gaps


def _sanitize_plan(
    plan: dict[str, Any],
    sample: dict[str, Any],
    catalog: SfxCatalog,
    config: PipelineConfig,
) -> list[SfxEvent]:
    events: list[SfxEvent] = []
    duration_ms = int(sample["duration_ms"])
    raw_events = plan.get("events") if isinstance(plan, dict) else None
    for raw in raw_events or []:
        if not isinstance(raw, dict):
            continue
        category = str(raw.get("category") or "").strip()
        label = str(raw.get("label") or "").strip()
        if (category, label) not in catalog:
            continue
        start_ms = int(max(0, min(duration_ms - 1, int(raw.get("start_ms") or 0))))
        event_duration = int(raw.get("duration_ms") or 1000)
        event_duration = max(200, min(5000, event_duration, duration_ms - start_ms))
        gain_db = _clamp_float(raw.get("gain_db"), -36.0, -6.0, config.sfx_default_gain_db)
        ducking_db = _clamp_float(raw.get("ducking_db"), -8.0, 0.0, config.sfx_ducking_db)
        events.append(
            SfxEvent(
                event_id=f"sfx_{len(events) + 1:03d}",
                category=category,
                label=label,
                start_ms=start_ms,
                duration_ms=event_duration,
                gain_db=round(gain_db, 2),
                ducking_db=round(ducking_db, 2),
                reason=str(raw.get("reason") or ""),
            )
        )
        if len(events) >= max(0, config.sfx_max_events):
            break
    return events or _fallback_events(sample, catalog, config)


def _fallback_events(
    sample: dict[str, Any],
    catalog: SfxCatalog,
    config: PipelineConfig,
) -> list[SfxEvent]:
    preferred = [
        ("Human sounds", "laughter"),
        ("Human sounds", "non_speech_vocalizations"),
        ("Human sounds", "body_movement_actions"),
        ("Human sounds", "breathing"),
    ]
    labels = [item for item in preferred if item in catalog] or sorted(catalog)[:1]
    if not labels or config.sfx_max_events <= 0:
        return []
    gaps = _dialogue_gaps(sample.get("turns", []), int(sample["duration_ms"]))
    start_ms = gaps[0]["start_ms"] if gaps else max(0, int(sample["duration_ms"]) // 2)
    category, label = labels[0]
    return [
        SfxEvent(
            event_id="sfx_001",
            category=category,
            label=label,
            start_ms=start_ms,
            duration_ms=min(1200, int(sample["duration_ms"]) - start_ms),
            gain_db=config.sfx_default_gain_db,
            ducking_db=config.sfx_ducking_db,
            reason="fallback sparse human sound",
        )
    ]


def _clamp_float(value: object, low: float, high: float, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(low, min(high, parsed))


def _select_assets(
    events: list[SfxEvent],
    catalog: SfxCatalog,
    sample_id: str,
    config: PipelineConfig,
) -> list[SfxEvent]:
    for event in events:
        assets = catalog[(event.category, event.label)]
        seed = f"{config.sfx_random_seed}:{sample_id}:{event.event_id}:{event.category}:{event.label}:{event.start_ms}"
        rng = random.Random(int(hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16], 16))
        ranked = sorted(((_asset_penalty(asset, event), rng.random()), asset) for asset in assets)
        best_penalty = ranked[0][0][0]
        best_assets = [asset for (penalty, _), asset in ranked if penalty == best_penalty]
        asset = rng.choice(best_assets)
        event.asset_path = str(asset.path)
    return events


def _asset_penalty(asset: SfxAsset, event: SfxEvent) -> int:
    if event.label in {
        "body_impacts",
        "horror_vocal_effects",
        "processed_vocals",
        "pain_death_vocalizations",
        "screaming_shouting",
    }:
        return 0
    text = f"{asset.item} {asset.path.name}".lower()
    avoid_terms = (
        "evil",
        "villain",
        "horror",
        "terror",
        "monster",
        "ghost",
        "alien",
        "robot",
        "death",
        "dead",
        "dying",
        "scream",
        "fight",
        "fighting",
        "hit",
        "war",
    )
    return sum(1 for term in avoid_terms if term in text)


def _render_mix(
    sample: dict[str, Any],
    sample_dir: Path,
    events: list[SfxEvent],
    config: PipelineConfig,
    output_path: Path,
) -> None:
    base_path = sample_dir / sample["audio_files"][config.stereo_audio_name]
    mixed = AudioSegment.from_file(base_path).set_frame_rate(config.sample_rate).set_channels(2).set_sample_width(2)
    for event in events:
        if event.asset_path is None:
            continue
        sfx = AudioSegment.from_file(event.asset_path).set_frame_rate(config.sample_rate).set_channels(2).set_sample_width(2)
        max_len = max(1, min(event.duration_ms, len(mixed) - event.start_ms))
        if max_len <= 0:
            continue
        sfx = sfx[:max_len]
        fade = min(config.sfx_fade_ms, max(1, len(sfx) // 3))
        sfx = sfx.apply_gain(event.gain_db).fade_in(fade).fade_out(fade)
        if event.ducking_db < 0:
            mixed = _duck(mixed, event.start_ms, event.start_ms + len(sfx), event.ducking_db)
        mixed = mixed.overlay(sfx, position=event.start_ms)[: len(mixed)]
        event.end_ms = event.start_ms + len(sfx)
        event.duration_ms = len(sfx)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mixed.export(output_path, format="wav")


def _duck(audio: AudioSegment, start_ms: int, end_ms: int, gain_db: float) -> AudioSegment:
    start = max(0, min(start_ms, len(audio)))
    end = max(start, min(end_ms, len(audio)))
    if end <= start:
        return audio
    return audio[:start] + audio[start:end].apply_gain(gain_db) + audio[end:]


def _event_dict(event: SfxEvent, sample_dir: Path) -> dict[str, Any]:
    asset_path = event.asset_path
    if asset_path:
        try:
            asset_path = str(Path(asset_path).resolve().relative_to(sample_dir.resolve()))
        except ValueError:
            try:
                asset_path = str(Path(asset_path).resolve().relative_to(Path.cwd().resolve()))
            except ValueError:
                asset_path = str(Path(asset_path))
    return {
        "event_id": event.event_id,
        "category": event.category,
        "label": event.label,
        "start_ms": event.start_ms,
        "end_ms": event.end_ms if event.end_ms is not None else event.start_ms + event.duration_ms,
        "duration_ms": event.duration_ms,
        "gain_db": event.gain_db,
        "ducking_db": event.ducking_db,
        "asset_path": asset_path,
        "reason": event.reason,
    }
