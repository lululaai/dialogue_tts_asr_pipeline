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
    scene: str | None = None
    intensity: int | None = None
    asset_path: str | None = None
    end_ms: int | None = None


SfxCatalog = dict[tuple[str, str], list[SfxAsset]]

SFX_EVENT_MIN_DURATION_MS = 150
SFX_EVENT_MAX_DURATION_MS = 12000

SFX_SCENES: list[dict[str, Any]] = [
    {
        "scene": "indoor_argument",
        "cues": ["argument", "angry exchange", "tense voices", "fight in a room"],
        "sound_profile": "tight indoor room tone, restrained body movement, occasional object handling.",
    },
    {
        "scene": "office_talk",
        "cues": ["office", "coworker", "meeting", "computer", "desk", "paperwork"],
        "sound_profile": "quiet office ambience, keyboard or paper details, distant lobby noise.",
    },
    {
        "scene": "indoor_room_chat",
        "cues": ["home", "bedroom", "living room", "private indoor conversation"],
        "sound_profile": "soft room tone, small household movement, low non-distracting ambience.",
    },
    {
        "scene": "restaurant_chat",
        "cues": ["restaurant", "dinner", "server", "menu", "meal"],
        "sound_profile": "light restaurant crowd bed, dishes or cutlery only when text supports it.",
    },
    {
        "scene": "rainy_street_chat",
        "cues": ["rain", "wet street", "umbrella", "stormy walk", "drizzle outside"],
        "sound_profile": "rain or wet traffic ambience, occasional footsteps or vehicles.",
    },
    {
        "scene": "sunny_street_chat",
        "cues": ["street", "walking outside", "sunny day", "sidewalk", "city"],
        "sound_profile": "gentle urban ambience, light traffic, pedestrians, no rain.",
    },
    {
        "scene": "cafe_chat",
        "cues": ["cafe", "coffee", "barista", "espresso", "tea shop"],
        "sound_profile": "small cafe ambience, cups, register, espresso machine when cued.",
    },
    {
        "scene": "library_study_chat",
        "cues": ["library", "study", "books", "quiet reading", "homework"],
        "sound_profile": "very quiet room tone, page turns, soft paper movement, minimal crowd.",
    },
    {
        "scene": "factory_workshop_chat",
        "cues": ["factory", "workshop", "machine shop", "tools", "industrial work"],
        "sound_profile": "low industrial ambience, tools or machinery only if dialogue clearly points there.",
    },
    {
        "scene": "after_exercise_chat",
        "cues": ["gym", "workout", "running", "sports practice", "tired after exercise"],
        "sound_profile": "breathing, clothing movement, gym or sports ambience at low level.",
    },
    {
        "scene": "kitchen_cooking_chat",
        "cues": ["kitchen", "cooking", "pan", "recipe", "dishes", "food prep"],
        "sound_profile": "light appliance hum, cooking sizzle, water or dish sounds when cued.",
    },
    {
        "scene": "car_interior_chat",
        "cues": ["car", "driving", "road trip", "traffic from inside", "passenger"],
        "sound_profile": "vehicle interior tone, subdued road noise, turn signal or horn only when cued.",
    },
    {
        "scene": "public_transport_chat",
        "cues": ["bus", "subway", "train ride", "commute", "platform"],
        "sound_profile": "transit ambience, doors, announcements, vehicle motion when grounded.",
    },
    {
        "scene": "park_walk_chat",
        "cues": ["park", "walk", "grass", "trees", "picnic"],
        "sound_profile": "outdoor nature ambience, footsteps, birds or wind if consistent with text.",
    },
    {
        "scene": "hospital_clinic_chat",
        "cues": ["hospital", "clinic", "doctor", "nurse", "appointment", "medical"],
        "sound_profile": "quiet clinical room tone, soft equipment beeps or hallway ambience when cued.",
    },
    {
        "scene": "school_classroom_chat",
        "cues": ["school", "class", "teacher", "classroom", "students"],
        "sound_profile": "classroom room tone, paper, chairs, distant student ambience.",
    },
    {
        "scene": "shopping_mall_chat",
        "cues": ["mall", "store", "shopping", "cashier", "checkout"],
        "sound_profile": "retail ambience, register, footsteps, crowd bed at low level.",
    },
    {
        "scene": "airport_station_chat",
        "cues": ["airport", "station", "flight", "gate", "luggage", "departure"],
        "sound_profile": "transport hub ambience, rolling luggage, announcements when text supports them.",
    },
    {
        "scene": "beach_lakeside_chat",
        "cues": ["beach", "lake", "ocean", "shore", "swimming"],
        "sound_profile": "water and wind ambience, birds or splashes only when appropriate.",
    },
    {
        "scene": "nighttime_quiet_chat",
        "cues": ["night", "late", "sleepy", "quiet house", "whispering"],
        "sound_profile": "very low room tone, soft movements, avoid busy ambience.",
    },
]

SFX_SCENE_NAMES = {scene["scene"] for scene in SFX_SCENES}


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
        "scene": _plan_scene(plan),
        "scene_reason": str(plan.get("scene_reason") or "") if isinstance(plan, dict) else "",
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
    if config.sfx_max_events <= 0:
        return {"scene": None, "scene_reason": "sfx_max_events is 0", "events": []}
    plan_cache = Path(config.output_dir) / "cache" / "sfx_plans" / f"{sample['sample_id']}.json"
    if plan_cache.exists() and not force:
        return json.loads(plan_cache.read_text(encoding="utf-8"))
    system_instruction = _build_sfx_system_prompt(catalog, config)
    prompt = _build_sfx_user_prompt(sample)
    plan = generate_json(
        prompt,
        model=config.sfx_planner_model,
        system_instruction=system_instruction,
        max_retries=config.max_retries,
    )
    plan_cache.parent.mkdir(parents=True, exist_ok=True)
    plan_cache.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    return plan


def _build_sfx_system_prompt(
    catalog: SfxCatalog,
    config: PipelineConfig,
) -> str:
    labels = [
        [category, label, len(assets)]
        for (category, label), assets in sorted(catalog.items())
    ]
    scenes = [
        {
            "scene": scene["scene"],
            "cues": scene["cues"],
        }
        for scene in SFX_SCENES
    ]
    return json.dumps(
        {
            "task": "Create a sparse, dialogue-safe sound-effect plan for a stereo two-speaker dialogue mix.",
            "input_format": {
                "turns": ["turn_id", "stream", "start_ms", "end_ms", "text"],
                "dialogue_gaps": ["start_ms", "end_ms", "duration_ms"],
                "available_sfx_labels": ["category", "label", "asset_count"],
            },
            "rules": [
                "Return only valid JSON.",
                "First choose one scene from available_scenes that best fits the dialogue. If no scene is explicit, infer the most natural low-risk scene from the text.",
                "Use only category/label pairs from available_sfx_labels.",
                "Choose each event by the strongest cue in the dialogue text, combining that cue with the selected scene: named places, objects, activities, sports, vehicles, weather, animals, music, machines, doors, phones, crowds, or clear speaker reactions.",
                "The selected scene is a grounding constraint, not permission to add generic ambience. Events still need a clear dialogue cue or a scene-consistent gap-filling reason.",
                "Do not prefer any category by default. Human sounds are appropriate for explicit laughter, sighs, breaths, coughs, exertion, or other speaker reactions; non-human sounds are appropriate when the dialogue provides a concrete contextual cue.",
                "Prefer precise, localized effects over broad ambience. Use ambience only when the selected scene is strongly implied and a short bed would improve realism.",
                "If multiple categories fit, pick the one that best matches the specific textual cue, not the safest or most generic category.",
                "Do not invent unrelated off-screen events just to make the mix busy; if there is no meaningful textual cue, return fewer events or no event.",
                "Use at most max_events events.",
                "Choose start_ms and end_ms flexibly from the timeline. Prefer dialogue gaps, but allow quiet overlap when a scene bed or reaction naturally sits under speech.",
                "Choose duration_ms from the event type: short foley/reactions 150-1200 ms, object actions 300-2500 ms, ambience/weather/traffic/crowd beds 1500-12000 ms. Keep each event inside duration_ms.",
                "Choose intensity from 1 to 5 and map it to gain_db: 1 is barely audible around -32 to -28 dB, 2 is subtle around -28 to -23 dB, 3 is present around -23 to -18 dB, 4 is noticeable around -18 to -12 dB, 5 is foreground around -12 to -6 dB and should be rare.",
                "If an event overlaps speech, use intensity 1-2, gain_db <= -20, and ducking_db between -4 and -1 unless the dialogue explicitly calls for a foreground sound.",
                "Avoid loud impacts, horror, violence, or distracting sounds unless the dialogue explicitly calls for that kind of sound.",
                "Output schema: {\"scene\":\"cafe_chat\",\"scene_reason\":\"short reason\",\"events\":[{\"scene\":\"cafe_chat\",\"category\":\"Human sounds\",\"label\":\"laughter\",\"start_ms\":1234,\"end_ms\":2134,\"duration_ms\":900,\"intensity\":2,\"gain_db\":-24,\"ducking_db\":-2,\"reason\":\"short reason\"}]}",
            ],
            "selection_policy": {
                "priority": "Select the best scene first, then match dialogue content, then choose the quietest event that supports that content.",
                "category_balance": "Use Human sounds only when the cue is human; use non-human categories when the cue is an object, place, activity, environment, animal, vehicle, music, or crowd.",
                "timing": "Use dialogue_gaps for clean placement when possible. For ambience beds, align with a natural span of the conversation instead of forcing a fixed length.",
                "intensity": "Most events should be intensity 1-3. Use intensity 4-5 only for explicit foreground actions.",
                "silence_is_ok": "Return fewer than max_events if extra sounds would feel ungrounded or repetitive.",
            },
            "max_events": config.sfx_max_events,
            "available_scenes": scenes,
            "available_sfx_labels": labels,
        },
        ensure_ascii=False,
    )


def _build_sfx_user_prompt(sample: dict[str, Any]) -> str:
    turn_dicts = [
        {
            "start_ms": turn["start_ms"],
            "end_ms": turn["end_ms"],
        }
        for turn in sample.get("turns", [])
    ]
    turns = [
        [
            turn["turn_id"],
            turn["stream"],
            turn["start_ms"],
            turn["end_ms"],
            turn["text"],
        ]
        for turn in sample.get("turns", [])
    ]
    gaps = _dialogue_gaps(turn_dicts, int(sample["duration_ms"]))
    return json.dumps(
        {
            "duration_ms": sample["duration_ms"],
            "dialogue_gaps": gaps[:24],
            "turns": turns[:80],
        },
        ensure_ascii=False,
    )


def _build_prompt(
    sample: dict[str, Any],
    catalog: SfxCatalog,
    config: PipelineConfig,
) -> str:
    return json.dumps(
        {
            "system_instruction": json.loads(_build_sfx_system_prompt(catalog, config)),
            "user": json.loads(_build_sfx_user_prompt(sample)),
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
    plan_scene = _normalize_scene(plan.get("scene")) if isinstance(plan, dict) else None
    for raw in raw_events or []:
        if not isinstance(raw, dict):
            continue
        category = str(raw.get("category") or "").strip()
        label = str(raw.get("label") or "").strip()
        if (category, label) not in catalog:
            continue
        start_ms = int(max(0, min(duration_ms - 1, int(raw.get("start_ms") or 0))))
        event_duration = _event_duration_from_plan(raw, start_ms, duration_ms)
        gain_db = _clamp_float(raw.get("gain_db"), -36.0, -6.0, config.sfx_default_gain_db)
        ducking_db = _clamp_float(raw.get("ducking_db"), -8.0, 0.0, config.sfx_ducking_db)
        intensity = _clamp_intensity(raw.get("intensity"))
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
                scene=_normalize_scene(raw.get("scene")) or plan_scene,
                intensity=intensity,
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
            scene="indoor_room_chat",
            intensity=2,
        )
    ]


def _normalize_scene(value: object) -> str | None:
    scene = str(value or "").strip()
    if scene in SFX_SCENE_NAMES:
        return scene
    return None


def _plan_scene(plan: object) -> str | None:
    if not isinstance(plan, dict):
        return None
    scene = _normalize_scene(plan.get("scene"))
    if scene:
        return scene
    for raw in plan.get("events") or []:
        if isinstance(raw, dict):
            scene = _normalize_scene(raw.get("scene"))
            if scene:
                return scene
    return None


def _event_duration_from_plan(raw: dict[str, Any], start_ms: int, sample_duration_ms: int) -> int:
    remaining_ms = max(1, sample_duration_ms - start_ms)
    duration_value = raw.get("duration_ms")
    if duration_value is None and raw.get("end_ms") is not None:
        try:
            duration_value = int(raw["end_ms"]) - start_ms
        except (TypeError, ValueError):
            duration_value = None
    try:
        event_duration = int(duration_value) if duration_value is not None else 1000
    except (TypeError, ValueError):
        event_duration = 1000
    return max(SFX_EVENT_MIN_DURATION_MS, min(SFX_EVENT_MAX_DURATION_MS, event_duration, remaining_ms))


def _clamp_intensity(value: object) -> int | None:
    if value is None:
        return None
    try:
        return max(1, min(5, int(value)))
    except (TypeError, ValueError):
        return None


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
        "scene": event.scene,
        "intensity": event.intensity,
        "asset_path": asset_path,
        "reason": event.reason,
    }
