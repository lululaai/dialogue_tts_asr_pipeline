from __future__ import annotations

import argparse
import logging

from dotenv import load_dotenv

from .build_dataset import run_pipeline
from .config import GEMINI_TTS_VOICES, PipelineConfig


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {value!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build duplex dialogue audio from text turns.")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chunk-ms", type=int, default=160)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--user-agent", default="agent_1")
    parser.add_argument("--assistant-agent", default="agent_2")
    parser.add_argument("--user-voice", dest="user_tts_voice", default="Kore")
    parser.add_argument("--assistant-voice", dest="assistant_tts_voice", default="Puck")
    parser.add_argument("--tts-speed", type=float, default=1.2)
    parser.add_argument("--tts-provider", choices=("google", "openai"), default="google")
    parser.add_argument("--tts-model", default="gemini-2.5-flash-tts")
    parser.add_argument("--tts-random-voice", type=parse_bool, default=True)
    parser.add_argument(
        "--google-tts-voices",
        default=",".join(GEMINI_TTS_VOICES),
        help="Comma-separated Gemini TTS voice pool used when --tts-random-voice true.",
    )
    parser.add_argument("--asr-model", default="whisper-1")
    parser.add_argument("--asr-mode", choices=("turn", "chunk"), default="turn")
    parser.add_argument("--tts-response-format", default="wav")
    parser.add_argument("--generate-chunk-targets", type=parse_bool, default=False)
    parser.add_argument("--transcribe-each-chunk", type=parse_bool, default=False)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-tts", action="store_true")
    parser.add_argument("--skip-asr", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fallback-turn-overlap", type=parse_bool, default=True)
    parser.add_argument("--inter-turn-silence-ms", type=int, default=240)
    parser.add_argument("--turn-overlap-enabled", type=parse_bool, default=True)
    parser.add_argument("--turn-overlap-min-count", type=int, default=2)
    parser.add_argument("--turn-overlap-max-count", type=int, default=3)
    parser.add_argument("--turn-overlap-start-ratio", type=float, default=0.5)
    parser.add_argument("--backchannel-enabled", type=parse_bool, default=True)
    parser.add_argument("--backchannel-max-count", type=int, default=3)
    parser.add_argument("--backchannel-min-turn-duration-ms", type=int, default=3000)
    parser.add_argument("--backchannel-max-duration-ms", type=int, default=650)
    parser.add_argument("--backchannel-min-start-offset-ms", type=int, default=900)
    parser.add_argument("--backchannel-min-end-margin-ms", type=int, default=650)
    parser.add_argument("--backchannel-gain-db", type=float, default=-5.0)
    parser.add_argument(
        "--backchannel-phrases",
        default="yes,yeah,yep,right,correct,exactly,true,that's right,you are right,I agree,sure,of course,totally,absolutely",
        help="Comma-separated listener backchannel phrases.",
    )
    parser.add_argument(
        "--tts-concurrency",
        type=int,
        default=0,
        help="TTS worker count. 0 means one concurrent request per turn/backchannel item.",
    )
    parser.add_argument("--asr-concurrency", type=int, default=6)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.transcribe_each_chunk and not args.generate_chunk_targets:
        parser.error("--transcribe-each-chunk requires --generate-chunk-targets true")
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = PipelineConfig(
        input_json=args.input_json,
        output_dir=args.output_dir,
        chunk_ms=args.chunk_ms,
        sample_rate=args.sample_rate,
        user_agent=args.user_agent,
        assistant_agent=args.assistant_agent,
        user_tts_voice=args.user_tts_voice,
        assistant_tts_voice=args.assistant_tts_voice,
        tts_speed=args.tts_speed,
        tts_provider=args.tts_provider,
        tts_model=args.tts_model,
        tts_random_voice=args.tts_random_voice,
        google_tts_voices=tuple(voice.strip() for voice in args.google_tts_voices.split(",") if voice.strip()),
        asr_model=args.asr_model,
        asr_mode=args.asr_mode,
        tts_response_format=args.tts_response_format,
        inter_turn_silence_ms=args.inter_turn_silence_ms,
        generate_chunk_targets=args.generate_chunk_targets,
        transcribe_each_chunk=args.transcribe_each_chunk,
        max_retries=args.max_retries,
        fallback_turn_overlap=args.fallback_turn_overlap,
        turn_overlap_enabled=args.turn_overlap_enabled,
        turn_overlap_min_count=args.turn_overlap_min_count,
        turn_overlap_max_count=args.turn_overlap_max_count,
        turn_overlap_start_ratio=args.turn_overlap_start_ratio,
        backchannel_enabled=args.backchannel_enabled,
        backchannel_max_count=args.backchannel_max_count,
        backchannel_min_turn_duration_ms=args.backchannel_min_turn_duration_ms,
        backchannel_max_duration_ms=args.backchannel_max_duration_ms,
        backchannel_min_start_offset_ms=args.backchannel_min_start_offset_ms,
        backchannel_min_end_margin_ms=args.backchannel_min_end_margin_ms,
        backchannel_gain_db=args.backchannel_gain_db,
        backchannel_phrases=tuple(phrase.strip() for phrase in args.backchannel_phrases.split(",") if phrase.strip()),
        tts_concurrency=args.tts_concurrency,
        asr_concurrency=args.asr_concurrency,
    )
    run_pipeline(
        config,
        limit=args.limit,
        resume=args.resume,
        skip_tts=args.skip_tts,
        skip_asr=args.skip_asr,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
