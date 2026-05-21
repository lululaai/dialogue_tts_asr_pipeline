from __future__ import annotations

import argparse
import logging

from dotenv import load_dotenv

from .build_dataset import run_pipeline
from .config import PipelineConfig


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
    parser = argparse.ArgumentParser(description="Build chunk-level duplex TTS-ASR training JSON.")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chunk-ms", type=int, default=160)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--user-agent", default="agent_1")
    parser.add_argument("--assistant-agent", default="agent_2")
    parser.add_argument("--user-voice", dest="user_tts_voice", default="alloy")
    parser.add_argument("--assistant-voice", dest="assistant_tts_voice", default="coral")
    parser.add_argument("--tts-model", default="gpt-4o-mini-tts")
    parser.add_argument("--asr-model", default="gpt-4o-transcribe")
    parser.add_argument("--tts-response-format", default="wav")
    parser.add_argument("--transcribe-each-chunk", type=parse_bool, default=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-tts", action="store_true")
    parser.add_argument("--skip-asr", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fallback-turn-overlap", type=parse_bool, default=True)
    parser.add_argument("--inter-turn-silence-ms", type=int, default=240)
    parser.add_argument("--tts-concurrency", type=int, default=3)
    parser.add_argument("--asr-concurrency", type=int, default=6)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
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
        tts_model=args.tts_model,
        asr_model=args.asr_model,
        tts_response_format=args.tts_response_format,
        inter_turn_silence_ms=args.inter_turn_silence_ms,
        transcribe_each_chunk=args.transcribe_each_chunk,
        max_retries=args.max_retries,
        fallback_turn_overlap=args.fallback_turn_overlap,
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

