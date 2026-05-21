# Dialogue TTS-ASR Chunk Pipeline

Python pipeline for turning dialogue JSON into chunk-level duplex training samples.

## Install

Use Python 3.11+ and make sure `ffmpeg` is on `PATH`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=...
```

## Run

```bash
python3 -m pipeline.cli \
  --input-json /path/to/test_freq.json \
  --output-dir /path/to/output \
  --chunk-ms 160 \
  --sample-rate 24000 \
  --user-agent agent_1 \
  --assistant-agent agent_2 \
  --user-voice alloy \
  --assistant-voice coral \
  --tts-model gpt-4o-mini-tts \
  --asr-model gpt-4o-transcribe \
  --transcribe-each-chunk true
```

Useful flags:

```bash
--limit 1
--resume
--skip-tts
--skip-asr
--force
--fallback-turn-overlap true
--inter-turn-silence-ms 240
```

`--skip-tts` expects per-turn WAV files to already exist under
`samples/{sample_id}/audio/turns/`.

## Output

Each processed dialogue creates:

```text
output/
  manifest.jsonl
  failed.jsonl
  cache/
    tts/
    asr/
  samples/
    dialogue_000001/
      metadata.json
      sample.json
      audio/
        turns/
        user_voice.wav
        assistant_voice.wav
        chunks/
          user_voice/
          assistant_voice/
      asr/
        chunks/
```

The dataset references separate `user_voice.wav` and `assistant_voice.wav`
streams. It does not generate or reference a mixed `final_duplex_mix.wav`.

## Tests

Unit tests do not call OpenAI. They mock TTS/ASR and use tiny synthetic WAV files.

```bash
pytest
```

