# Dialogue TTS Overlap Pipeline

Python pipeline for turning dialogue JSON into duplex dialogue audio. Each text
turn is synthesized with TTS, then the per-turn WAV files are arranged on a
two-speaker timeline. By default, a few turn transitions overlap so one speaker
starts while the other is still talking.

## Install

Use Python 3.11+ and make sure `ffmpeg` is on `PATH`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
gcloud auth application-default login
gcloud auth application-default set-quota-project YOUR_PROJECT_ID
export GOOGLE_CLOUD_PROJECT=YOUR_PROJECT_ID
export GOOGLE_CLOUD_REGION=us-central1
```

## Run

```bash
python3 -m pipeline.cli \
  --input-json /path/to/test_freq.json \
  --output-dir /path/to/output \
  --sample-rate 24000 \
  --user-agent agent_1 \
  --assistant-agent agent_2 \
  --tts-speed 1.2 \
  --tts-provider google \
  --tts-model gemini-2.5-flash-tts \
  --tts-random-voice true
```

The default flow does not split audio into chunks and does not run ASR. It
generates per-turn audio, mono speaker streams, and a stereo file:
`user_voice.wav` is the left channel and `assistant_voice.wav` is the right
channel.

TTS uses Google Gemini TTS by default with `gemini-2.5-flash-tts` and a default
pace of `--tts-speed 1.2`. When `--tts-random-voice true` is enabled, each
dialogue picks two fixed voices from Google's 30 Gemini TTS voices: one for the
user stream and one for the assistant stream. The voice pool is `Zephyr`,
`Puck`, `Charon`, `Kore`, `Fenrir`, `Leda`, `Orus`, `Aoede`, `Callirrhoe`,
`Autonoe`, `Enceladus`, `Iapetus`, `Umbriel`, `Algieba`, `Despina`, `Erinome`,
`Algenib`, `Rasalgethi`, `Laomedeia`, `Achernar`, `Alnilam`, `Schedar`,
`Gacrux`, `Pulcherrima`, `Achird`, `Zubenelgenubi`, `Vindemiatrix`,
`Sadachbia`, `Sadaltager`, `Sulafat`. To use OpenAI TTS instead, pass
`--tts-provider openai --tts-model gpt-4o-mini-tts`, `--tts-random-voice false`,
and set `OPENAI_API_KEY`.

Turn overlaps are enabled by default. The scheduler chooses up to 2-3
non-adjacent cross-speaker transitions per dialogue. On an overlapped
transition, the next turn starts when the previous turn has reached
`--turn-overlap-start-ratio` of its duration, default `0.5`.

Listener backchannels are also enabled by default. For a few longer turns, the
pipeline synthesizes short listener responses such as `yes`, `yeah`, or `right`
with the listener's voice and overlays them quietly on the listener stream.
These do not become dialogue turns; they are recorded in `backchannel_events`.

Optional SFX mixing can add a sparse human-sound layer on top of
`duplex_stereo.wav`. It uses a Google Gemini text model to plan events, but the
actual event audio is selected only from `uploaded_segments_map_to_file.json`
and local files under `sfx/audio_segments/...`, matching the TOS key layout.
The original stereo file is preserved and the mixed file is written as
`audio/duplex_stereo_sfx.wav`.

Useful overlap and batch flags:

```bash
--turn-overlap-enabled true
--turn-overlap-min-count 2
--turn-overlap-max-count 3
--turn-overlap-start-ratio 0.5
--inter-turn-silence-ms 240
--backchannel-enabled true
--backchannel-max-count 3
--backchannel-min-turn-duration-ms 3000
--backchannel-max-duration-ms 650
--backchannel-min-start-offset-ms 900
--backchannel-min-end-margin-ms 650
--backchannel-gain-db -5
--backchannel-phrases "yes,yeah,yep,right,correct,exactly,true,that's right,you are right,I agree,sure,of course,totally,absolutely"
--sfx-enabled true
--sfx-root sfx
--sfx-map-path uploaded_segments_map_to_file.json
--sfx-planner-model gemini-3.5-flash
--sfx-max-events 4
--sample-concurrency 4
--tts-concurrency 4
--google-request-concurrency 8
```

For large batches, `--sample-concurrency` runs multiple dialogues at once while
`--google-request-concurrency` caps total concurrent Google TTS and SFX planner
requests across the whole process. A practical starting point on a medium
server is:

```bash
python3 -m pipeline.cli \
  --input-json data/test_freq.json \
  --output-dir output \
  --tts-provider google \
  --tts-model gemini-2.5-flash-tts \
  --sfx-enabled true \
  --sample-concurrency 4 \
  --tts-concurrency 4 \
  --google-request-concurrency 8 \
  --resume
```

The older chunk/ASR target path is still available by opting in:

```bash
python3 -m pipeline.cli \
  --input-json /path/to/test_freq.json \
  --output-dir /path/to/output \
  --generate-chunk-targets true \
  --chunk-ms 160 \
  --transcribe-each-chunk true \
  --asr-mode turn \
  --asr-model whisper-1
```

In `turn` ASR mode, each per-turn WAV is transcribed with `whisper-1`,
requesting word timestamps. Those word timestamps are converted to the dialogue
timeline and mapped into the fixed chunks.

The older per-chunk ASR path is still available:

```bash
python3 -m pipeline.cli \
  --input-json /path/to/test_freq.json \
  --output-dir /path/to/output \
  --generate-chunk-targets true \
  --transcribe-each-chunk true \
  --asr-mode chunk \
  --asr-model gpt-4o-transcribe
```

Useful flags:

```bash
--limit 1
--resume
--skip-tts
--skip-asr
--force
--fallback-turn-overlap true
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
        backchannels/
        user_voice.wav
        assistant_voice.wav
        duplex_stereo.wav
        duplex_stereo_sfx.wav  # when --sfx-enabled true
```

When `--generate-chunk-targets true` is used, the sample also includes
`chunk_targets` and writes:

```text
output/
  samples/
    dialogue_000001/
      audio/
        chunks/
          user_voice/
          assistant_voice/
      asr/
        turns/
        chunks/
```

`sample.json` records the final `turns` timeline, an `overlap_events` array
that identifies which turns overlap, and a `backchannel_events` array for
listener acknowledgements. When SFX mixing is enabled, it also records
`sfx_events` and `audio_files.duplex_stereo_sfx`.

## Tests

Unit tests do not call OpenAI. They mock TTS/ASR and use tiny synthetic WAV files.

```bash
pytest
```
