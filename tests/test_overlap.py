from __future__ import annotations

from pipeline.align_chunks import estimate_word_timings, find_overlapping_turns, words_overlapping_chunk


def test_overlap_and_estimated_words():
    turns = [
        {
            "turn_id": "u1",
            "stream": "user_voice",
            "message": "hello world",
            "start_ms": 0,
            "end_ms": 1000,
        }
    ]

    overlaps = find_overlapping_turns(turns, "user_voice", 400, 560)
    words = estimate_word_timings("hello world", 0, 1000)
    chunk_words = words_overlapping_chunk(words, 400, 560)

    assert overlaps[0]["turn_id"] == "u1"
    assert overlaps[0]["overlap_ratio"] == 1.0
    assert [word["word"] for word in chunk_words]

