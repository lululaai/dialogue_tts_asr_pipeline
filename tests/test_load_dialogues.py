from __future__ import annotations

from pipeline.load_dialogues import load_dialogues


def test_load_dialogues_normalizes_turns(config, tiny_input):
    dialogues = load_dialogues(tiny_input, config)

    assert len(dialogues) == 1
    assert dialogues[0]["sample_id"] == "dialogue_000001"
    assert dialogues[0]["source_dialogue_id"] == "d1"
    assert [turn["turn_id"] for turn in dialogues[0]["turns"]] == ["u1", "a1"]
    assert [turn["stream"] for turn in dialogues[0]["turns"]] == ["user_voice", "assistant_voice"]

