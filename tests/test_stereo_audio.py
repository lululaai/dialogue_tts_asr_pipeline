from __future__ import annotations

from pydub.generators import Sine

from pipeline.audio_utils import build_stereo_audio


def test_build_stereo_audio_maps_user_to_left_and_assistant_to_right():
    left = Sine(440).to_audio_segment(duration=200).apply_gain(-3)
    right = Sine(440).to_audio_segment(duration=200).apply_gain(-18)

    stereo = build_stereo_audio(left, right, 24000)
    left_channel, right_channel = stereo.split_to_mono()

    assert stereo.channels == 2
    assert left_channel.rms > right_channel.rms * 4
