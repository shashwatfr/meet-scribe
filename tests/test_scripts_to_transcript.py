"""Speaker-label stripping in scripts_to_transcript.py.
Run by pytest (uv run pytest).
"""
from scripts_to_transcript import script_to_segments, strip_label


def test_strip_label_removes_only_the_leading_label():
    assert strip_label("PRIYA: Hi Rahul, thanks.") == "Hi Rahul, thanks."
    assert strip_label("ARJUN: I can do the, wait, no. Sana, can you do it?") == \
        "I can do the, wait, no. Sana, can you do it?"
    assert strip_label("No label here: keep it.") == "No label here: keep it."


def test_segments_have_no_labels_and_increasing_times():
    segments = script_to_segments("PRIYA: One two three four five.\n\nRAHUL: Six seven.\n")
    assert [s["text"] for s in segments] == ["One two three four five.", "Six seven."]
    assert segments[0]["start"] == 0 and segments[0]["end"] == 2.0
    assert segments[1]["start"] == 2.5
