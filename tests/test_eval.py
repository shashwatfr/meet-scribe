"""Word alignment used by eval_transcripts.py. No API calls.
Run by pytest (uv run pytest).
"""
from eval_transcripts import align, normalize


def test_normalize_drops_labels_case_and_punctuation():
    assert normalize("PRIYA: Hi Rahul, it's the 13th.\nRAHUL: Off-site?") == \
        ["hi", "rahul", "its", "the", "13th", "off", "site"]


def test_align_counts_each_kind_of_error():
    ops = align("a b c d".split(), "a x c d e".split())
    assert [op for op in ops if op[0] != "OK"] == [("SUB", "b", "x"), ("INS", "", "e")]
    assert [op[0] for op in align(["a", "b"], ["b"])] == ["DEL", "OK"]
