"""LLM-assisted merge of a split meeting: the plan is applied with checks. No API calls.
Run by pytest (uv run pytest).
"""
import json

import pytest

from services import analyze
from services.analyze import AnalyzeError, DailyLimitError, apply_merge_plan, merge_pieces, number_items
from services.schemas import ActionItem, Decision, Extraction, MergePlan

PIECE_ENDS = [60.0, 120.0, 180.0]
PARTS = [
    Extraction(  # part 1: 0-60 s
        decisions=[Decision(topic="price", text="Ask for $15,000.", timestamp=54)],
        action_items=[ActionItem(task="mail the window with our counter offer", owner="Anita",
                                 due_date=None, timestamp=50)],
        open_topics=["offsite location", "report owner"]),
    Extraction(  # part 2: 60-120 s
        decisions=[Decision(topic="renewal price", text="Cap is $16,000.", timestamp=64)],
        action_items=[ActionItem(task="Book the venue within two days", owner=None,
                                 due_date="within two days", timestamp=78)],
        open_topics=["offsite dates"]),
    Extraction(  # part 3: 120-180 s, the recap
        decisions=[Decision(topic="offsite", text="Offsite in Pune.", timestamp=150)],
        action_items=[ActionItem(task="Email the vendor with the counteroffer", owner=None,
                                 due_date="by November 8th", timestamp=160),
                      ActionItem(task="Book the venue", owner="Vikram", due_date=None, timestamp=170)],
        open_topics=["report ownership"]),
]
# ids: a1 mail (50), a2 venue (78), a3 email (160), a4 venue recap (170)
#      d1 $15,000 (54), d2 $16,000 (64), d3 Pune (150); o1 offsite location, o2 report owner,
#      o3 offsite dates, o4 report ownership


def plan(**kwargs) -> MergePlan:
    base = {"action_clusters": [], "decision_clusters": [], "open_topic_clusters": [],
            "decided_open_topics": []}
    return MergePlan(**{**base, **kwargs})


def merged(p: MergePlan):
    return apply_merge_plan(number_items(PARTS, PIECE_ENDS), p)


def test_clusters_applied_task_from_keep_details_from_one_member():
    ex, log = merged(plan(action_clusters=[{"ids": ["a1", "a3"], "keep": "a3"},
                                           {"ids": ["a2", "a4"], "keep": "a4"}]))
    tasks = {a.task: a for a in ex.action_items}
    email = tasks["Email the vendor with the counteroffer"]  # wording from keep (a3)
    # a3 (keep) has no owner, a1 has: owner, due date AND time all come from a1
    assert (email.owner, email.due_date, email.timestamp) == ("Anita", None, 50)
    venue = tasks["Book the venue"]  # keep a4 has the owner, so a4 supplies everything
    assert (venue.owner, venue.due_date, venue.timestamp) == ("Vikram", None, 170)
    assert len(ex.action_items) == 2
    assert {c["details_from"] for c in log["clusters"] if c["kind"] == "action"} == {"a1", "a4"}


def test_due_date_member_used_when_no_member_has_an_owner():
    parts = [Extraction(decisions=[], open_topics=[], action_items=[
        ActionItem(task="Send poll", owner=None, due_date=None, timestamp=10),
        ActionItem(task="Poll the team", owner=None, due_date="Nov 15", timestamp=40)])]
    ex, _ = apply_merge_plan(number_items(parts, [60]),
                             plan(action_clusters=[{"ids": ["a1", "a2"], "keep": "a1"}]))
    [item] = ex.action_items
    assert (item.task, item.due_date, item.timestamp) == ("Send poll", "Nov 15", 40)


def test_ids_left_out_are_kept_unchanged():
    ex, log = merged(plan(action_clusters=[{"ids": ["a1", "a3"], "keep": "a3"}]))
    assert len(ex.action_items) == 3  # a2 and a4 untouched
    assert len(ex.decisions) == 3 and len(ex.open_topics) == 4
    assert log["roles"]["a2"] == log["roles"]["d1"] == log["roles"]["o4"] == "unchanged"


def test_unknown_wrong_kind_and_repeated_ids_are_ignored():
    ex, log = merged(plan(action_clusters=[{"ids": ["a1", "a99", "d1", "a3"], "keep": "a99"},
                                           {"ids": ["a3", "a2"], "keep": "a3"}],
                          decision_clusters=[{"ids": ["d1", "zz"]}],
                          decided_open_topics=[{"open_id": "o77", "decision_id": "d3"}]))
    assert "a99" not in log["roles"] and "zz" not in log["roles"]
    assert log["roles"]["d1"] == "keep"  # cluster of one valid id ("zz" ignored): d1 kept as is
    assert log["roles"]["a3"] == "member"  # used by the first cluster only
    first = next(c for c in log["clusters"] if c["kind"] == "action")
    assert first["ids"] == ["a1", "a3"] and first["keep"] == "a1"  # invalid keep -> earliest member
    assert len(ex.decisions) == 3


def test_later_decision_wins_even_if_listed_first_and_is_recorded():
    ex, log = merged(plan(decision_clusters=[{"ids": ["d2", "d1"]}]))
    texts = [d.text for d in ex.decisions]
    assert "Cap is $16,000." in texts and "Ask for $15,000." not in texts
    assert log["superseded_decisions"] == [{"old_text": "Ask for $15,000.", "old_timestamp": 54,
                                            "new_text": "Cap is $16,000.", "new_timestamp": 64}]
    assert log["roles"]["d1"] == "overridden"


def test_open_topic_dropped_only_when_the_decision_is_later():
    ex, log = merged(plan(decided_open_topics=[
        {"open_id": "o1", "decision_id": "d3"},    # offsite location (part 1, ends 60) <- Pune at 150: drop
        {"open_id": "o4", "decision_id": "d2"}]))  # report ownership (part 3, ends 180) <- 64: keep
    assert "offsite location" not in ex.open_topics and "report ownership" in ex.open_topics
    assert log["dropped_open_topics"] == [{"open_topic": "offsite location",
                                           "decision": "Offsite in Pune.", "decision_timestamp": 150}]


def test_open_topics_clustered():
    ex, log = merged(plan(open_topic_clusters=[{"ids": ["o2", "o4"], "keep": "o4"}]))
    assert ex.open_topics.count("report ownership") == 1 and "report owner" not in ex.open_topics


def test_every_input_id_is_accounted_for_exactly_once():
    numbered = number_items(PARTS, PIECE_ENDS)
    all_ids = [x["id"] for kind in ("actions", "decisions", "opens") for x in numbered[kind]]
    full = plan(action_clusters=[{"ids": ["a1", "a3"], "keep": "a3"}, {"ids": ["a3", "a4"], "keep": "a4"}],
                decision_clusters=[{"ids": ["d1", "d2"]}],
                open_topic_clusters=[{"ids": ["o2", "o4"], "keep": "o4"}, {"ids": ["o1"], "keep": "o1"}],
                decided_open_topics=[{"open_id": "o1", "decision_id": "d3"}])
    _, log = apply_merge_plan(numbered, full)
    assert sorted(log["roles"]) == sorted(all_ids)  # each id exactly once (a dict key)
    assert set(log["roles"].values()) <= {"keep", "member", "unchanged", "overridden", "dropped"}
    overridden = [i for i, r in log["roles"].items() if r == "overridden"]
    dropped = [i for i, r in log["roles"].items() if r == "dropped"]
    assert len(overridden) == len(log["superseded_decisions"]) == 1
    assert len(dropped) == len(log["dropped_open_topics"]) == 1


# ---------- the merge call ----------

def test_merge_call_sends_only_extraction_json(monkeypatch):
    seen = []

    def fake(messages, on_progress=None, **kwargs):
        seen.append(messages)
        return json.dumps(plan(decision_clusters=[{"ids": ["d1", "d2"]}]).model_dump()), {"calls": 1}

    monkeypatch.setattr(analyze, "call_llm", fake)
    ex, log = merge_pieces(PARTS, PIECE_ENDS)
    sent = json.loads(seen[0][1]["content"])
    assert set(sent) == {"action_items", "decisions", "open_topics"}
    assert sent["action_items"][0] == {"id": "a1", "part": 1, "time": 50,
                                       "task": "mail the window with our counter offer",
                                       "owner": "Anita", "due_date": None}
    assert len(ex.decisions) == 2


def test_merge_falls_back_to_code_merge_on_error(monkeypatch):
    monkeypatch.setattr(analyze, "call_llm", lambda *a, **k: ("not json", {"calls": 1}))
    ex, log = merge_pieces(PARTS, PIECE_ENDS)
    assert "simple merge used" in log["fallback"]
    assert len(ex.action_items) >= 3  # the code merge still produced a result


def test_daily_limit_is_not_swallowed_by_the_fallback(monkeypatch):
    def limit(*a, **k):
        raise DailyLimitError("Groq rate limit hit (asked to wait 3600 s).")

    monkeypatch.setattr(analyze, "call_llm", limit)
    with pytest.raises(DailyLimitError):
        merge_pieces(PARTS, PIECE_ENDS)
    assert issubclass(DailyLimitError, AnalyzeError)


def test_single_piece_makes_no_merge_call(monkeypatch):
    calls = []
    summary = {"overview": "o", "key_topics": []}
    extraction = {"decisions": [], "action_items": [], "open_topics": []}

    def fake(messages, on_progress=None, **kwargs):
        calls.append(messages[0]["content"])
        return json.dumps(extraction if "Extract" in messages[0]["content"] else summary), {"calls": 1}

    monkeypatch.setattr(analyze, "call_llm", fake)
    result = analyze.analyze([{"id": 0, "start": 0.0, "end": 5.0, "text": "Hi."}])
    assert len(calls) == 2 and result["merge_log"] is None
    assert result["extraction"]["superseded_decisions"] == []


def test_merge_summaries_prompt_has_the_no_who_or_when_rule():
    assert "Do not say who will do what" in analyze.MERGE_TASK
