"""History page: the signed-in user's meetings with summary, decisions, action items,
transcript, Word download, and an Ask tab for questions.

Calls services/store.py, pipeline.py (retries), export.py and qa.py. Signed-in users only.
"""
import logging

import streamlit as st

from services.analyze import AnalyzeError, segments_around
from services.export import build_docx, export_filename
from services.index import EmbeddingError, indexed_ids, unindexed_ids
from services.pipeline import PipelineError, retry_analysis, retry_index
from services.qa import answer
from services.store import delete_meeting, get_meeting, get_segments, list_meetings
from ui import md, mmss, require_login

log = logging.getLogger("meetscribe")

user = require_login()
st.title("Your meetings")

meetings = list_meetings(user["id"])  # also marks interrupted meetings as failed
if not meetings:
    st.info("No meetings yet. Upload a recording first.")
    st.stop()

labels = {m["id"]: f"{m['title']} | {m['created_at'][:16].replace('T', ' ')} UTC | {m['status']}"
          for m in meetings}
meeting_id = st.selectbox("Meeting", list(labels), format_func=labels.get)
meeting = get_meeting(user["id"], meeting_id)  # scoped to this user
if meeting is None:
    st.error("Meeting not found.")
    st.stop()

transcript = meeting["transcript"]
segments = transcript["segments"] if transcript else []
is_indexed = meeting_id in indexed_ids(user["id"])
st.subheader(md(meeting["title"]))
details = [meeting["source_filename"], f"duration {mmss(meeting['duration_seconds'])}",
           f"language {(transcript or {}).get('language') or 'unknown'}",
           f"created {meeting['created_at'][:16].replace('T', ' ')} UTC",
           f"indexed for questions: {'yes' if is_indexed else 'no'}"]
st.caption(md(" · ".join(details)))
if meeting["timings"]:
    st.caption(md("Timings: " + " · ".join(f"{k} {v:.1f} s" for k, v in meeting["timings"].items())))

status = meeting["status"]
if status in ("done", "analysis_failed"):
    docx_bytes = build_docx(user["id"], meeting_id)  # None if there is no transcript
    if docx_bytes:
        st.download_button(
            "Download as Word", docx_bytes,
            file_name=export_filename(meeting["title"], meeting["created_at"]),
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            on_click="ignore")  # download only; the page does not reload

if status == "failed":
    st.error(md(meeting["error_message"] or "Processing failed."))
elif status in ("transcribing", "transcribed"):
    st.info("Still processing. If the upload page was closed, this meeting will be "
            "marked failed after a while.")
elif status == "analysis_failed":
    st.error(md(meeting["error_message"] or "The analysis failed."))
    if st.button("Retry analysis", type="primary"):
        with st.status("Analysing again...", expanded=True) as box:
            try:
                retry_analysis(user["id"], meeting_id,
                               on_progress=lambda message: box.write(md(message)))
            except PipelineError as err:
                log.exception("Retry failed for meeting %s", err.meeting_id)
                st.error(f"Something went wrong, meeting {err.meeting_id} was marked failed.")
                st.stop()
        st.rerun()


def show_sources(t: float) -> None:
    with st.expander("Show the transcript lines around this"):
        for seg in segments_around(segments, t):
            st.markdown(f"`{mmss(seg['start'])}` {md(seg['text'])}")


def show_cited_lines(citation: dict) -> None:
    """The transcript segments inside a cited chunk's time range."""
    source = segments if citation["meeting_id"] == meeting_id \
        else (get_segments(user["id"], citation["meeting_id"]) or [])
    with st.expander("Show the cited lines"):
        for seg in source:
            if citation["start"] <= seg["start"] <= citation["end"]:
                st.markdown(f"`{mmss(seg['start'])}` {md(seg['text'])}")


def retry_index_button() -> None:
    if st.button("Retry indexing"):
        try:
            with st.spinner("Indexing..."):
                retry_index(user["id"], meeting_id)
        except EmbeddingError as err:
            st.error(md(err))
            return
        st.rerun()


def show_ask_tab() -> None:
    scope = st.radio("Search in", ["This meeting", "All my meetings"], horizontal=True)
    if scope == "This meeting" and not is_indexed:
        st.info("This meeting is not indexed for questions yet.")
        retry_index_button()
        return
    if scope == "All my meetings" and (missing := len(unindexed_ids(user["id"]))):
        st.caption(f"{missing} of your meetings are not indexed and are not searched.")

    # Questions and answers live only in this browser session (not saved).
    history = st.session_state.setdefault("qa_history", {}).setdefault((meeting_id, scope), [])
    for item in history:
        st.markdown(f"**Q:** {md(item['question'])}")
        result = item["result"]
        st.markdown(md(result["answer"]))
        for c in result["citations"]:
            st.markdown(f"`[{mmss(c['start'])}]` {md(c['title'])}")
            show_cited_lines(c)
        st.divider()

    question = st.chat_input("Ask a question about " +
                             ("this meeting" if scope == "This meeting" else "all your meetings"))
    if question:
        try:
            with st.spinner("Searching and answering..."):
                result = answer(user["id"], question,
                                meeting_id if scope == "This meeting" else None)
        except (EmbeddingError, AnalyzeError) as err:
            st.error(md(err))
            return
        history.append({"question": question, "result": result})
        st.rerun()


analysis = meeting["analysis"]
tab_names = (["Summary", "Decisions and Action Items"] if analysis else []) + \
    (["Transcript", "Ask"] if segments else [])
tabs = dict(zip(tab_names, st.tabs(tab_names))) if tab_names else {}
transcript_tab = tabs.get("Transcript")
if "Ask" in tabs:
    with tabs["Ask"]:
        show_ask_tab()
if analysis:
    summary_tab, items_tab = tabs["Summary"], tabs["Decisions and Action Items"]
    with summary_tab:
        st.markdown(md(analysis["summary"]["overview"]))
        st.markdown("**Key topics**")
        for topic in analysis["summary"]["key_topics"]:
            st.markdown(f"- {md(topic)}")
    with items_tab:
        extraction = analysis["extraction"]
        st.markdown("#### Decisions")
        for d in extraction["decisions"] or []:
            st.markdown(f"**[{mmss(d['timestamp'])}]** {md(d['text'])}")
            show_sources(d["timestamp"])
        if not extraction["decisions"]:
            st.markdown("None.")
        st.markdown("#### Action items")
        st.caption("Owner is shown only when the conversation names them.")
        for a in extraction["action_items"]:
            st.markdown(f"**[{mmss(a['timestamp'])}]** {md(a['task'])}  \n"
                        f"Owner: {md(a['owner'] or '-')} · Due: {md(a['due_date'] or '-')}")
            show_sources(a["timestamp"])
        if not extraction["action_items"]:
            st.markdown("None.")
        st.markdown("#### Open topics")
        for topic in extraction["open_topics"] or ["None."]:
            st.markdown(f"- {md(topic)}")

if transcript_tab is not None:
    with transcript_tab:
        for seg in segments:
            st.markdown(f"`{mmss(seg['start'])}` {md(seg['text'])}")

st.divider()
if st.session_state.get("confirm_delete") == meeting_id:
    st.warning("Delete this meeting, its transcript and its analysis permanently?")
    yes, cancel = st.columns(2)
    if yes.button("Yes, delete"):
        delete_meeting(user["id"], meeting_id)
        del st.session_state["confirm_delete"]
        st.rerun()
    if cancel.button("Cancel"):
        del st.session_state["confirm_delete"]
        st.rerun()
elif st.button("Delete meeting"):
    st.session_state["confirm_delete"] = meeting_id
    st.rerun()
