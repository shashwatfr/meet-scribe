"""Upload page: pick a recording, process it, and watch each step as it happens.

Streamlit lists it in the sidebar. It calls services.pipeline.process_upload() and only
works for a signed-in user (ui.require_login).
"""
import logging

import streamlit as st

from services.audio import ALLOWED_EXTENSIONS, AudioError
from services.pipeline import PipelineError, process_upload
from ui import md, require_login

log = logging.getLogger("meetscribe")

user = require_login()
st.title("Upload a recording")

max_mb = st.get_option("server.maxUploadSize")
st.caption(f"Max upload size: {max_mb:,} MB. Types: {', '.join(ALLOWED_EXTENSIONS)}")
st.info("Tip: video files are large. Extract the audio first and upload that instead:  \n"
        "`ffmpeg -i meeting.mp4 -vn -ac 1 -ar 16000 -b:a 32k meeting.mp3`  \n"
        f"At about 58 MB per minute of video, {max_mb:,} MB is roughly "
        f"{round(max_mb / 58)} minutes of video.")


def start() -> None:
    st.session_state["running"] = True


running = st.session_state.get("running", False)
uploaded = st.file_uploader("Recording", type=[e.lstrip(".") for e in ALLOWED_EXTENSIONS],
                            disabled=running)
title = st.text_input("Title (optional)", placeholder="Defaults to the file name",
                      disabled=running)
st.button("Process recording", type="primary", on_click=start,
          disabled=running or uploaded is None)

if running and uploaded is not None:
    st.warning("Stay on this page until processing finishes.")
    with st.status("Processing...", expanded=True) as status:
        try:
            meeting = process_upload(uploaded, uploaded.name, user["id"],
                                     title.strip() or None,
                                     on_progress=lambda message: status.write(md(message)))
            st.session_state["last_meeting"] = (meeting["id"], meeting["title"], meeting["status"])
            status.update(label="Finished", state="complete")
        except AudioError as err:  # bad file type etc.; nothing was saved
            status.update(label="Could not process this file", state="error")
            st.error(md(err))
        except PipelineError as err:
            log.exception("Pipeline failed for meeting %s", err.meeting_id)  # traceback to terminal
            status.update(label="Failed", state="error")
            st.error(f"Something went wrong, meeting {err.meeting_id} was marked failed.")
        except Exception:
            log.exception("Upload failed before a meeting was created")
            status.update(label="Failed", state="error")
            st.error("Something went wrong. Please try again.")
        finally:
            st.session_state["running"] = False
elif running:  # the file was removed while starting
    st.session_state["running"] = False

if "last_meeting" in st.session_state:
    meeting_id, meeting_title, meeting_status = st.session_state["last_meeting"]
    st.success(f"Meeting {meeting_id}: {md(meeting_title)} - status {md(meeting_status)}")
    st.page_link("pages/2_History.py", label="Open it in your meetings")
