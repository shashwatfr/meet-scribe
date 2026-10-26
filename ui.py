"""Small helpers shared by the Streamlit pages: the login check, the sidebar, and safe text.

Used by app.py and every page in pages/. Display only; all logic lives in services/.
"""
import re

import streamlit as st

# Characters Markdown (and Streamlit's $...$ LaTeX) would interpret.
_MARKDOWN_CHARS = re.compile(r"([\\`*_{}\[\]()#+\-.!|<>~$:])")


def md(text) -> str:
    """Escape text so st.markdown shows it literally ("$18,000 ... $15,000" stays as typed)."""
    return _MARKDOWN_CHARS.sub(r"\\\1", str(text))


def mmss(seconds) -> str:
    seconds = int(seconds or 0)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def sidebar(user: dict) -> None:
    with st.sidebar:
        st.markdown(f"Signed in as **{md(user['username'])}** ({md(user['role'])})")
        if st.button("Log out"):
            st.session_state.clear()
            st.switch_page("app.py")  # back to the login form, from any page


def require_login() -> dict:
    """Return the signed-in user, or stop the page. Pages can be opened directly by URL."""
    user = st.session_state.get("user")
    if user is None:
        st.warning("Please log in first: open the MeetScribe main page.")
        st.stop()
    sidebar(user)
    return user
