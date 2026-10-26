"""MeetScribe main page: the login form, and links to the other pages once signed in.

Streamlit runs this file first (uv run streamlit run app.py); the files in pages/ appear
in the sidebar. Logging in calls services.auth.verify_login().
"""
import streamlit as st

from services.auth import ensure_demo_users, verify_login
from ui import md, sidebar

st.set_page_config(page_title="MeetScribe")


@st.cache_resource  # once per server process, not on every page refresh
def create_hosted_demo_users() -> list[str]:
    return ensure_demo_users()  # does nothing unless DEMO_PASSWORD_* secrets are set


create_hosted_demo_users()
st.title("MeetScribe")

user = st.session_state.get("user")
if user is None:
    with st.form("login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in")
    if submitted:
        found = verify_login(username, password)
        if found:
            st.session_state["user"] = found  # id, username, role (never the hash)
            st.rerun()
        else:
            st.error("Wrong username or password.")  # same message for unknown users
    st.caption("A browser refresh signs you out; your meetings stay saved.")
else:
    sidebar(user)
    st.markdown(f"Signed in as **{md(user['username'])}**.")
    st.page_link("pages/1_Upload.py", label="Upload a recording")
    st.page_link("pages/2_History.py", label="Your meetings")
    if user["role"] == "admin":
        st.page_link("pages/3_Admin.py", label="Admin")
