"""
Profile page — basic account info.
"""

import sys
from pathlib import Path

_FRONTEND_DIR = str(Path(__file__).resolve().parent.parent)
if _FRONTEND_DIR not in sys.path:
    sys.path.insert(0, _FRONTEND_DIR)

import streamlit as st
from components.auth import require_auth, show_user_sidebar, logout
from fe_config import PAGE_TITLE, PAGE_ICON

st.set_page_config(page_title=f"Profile | {PAGE_TITLE}", page_icon=PAGE_ICON)
require_auth()
show_user_sidebar()

st.title("Profile")

st.markdown(f"**Email:** {st.session_state.get('user_email', '—')}")
st.markdown(f"**User ID:** `{st.session_state.get('user_id', '—')}`")

st.divider()

if st.button("Logout", type="primary"):
    logout()
