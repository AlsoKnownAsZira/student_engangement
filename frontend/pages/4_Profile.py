"""
Profile page — account info with styled layout.
"""

import sys
from pathlib import Path

_FRONTEND_DIR = str(Path(__file__).resolve().parent.parent)
if _FRONTEND_DIR not in sys.path:
    sys.path.insert(0, _FRONTEND_DIR)

import streamlit as st
from components.auth import require_auth, show_user_sidebar, logout
from components.styles import inject_global_css, hero_section, card, init_theme, _palette
from fe_config import PAGE_TITLE, PAGE_ICON

st.set_page_config(page_title=f"Profile | {PAGE_TITLE}", page_icon=PAGE_ICON, layout="wide")
require_auth()
init_theme()
inject_global_css()
show_user_sidebar()

p = _palette()
email = st.session_state.get("user_email", "—")
user_id = st.session_state.get("user_id", "—")
initial = email[0].upper() if email and email != "—" else "U"

hero_section(
    title="Your Profile",
    subtitle="Account information and settings",
    emoji="👤",
)

# ── Profile card ──────────────────────────────────────────────────────────

_, center, _ = st.columns([1, 2, 1])

with center:
    st.markdown(f"""
    <div style="
        background:{p['bg_card']};
        border:1px solid {p['border']};
        border-radius:16px;
        padding:2rem;
        text-align:center;
        box-shadow:0 4px 16px {p['shadow']};
    ">
        <div style="
            width:80px; height:80px;
            background:linear-gradient(135deg, {p['accent']}, {p['accent_hover']}, #a78bfa);
            border-radius:50%;
            display:inline-flex;
            align-items:center;
            justify-content:center;
            font-size:2rem;
            font-weight:700;
            color:#fff;
            margin-bottom:1rem;
            box-shadow:0 4px 12px rgba(99,102,241,0.3);
        ">{initial}</div>

        <div style="
            color:{p['text_primary']};
            font-size:1.2rem;
            font-weight:700;
            margin-bottom:0.3rem;
            font-family:'Inter',sans-serif;
        ">{email}</div>

        <div style="
            color:{p['text_muted']};
            font-size:0.8rem;
            font-family:monospace;
            margin-bottom:1.5rem;
        ">ID: {user_id[:8]}…</div>

        <div style="
            background:{p['bg_secondary']};
            border-radius:10px;
            padding:1rem;
            border:1px solid {p['border']};
        ">
            <div style="display:flex; justify-content:space-between; margin-bottom:0.5rem;">
                <span style="color:{p['text_secondary']}; font-size:0.88rem;">Email</span>
                <span style="color:{p['text_primary']}; font-size:0.88rem; font-weight:500;">{email}</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="color:{p['text_secondary']}; font-size:0.88rem;">User ID</span>
                <span style="color:{p['text_primary']}; font-size:0.88rem; font-family:monospace;">{user_id[:16]}…</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")
    if st.button("🚪 Logout", type="primary", use_container_width=True):
        logout()
