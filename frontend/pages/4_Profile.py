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
from components.styles import inject_global_css, hero_section, init_theme, _palette
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

hero_section(title="Your Profile", subtitle="Account information and settings", emoji="👤")

# ── Profile card ──────────────────────────────────────────────────────────

_, center, _ = st.columns([1, 2, 1])

with center:
    card_style = (
        f"background:{p['bg_card']};border:1px solid {p['border']};border-radius:16px;"
        f"padding:2rem;text-align:center;box-shadow:0 4px 16px {p['shadow']};"
    )
    avatar_style = (
        f"width:80px;height:80px;background:linear-gradient(135deg,{p['accent']},{p['accent_hover']},#a78bfa);"
        f"border-radius:50%;display:inline-flex;align-items:center;justify-content:center;"
        f"font-size:2rem;font-weight:700;color:#fff !important;margin-bottom:1rem;box-shadow:0 4px 12px rgba(99,102,241,0.3);"
    )
    name_style = f"color:{p['text_primary']} !important;font-size:1.2rem;font-weight:700;margin-bottom:0.3rem;font-family:Inter,sans-serif;"
    id_style = f"color:{p['text_muted']} !important;font-size:0.8rem;font-family:monospace;margin-bottom:1.5rem;"
    info_style = f"background:{p['bg_secondary']};border-radius:10px;padding:1rem;border:1px solid {p['border']};"
    row_style = "display:flex;justify-content:space-between;margin-bottom:0.5rem;"
    label_s = f"color:{p['text_secondary']} !important;font-size:0.88rem;"
    val_s = f"color:{p['text_primary']} !important;font-size:0.88rem;font-weight:500;"

    uid_short = user_id[:8] if user_id != "—" else "—"
    uid_med = user_id[:16] if user_id != "—" else "—"

    st.markdown(
        f'<div style="{card_style}">'
        f'<div style="{avatar_style}">{initial}</div>'
        f'<div style="{name_style}">{email}</div>'
        f'<div style="{id_style}">ID: {uid_short}…</div>'
        f'<div style="{info_style}">'
        f'<div style="{row_style}"><span style="{label_s}">Email</span><span style="{val_s}">{email}</span></div>'
        f'<div style="display:flex;justify-content:space-between;"><span style="{label_s}">User ID</span><span style="{val_s}font-family:monospace;">{uid_med}…</span></div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown("")
    if st.button("🚪 Logout", type="primary", use_container_width=True):
        logout()
