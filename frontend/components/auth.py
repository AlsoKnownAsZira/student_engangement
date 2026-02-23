"""
Auth UI components — login / signup forms, session helpers, theme toggle.
All HTML uses compact single-line tags to avoid Streamlit markdown parsing issues.
"""

from __future__ import annotations
import sys
from pathlib import Path

_FRONTEND_DIR = str(Path(__file__).resolve().parent.parent)
if _FRONTEND_DIR not in sys.path:
    sys.path.insert(0, _FRONTEND_DIR)

import requests
import streamlit as st
from services.api_client import APIClient


def init_session_state():
    defaults = {
        "access_token": None, "refresh_token": None,
        "user_id": None, "user_email": None, "theme_mode": "dark",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def is_logged_in() -> bool:
    return st.session_state.get("access_token") is not None


def get_api_client() -> APIClient:
    return APIClient(token=st.session_state.get("access_token"))


def logout():
    for k in ["access_token", "refresh_token", "user_id", "user_email"]:
        st.session_state[k] = None
    st.rerun()


def require_auth():
    init_session_state()
    if not is_logged_in():
        st.warning("Please log in to access this page.")
        st.switch_page("app.py")
        st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# AUTH PAGE
# ═══════════════════════════════════════════════════════════════════════════════

def show_auth_page():
    init_session_state()

    from components.styles import inject_global_css, _palette, init_theme
    init_theme()
    inject_global_css()
    p = _palette()

    # Centered hero
    grad = f"background:linear-gradient(135deg,{p['accent']} 0%,{p['accent_hover']} 50%,#a78bfa 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;"
    title_s = f"{grad}font-size:2.2rem;font-weight:700;margin:0;font-family:Inter,sans-serif;"
    sub_s = f"color:{p['text_secondary']};font-size:1.05rem;margin-top:0.5rem;font-family:Inter,sans-serif;"

    st.markdown(
        f'<div style="text-align:center;padding:2rem 0 1rem 0;">'
        f'<div style="font-size:3.5rem;margin-bottom:0.5rem;">🎓</div>'
        f'<h1 style="{title_s}">Classroom Engagement Analyzer</h1>'
        f'<p style="{sub_s}">AI-powered student engagement analysis from classroom videos</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Form container
    _, center_col, _ = st.columns([1, 2, 1])

    with center_col:
        tab_login, tab_signup = st.tabs(["🔐 Login", "✨ Sign Up"])
        client = APIClient()

        with tab_login:
            with st.form("login_form"):
                st.markdown(f'<p style="color:{p["text_secondary"]};margin-bottom:0.5rem;">Welcome back! Sign in to your account.</p>', unsafe_allow_html=True)
                email = st.text_input("Email", placeholder="you@example.com")
                password = st.text_input("Password", type="password", placeholder="Your password")
                submitted = st.form_submit_button("Login", use_container_width=True, type="primary")

            if submitted:
                if not email or not password:
                    st.error("Please fill in all fields.")
                else:
                    with st.spinner("Logging in…"):
                        try:
                            data = client.login(email, password)
                            _set_session(data)
                            st.success("Logged in!")
                            st.rerun()
                        except requests.exceptions.ConnectionError:
                            st.error("Cannot reach the backend. Is the FastAPI server running on port 8000?")
                        except requests.exceptions.Timeout:
                            st.error("Request timed out. Please try again.")
                        except Exception as e:
                            st.error(f"Login failed: {e}")

        with tab_signup:
            with st.form("signup_form"):
                st.markdown(f'<p style="color:{p["text_secondary"]};margin-bottom:0.5rem;">Create a new account to get started.</p>', unsafe_allow_html=True)
                full_name = st.text_input("Full Name (optional)", placeholder="John Doe")
                email_s = st.text_input("Email", key="signup_email", placeholder="you@example.com")
                password_s = st.text_input("Password", type="password", key="signup_pw", placeholder="Min. 6 characters")
                password_confirm = st.text_input("Confirm Password", type="password", placeholder="Re-enter password")
                submitted_s = st.form_submit_button("Create Account", use_container_width=True, type="primary")

            if submitted_s:
                if not email_s or not password_s:
                    st.error("Please fill in all required fields.")
                elif password_s != password_confirm:
                    st.error("Passwords do not match.")
                elif len(password_s) < 6:
                    st.error("Password must be at least 6 characters.")
                else:
                    with st.spinner("Creating account…"):
                        try:
                            data = client.signup(email_s, password_s, full_name)
                            if data.get("needs_confirmation"):
                                st.success("Account created! Please check your email to confirm your account, then log in.")
                            else:
                                _set_session(data)
                                st.success("Account created! You are now logged in.")
                                st.rerun()
                        except requests.exceptions.ConnectionError:
                            st.error("Cannot reach the backend. Is the FastAPI server running on port 8000?")
                        except requests.exceptions.Timeout:
                            st.error("Request timed out. Please try again.")
                        except Exception as e:
                            st.error(f"Sign up failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

def show_user_sidebar():
    from components.styles import init_theme, is_dark, toggle_theme, _palette
    init_theme()
    p = _palette()

    if is_logged_in():
        email = st.session_state.get("user_email", "User")
        initial = email[0].upper() if email else "U"

        card_s = f"background:{p['bg_card']};border:1px solid {p['border']};border-radius:12px;padding:1rem;margin-bottom:1rem;text-align:center;"
        avatar_s = (
            f"width:48px;height:48px;background:linear-gradient(135deg,{p['accent']},{p['accent_hover']});"
            f"border-radius:50%;display:inline-flex;align-items:center;justify-content:center;"
            f"font-size:1.3rem;font-weight:700;color:#fff;margin-bottom:0.5rem;"
        )
        name_s = f"color:{p['text_primary']};font-weight:600;font-size:0.9rem;word-break:break-all;font-family:Inter,sans-serif;"

        st.sidebar.markdown(
            f'<div style="{card_s}">'
            f'<div style="{avatar_s}">{initial}</div>'
            f'<div style="{name_s}">{email}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Theme toggle
    toggle_label = "☀️ Light" if is_dark() else "🌙 Dark"
    if st.sidebar.button(f"Switch to {toggle_label} Mode", use_container_width=True):
        toggle_theme()
        st.rerun()

    if is_logged_in():
        st.sidebar.divider()
        if st.sidebar.button("🚪 Logout", use_container_width=True):
            logout()


def _set_session(data: dict):
    st.session_state["access_token"] = data["access_token"]
    st.session_state["refresh_token"] = data["refresh_token"]
    st.session_state["user_id"] = data["user_id"]
    st.session_state["user_email"] = data["email"]
