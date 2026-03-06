"""
Auth UI components — login / signup forms, session helpers.
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
        "user_id": None, "user_email": None,
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
    st.markdown(
        '<div style="text-align:center;padding:2.5rem 0 1.5rem 0;">'
        '<div style="font-size:3.5rem;margin-bottom:0.5rem;filter:drop-shadow(0 0 16px rgba(56,189,248,0.4));">🎓</div>'
        '<h1 style="color:#67e8f9;font-size:2.4rem;font-weight:800;margin:0;font-family:Outfit,Inter,sans-serif;letter-spacing:-0.02em;text-shadow:0 0 30px rgba(56,189,248,0.3);">Classroom Engagement Analyzer</h1>'
        f'<p style="color:#94a3b8;font-size:1.05rem;margin-top:0.6rem;font-family:Inter,sans-serif;">AI-powered student engagement analysis from classroom videos</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Form container
    _, center_col, _ = st.columns([1, 2, 1])

    with center_col:
        tab_login, tab_signup = st.tabs(["🔐 Login", "✨ Sign Up"])
        client = APIClient()

        with tab_login:
            with st.form("login_form"):
                st.markdown('<p style="color:#94a3b8;margin-bottom:0.5rem;">Welcome back! Sign in to your account.</p>', unsafe_allow_html=True)
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
                st.markdown('<p style="color:#94a3b8;margin-bottom:0.5rem;">Create a new account to get started.</p>', unsafe_allow_html=True)
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
    from components.styles import _palette
    p = _palette()

    if is_logged_in():
        email = st.session_state.get("user_email", "User")
        initial = email[0].upper() if email else "U"

        card_s = f"background:{p['bg_card']};border:1px solid {p['border']};border-radius:16px;padding:1.2rem;margin-bottom:1rem;text-align:center;backdrop-filter:blur(12px);"
        avatar_s = (
            f"width:48px;height:48px;background:linear-gradient(135deg,{p['accent']},{p['accent2']});"
            f"border-radius:50%;display:inline-flex;align-items:center;justify-content:center;"
            f"font-size:1.3rem;font-weight:700;color:#0f172a;margin-bottom:0.5rem;"
            f"box-shadow:0 4px 12px rgba(56,189,248,0.25);"
        )
        name_s = f"color:{p['text_primary']};font-weight:600;font-size:0.9rem;word-break:break-all;font-family:Inter,sans-serif;"

        st.sidebar.markdown(
            f'<div style="{card_s}">'
            f'<div style="{avatar_s}">{initial}</div>'
            f'<div style="{name_s}">{email}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.sidebar.divider()
        if st.sidebar.button("🚪 Logout", use_container_width=True):
            logout()


def _set_session(data: dict):
    st.session_state["access_token"] = data["access_token"]
    st.session_state["refresh_token"] = data["refresh_token"]
    st.session_state["user_id"] = data["user_id"]
    st.session_state["user_email"] = data["email"]
