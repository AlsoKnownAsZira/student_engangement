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
    """Ensure auth-related keys exist in session state."""
    defaults = {
        "access_token": None,
        "refresh_token": None,
        "user_id": None,
        "user_email": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def is_logged_in() -> bool:
    return st.session_state.get("access_token") is not None


def get_api_client() -> APIClient:
    """Return an APIClient with the current JWT token."""
    return APIClient(token=st.session_state.get("access_token"))


def logout():
    for k in ["access_token", "refresh_token", "user_id", "user_email"]:
        st.session_state[k] = None
    st.rerun()


def require_auth():
    """Call at the top of any protected page. Stops execution if not logged in."""
    init_session_state()
    if not is_logged_in():
        st.warning("Please log in to access this page.")
        st.switch_page("app.py")
        st.stop()


def show_auth_page():
    """Render the login / signup form. Called from the main app."""
    init_session_state()

    st.title("🎓 Classroom Engagement Analyzer")
    st.markdown("Analyze student engagement from classroom videos using AI.")

    tab_login, tab_signup = st.tabs(["Login", "Sign Up"])

    client = APIClient()

    # ── Login tab ─────────────────────────────────────────────────────────
    with tab_login:
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", use_container_width=True)

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

    # ── Signup tab ────────────────────────────────────────────────────────
    with tab_signup:
        with st.form("signup_form"):
            full_name = st.text_input("Full Name (optional)")
            email_s = st.text_input("Email", key="signup_email")
            password_s = st.text_input("Password", type="password", key="signup_pw")
            password_confirm = st.text_input("Confirm Password", type="password")
            submitted_s = st.form_submit_button("Create Account", use_container_width=True)

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
                            st.success(
                                "Account created! Please check your email to confirm "
                                "your account, then log in."
                            )
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


def show_user_sidebar():
    """Render user info + logout button in sidebar."""
    if is_logged_in():
        st.sidebar.markdown(f"**{st.session_state['user_email']}**")
        if st.sidebar.button("Logout"):
            logout()


def _set_session(data: dict):
    st.session_state["access_token"] = data["access_token"]
    st.session_state["refresh_token"] = data["refresh_token"]
    st.session_state["user_id"] = data["user_id"]
    st.session_state["user_email"] = data["email"]
