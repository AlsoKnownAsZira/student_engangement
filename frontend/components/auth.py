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
from streamlit_cookies_controller import CookieController
from i18n import t, lang_selector

# Initialize cookie controller to enable persistent sessions
cookie_ctrl = CookieController()


def init_session_state():
    # Attempt to read from cookies
    acc_tok = cookie_ctrl.get("access_token")
    ref_tok = cookie_ctrl.get("refresh_token")
    u_id = cookie_ctrl.get("user_id")
    u_email = cookie_ctrl.get("user_email")

    defaults = {
        "access_token": acc_tok, 
        "refresh_token": ref_tok,
        "user_id": u_id, 
        "user_email": u_email,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
        elif st.session_state[k] is None and v is not None:
            # Sync cookie to session if session was cleared
            st.session_state[k] = v


def is_logged_in() -> bool:
    return st.session_state.get("access_token") is not None


def get_api_client() -> APIClient:
    return APIClient(token=st.session_state.get("access_token"))


def logout():
    for k in ["access_token", "refresh_token", "user_id", "user_email"]:
        st.session_state[k] = None
        cookie_ctrl.remove(k)
    st.rerun()


def require_auth():
    init_session_state()
    if not is_logged_in():
        st.warning(t("require_auth_warn"))
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
        '<div style="font-size:3.5rem;margin-bottom:0.5rem;">🎓</div>'
        '<h1 style="font-size:2.4rem;font-weight:800;margin:0;font-family:Inter,sans-serif;letter-spacing:-0.02em;color:var(--text-color);">Classroom Engagement Analyzer</h1>'
        f'<p style="font-size:1.05rem;margin-top:0.6rem;font-family:Inter,sans-serif;color:var(--text-color);opacity:0.8;">{t("auth_subtitle")}</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Language toggle — visible on auth page too
    _, lang_col, _ = st.columns([3, 1, 3])
    with lang_col:
        options = ["🇮🇩 Indonesia", "🇬🇧 English"]
        from i18n import get_lang
        current_idx = 0 if get_lang() == "ID" else 1
        chosen = st.radio("🌐", options, index=current_idx, horizontal=True,
                          label_visibility="collapsed", key="__lang_radio_auth")
        new_lang = "ID" if "Indonesia" in chosen else "EN"
        if new_lang != get_lang():
            st.session_state["lang"] = new_lang
            st.rerun()

    # Form container
    _, center_col, _ = st.columns([1, 2, 1])

    with center_col:
        tab_login, tab_signup = st.tabs([t("tab_login"), t("tab_signup")])
        client = APIClient()

        with tab_login:
            with st.form("login_form"):
                st.markdown(f'<p style="font-family:Inter,sans-serif;color:var(--text-color);opacity:0.8;margin-bottom:0.5rem;">{t("login_welcome")}</p>', unsafe_allow_html=True)
                email = st.text_input(t("label_email"), placeholder="you@example.com")
                password = st.text_input(t("label_password"), type="password", placeholder=t("ph_password"))
                submitted = st.form_submit_button(t("btn_login"), use_container_width=True, type="primary")

            if submitted:
                if not email or not password:
                    st.error(t("err_fill_all"))
                else:
                    with st.spinner(t("spinner_login")):
                        try:
                            data = client.login(email, password)
                            _set_session(data)
                            st.success(t("success_login"))
                            st.rerun()
                        except requests.exceptions.ConnectionError:
                            st.error(t("err_backend"))
                        except requests.exceptions.Timeout:
                            st.error(t("err_timeout"))
                        except requests.exceptions.HTTPError as e:
                            try:
                                detail = e.response.json().get("detail", "")
                                if "Invalid login credentials" in detail:
                                    st.error(t("err_invalid_creds"))
                                else:
                                    st.error(t("err_login_fail", detail))
                            except Exception:
                                st.error(t("err_login_fail", e))
                        except Exception as e:
                            st.error(t("err_login_fail", e))

        with tab_signup:
            with st.form("signup_form"):
                st.markdown(f'<p style="font-family:Inter,sans-serif;color:var(--text-color);opacity:0.8;margin-bottom:0.5rem;">{t("signup_welcome")}</p>', unsafe_allow_html=True)
                full_name = st.text_input(t("label_fullname"), placeholder=t("ph_fullname"))
                email_s = st.text_input(t("label_email"), key="signup_email", placeholder="you@example.com")
                password_s = st.text_input(t("label_password"), type="password", key="signup_pw", placeholder="Min. 6")
                password_confirm = st.text_input(t("label_confirm_pw"), type="password", placeholder=t("ph_confirm_pw"))
                submitted_s = st.form_submit_button(t("btn_create"), use_container_width=True, type="primary")

            if submitted_s:
                if not email_s or not password_s:
                    st.error(t("err_fill_required"))
                elif password_s != password_confirm:
                    st.error(t("err_pw_mismatch"))
                elif len(password_s) < 6:
                    st.error(t("err_pw_short"))
                else:
                    with st.spinner(t("spinner_signup")):
                        try:
                            data = client.signup(email_s, password_s, full_name)
                            if data.get("needs_confirmation"):
                                st.success(t("success_signup_confirm"))
                            else:
                                _set_session(data)
                                st.success(t("success_signup"))
                                st.rerun()
                        except requests.exceptions.ConnectionError:
                            st.error(t("err_backend"))
                        except requests.exceptions.Timeout:
                            st.error(t("err_timeout"))
                        except requests.exceptions.HTTPError as e:
                            try:
                                detail = e.response.json().get("detail", "")
                                if "already registered" in detail.lower() or "user already exists" in detail.lower():
                                    st.error(t("err_email_taken"))
                                else:
                                    st.error(t("err_signup_fail", detail))
                            except Exception:
                                st.error(t("err_signup_fail", e))
                        except Exception as e:
                            st.error(t("err_signup_fail", e))


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

def show_user_sidebar():
    from components.styles import _palette
    p = _palette()

    if is_logged_in():
        email = st.session_state.get("user_email", "User")
        initial = email[0].upper() if email else "U"

        card_s = "background:var(--secondary-background-color);border:1px solid rgba(128,128,128,0.15);border-radius:12px;padding:1.2rem;margin-bottom:1rem;text-align:center;box-shadow:0 2px 4px rgba(0,0,0,0.05);"
        avatar_s = (
            "width:48px;height:48px;background:var(--primary-color);"
            "border-radius:50%;display:inline-flex;align-items:center;justify-content:center;"
            "font-size:1.3rem;font-weight:700;color:white;margin-bottom:0.5rem;"
            "box-shadow:0 2px 4px rgba(0,0,0,0.1);"
        )
        name_s = "color:var(--text-color);font-weight:600;font-size:0.9rem;word-break:break-all;font-family:Inter,sans-serif;"

        st.sidebar.markdown(
            f'<div style="{card_s}">'
            f'<div style="{avatar_s}">{initial}</div>'
            f'<div style="{name_s}">{email}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.sidebar.divider()
        if st.sidebar.button(t("logout"), use_container_width=True):
            logout()

    st.sidebar.divider()
    lang_selector()


def _set_session(data: dict):
    st.session_state["access_token"] = data["access_token"]
    st.session_state["refresh_token"] = data["refresh_token"]
    st.session_state["user_id"] = data["user_id"]
    st.session_state["user_email"] = data["email"]

    # Persist to cookies
    cookie_ctrl.set("access_token", data["access_token"])
    cookie_ctrl.set("refresh_token", data["refresh_token"])
    cookie_ctrl.set("user_id", data["user_id"])
    cookie_ctrl.set("user_email", data["email"])
