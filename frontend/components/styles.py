"""
Shared styles & theme system — Clean, modern, and accessible design.
Relies on Streamlit's native CSS variables for seamless Light/Dark mode support.
"""

from __future__ import annotations
import streamlit as st


# ═══════════════════════════════════════════════════════════════════════════════
# PALETTE & CONFIG (Using Streamlit CSS Variables where possible)
# ═══════════════════════════════════════════════════════════════════════════════

def _palette() -> dict:
    # Kept for compatibility if some files still import it
    return {
        "bg_primary": "var(--background-color)", "bg_secondary": "var(--secondary-background-color)",
        "bg_card": "var(--secondary-background-color)", "bg_card_hover": "var(--secondary-background-color)",
        "text_primary": "var(--text-color)", "text_secondary": "var(--text-color)", "text_muted": "var(--text-color)",
        "border": "rgba(128, 128, 128, 0.2)", "border_hover": "var(--primary-color)",
        "accent": "var(--primary-color)", "accent2": "var(--primary-color)",
        "success": "#10b981", "warning": "#f59e0b", "danger": "#ef4444",
        "shadow": "rgba(0, 0, 0, 0.05)",
    }

def get_chart_colors() -> dict:
    return {
        "bg": "rgba(0,0,0,0)", "grid": "rgba(128, 128, 128, 0.1)", "text": "var(--text-color)",
        "paper_bg": "rgba(0,0,0,0)", "font_color": "var(--text-color)",
    }

def init_theme():
    """No-op kept for compatibility."""
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# INJECT CSS
# ═══════════════════════════════════════════════════════════════════════════════

def inject_global_css():
    st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ── Base ──────────────────────────────────────────────────────────────── */
.stApp {
    font-family: 'Inter', sans-serif !important;
}

/* ── Metrics ────────────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: var(--secondary-background-color);
    border: 1px solid rgba(128, 128, 128, 0.15);
    border-radius: 12px;
    padding: 20px 24px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    transition: all 0.2s ease;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
}
[data-testid="stMetric"] label {
    font-weight: 500 !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-weight: 700 !important;
}

/* ── Buttons ────────────────────────────────────────────────────────────── */
.stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"] {
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
}
.stButton > button[kind="primary"]:hover, .stButton > button[data-testid="stBaseButton-primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15) !important;
}
.stButton > button[kind="secondary"], .stButton > button[data-testid="stBaseButton-secondary"] {
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.2s ease !important;
}

/* ── Tabs ────────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 6px 6px 0 0 !important;
    font-weight: 500 !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── File uploader ──────────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    border-radius: 12px !important;
    background: var(--secondary-background-color) !important;
    border: 2px dashed rgba(128, 128, 128, 0.3) !important;
    transition: all 0.2s ease !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--primary-color) !important;
    background: rgba(128, 128, 128, 0.05) !important;
}

/* ── DataFrame ──────────────────────────────────────────────────────────── */
.stDataFrame {
    border-radius: 8px !important;
    overflow: hidden !important;
    border: 1px solid rgba(128, 128, 128, 0.15) !important;
}

/* ── Inputs ──────────────────────────────────────────────────────────────── */
.stTextInput > div > div > input {
    border-radius: 8px !important;
}

/* ── Alerts ──────────────────────────────────────────────────────────────── */
.stAlert {
    border-radius: 8px !important;
}

/* ── Smooth scroll ──────────────────────────────────────────────────────── */
html { scroll-behavior: smooth; }
</style>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# HTML HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def hero_section(title: str, subtitle: str, emoji: str = "🎓"):
    style = (
        "padding: 3rem 2rem; margin-bottom: 2rem; text-align: center;"
        "background: var(--secondary-background-color); border-radius: 16px;"
        "box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05); border: 1px solid rgba(128, 128, 128, 0.15);"
    )
    html = (
        f'<div style="{style}">'
        f'<div style="font-size:3rem;margin-bottom:0.5rem;">{emoji}</div>'
        f'<h1 style="font-size:2.2rem;font-weight:800;margin:0 0 0.5rem 0;font-family:Inter,sans-serif;color:var(--text-color);">{title}</h1>'
        f'<p style="font-size:1.1rem;margin:0;font-family:Inter,sans-serif;max-width:600px;margin:0 auto;color:var(--text-color);opacity:0.8;">{subtitle}</p>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)

def card(content_html: str, extra_style: str = ""):
    style = (
        "background: var(--secondary-background-color); border: 1px solid rgba(128, 128, 128, 0.15); border-radius: 12px;"
        "padding: 1.5rem; margin-bottom: 1rem; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);"
        f"transition: all 0.2s ease; {extra_style}"
    )
    st.markdown(f'<div style="{style}">{content_html}</div>', unsafe_allow_html=True)

def feature_card(emoji: str, title: str, description: str, accent: str = ""):
    outer = (
        "background: var(--secondary-background-color); border: 1px solid rgba(128, 128, 128, 0.15);"
        "border-radius: 12px; padding: 1.5rem; margin-bottom: 0.5rem; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);"
        "transition: all 0.2s ease; position: relative; overflow: hidden;"
    )
    accent_bar = f"position:absolute;top:0;left:0;right:0;height:4px;background:var(--primary-color);opacity:0.8;"
    html = (
        f'<div style="{outer}">'
        f'<div style="{accent_bar}"></div>'
        f'<div style="font-size:1.8rem;margin-bottom:0.6rem;">{emoji}</div>'
        f'<div style="font-size:1.1rem;font-weight:700;margin-bottom:0.4rem;font-family:Inter,sans-serif;color:var(--text-color);">{title}</div>'
        f'<div style="font-size:0.9rem;line-height:1.5;font-family:Inter,sans-serif;color:var(--text-color);opacity:0.7;">{description}</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)

def status_badge(status: str) -> str:
    colors = {"completed": "#10b981", "processing": "#f59e0b", "failed": "#ef4444", "uploading": "var(--primary-color)"}
    emojis = {"completed": "✅", "processing": "⏳", "failed": "❌", "uploading": "📤"}
    c = colors.get(status, "gray")
    e = emojis.get(status, "❔")
    style = (
        f"display:inline-flex;align-items:center;gap:6px;padding:4px 12px;border-radius:12px;"
        f"font-size:0.85rem;font-weight:600;font-family:Inter,sans-serif;"
        f"background:{c}15;color:{c}; border: 1px solid {c}30;"
    )
    return f'<span style="{style}">{e} {status.capitalize()}</span>'

def section_header(title: str, emoji: str = ""):
    html = (
        f'<div style="display:flex;align-items:center;gap:12px;margin:2rem 0 1rem 0;">'
        f'<span style="font-size:1.4rem;">{emoji}</span>'
        f'<h3 style="font-weight:700;margin:0;font-family:Inter,sans-serif;color:var(--text-color);">{title}</h3>'
        f'<div style="flex:1;height:1px;background:rgba(128,128,128,0.2);margin-left:12px;"></div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)
