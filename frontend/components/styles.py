"""
Shared styles & theme system — dark mode with navy-purple accents.
All HTML is kept compact (single-line tags) to prevent Streamlit markdown parser issues.
"""

from __future__ import annotations
import streamlit as st


# ═══════════════════════════════════════════════════════════════════════════════
# PALETTE (dark mode only)
# ═══════════════════════════════════════════════════════════════════════════════

_P = {
    "bg_primary": "#0b1120", "bg_secondary": "#111827",
    "bg_card": "#1e293b", "bg_card_hover": "#263348",
    "text_primary": "#f1f5f9", "text_secondary": "#94a3b8", "text_muted": "#64748b",
    "border": "#334155",
    "accent": "#6366f1", "accent_hover": "#818cf8",
    "accent_subtle": "rgba(99,102,241,0.12)",
    "gradient_start": "#1e1b4b", "gradient_mid": "#312e81", "gradient_end": "#4338ca",
    "success": "#2ecc71", "warning": "#f39c12", "danger": "#e74c3c",
    "shadow": "rgba(0,0,0,0.4)",
    "glass_border": "rgba(99,102,241,0.2)",
    "chart_bg": "rgba(0,0,0,0)", "chart_grid": "rgba(148,163,184,0.1)",
    "chart_text": "#94a3b8",
}


def _palette() -> dict:
    return _P


def get_chart_colors() -> dict:
    return {
        "bg": _P["chart_bg"], "grid": _P["chart_grid"], "text": _P["chart_text"],
        "paper_bg": _P["chart_bg"], "font_color": _P["text_primary"],
    }


def init_theme():
    """No-op kept for compatibility."""
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# INJECT CSS
# ═══════════════════════════════════════════════════════════════════════════════

def inject_global_css():
    p = _P
    st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
.stApp, .stApp > header {{ background-color: {p["bg_primary"]} !important; color: {p["text_primary"]} !important; font-family: 'Inter', sans-serif !important; }}
.stApp [data-testid="stSidebar"] {{ background: linear-gradient(180deg, #0b1120 0%, #131c33 100%) !important; border-right: 1px solid {p["border"]} !important; }}
[data-testid="stMetric"] {{ background: {p["bg_card"]}; border: 1px solid {p["border"]}; border-radius: 12px; padding: 16px 20px; box-shadow: 0 2px 8px {p["shadow"]}; transition: transform 0.2s ease, box-shadow 0.2s ease; }}
[data-testid="stMetric"]:hover {{ transform: translateY(-2px); box-shadow: 0 6px 20px {p["shadow"]}; }}
[data-testid="stMetric"] label {{ color: {p["text_secondary"]} !important; font-weight: 500 !important; }}
[data-testid="stMetric"] [data-testid="stMetricValue"] {{ color: {p["text_primary"]} !important; font-weight: 700 !important; }}
.stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"] {{ background: linear-gradient(135deg, {p["accent"]} 0%, {p["accent_hover"]} 100%) !important; border: none !important; color: #fff !important; font-weight: 600 !important; border-radius: 10px !important; padding: 0.6rem 1.5rem !important; transition: all 0.2s ease !important; box-shadow: 0 2px 8px rgba(99,102,241,0.3) !important; }}
.stButton > button[kind="primary"]:hover, .stButton > button[data-testid="stBaseButton-primary"]:hover {{ transform: translateY(-1px) !important; box-shadow: 0 4px 16px rgba(99,102,241,0.4) !important; }}
.stButton > button[kind="secondary"], .stButton > button[data-testid="stBaseButton-secondary"] {{ background: {p["bg_card"]} !important; border: 1px solid {p["border"]} !important; color: {p["text_primary"]} !important; font-weight: 500 !important; border-radius: 10px !important; transition: all 0.2s ease !important; }}
.stButton > button[kind="secondary"]:hover, .stButton > button[data-testid="stBaseButton-secondary"]:hover {{ background: {p["bg_card_hover"]} !important; border-color: {p["accent"]} !important; }}
.stButton > button {{ color: {p["text_primary"]} !important; border: 1px solid {p["border"]} !important; border-radius: 10px !important; font-weight: 500 !important; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 8px; background: {p["bg_secondary"]} !important; border-radius: 12px; padding: 4px; }}
.stTabs [data-baseweb="tab"] {{ border-radius: 8px !important; color: {p["text_secondary"]} !important; font-weight: 500 !important; }}
.stTabs [aria-selected="true"] {{ background: {p["accent"]} !important; color: #fff !important; }}
[data-testid="stFileUploader"] {{ border: 2px dashed {p["border"]} !important; border-radius: 12px !important; background: {p["bg_card"]} !important; }}
[data-testid="stFileUploader"]:hover {{ border-color: {p["accent"]} !important; }}
.stDataFrame {{ border: 1px solid {p["border"]} !important; border-radius: 10px !important; overflow: hidden !important; }}
hr {{ border-color: {p["border"]} !important; opacity: 0.5; }}
.stTextInput > div > div > input {{ background: {p["bg_card"]} !important; border: 1px solid {p["border"]} !important; border-radius: 8px !important; color: {p["text_primary"]} !important; }}
.stTextInput > div > div > input:focus {{ border-color: {p["accent"]} !important; box-shadow: 0 0 0 2px {p["accent_subtle"]} !important; }}
.stAlert {{ border-radius: 10px !important; border-left: 4px solid !important; }}
.stProgress > div > div > div {{ background: linear-gradient(90deg, {p["accent"]}, {p["accent_hover"]}) !important; border-radius: 4px !important; }}
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: {p["bg_secondary"]}; }}
::-webkit-scrollbar-thumb {{ background: {p["text_muted"]}; border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: {p["accent"]}; }}
</style>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# HTML HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def hero_section(title: str, subtitle: str, emoji: str = "🎓"):
    p = _P
    hero_style = (
        f"background:linear-gradient(135deg,{p['gradient_start']} 0%,{p['gradient_mid']} 50%,{p['gradient_end']} 100%);"
        f"border-radius:16px;padding:2.5rem 2rem;margin-bottom:1.5rem;text-align:center;"
        f"box-shadow:0 4px 20px {p['shadow']};border:1px solid {p['glass_border']};"
    )
    html = (
        f'<div style="{hero_style}">'
        f'<div style="font-size:2.8rem;margin-bottom:0.3rem;">{emoji}</div>'
        f'<h1 style="color:#f1f5f9;font-size:2rem;font-weight:700;margin:0 0 0.4rem 0;font-family:Inter,sans-serif;">{title}</h1>'
        f'<p style="color:#cbd5e1;font-size:1.05rem;margin:0;font-family:Inter,sans-serif;">{subtitle}</p>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def card(content_html: str, extra_style: str = ""):
    p = _P
    style = (
        f"background:{p['bg_card']};border:1px solid {p['border']};border-radius:14px;"
        f"padding:1.5rem;margin-bottom:1rem;box-shadow:0 2px 10px {p['shadow']};"
        f"transition:transform 0.2s ease,box-shadow 0.2s ease;{extra_style}"
    )
    st.markdown(f'<div style="{style}">{content_html}</div>', unsafe_allow_html=True)


def feature_card(emoji: str, title: str, description: str, accent: str = ""):
    p = _P
    ac = accent or p["accent"]
    outer = (
        f"background:{p['bg_card']};border:1px solid {p['border']};border-left:4px solid {ac};"
        f"border-radius:12px;padding:1.4rem;margin-bottom:0.5rem;box-shadow:0 2px 8px {p['shadow']};"
    )
    html = (
        f'<div style="{outer}">'
        f'<div style="font-size:1.6rem;margin-bottom:0.4rem;">{emoji}</div>'
        f'<div style="color:#f1f5f9;font-size:1.05rem;font-weight:600;margin-bottom:0.3rem;font-family:Inter,sans-serif;">{title}</div>'
        f'<div style="color:#94a3b8;font-size:0.9rem;font-family:Inter,sans-serif;">{description}</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def status_badge(status: str) -> str:
    colors = {"completed": _P["success"], "processing": _P["warning"], "failed": _P["danger"], "uploading": _P["accent"]}
    emojis = {"completed": "✅", "processing": "⏳", "failed": "❌", "uploading": "📤"}
    c = colors.get(status, _P["text_muted"])
    e = emojis.get(status, "❔")
    style = f"display:inline-block;padding:3px 10px;border-radius:6px;font-size:0.82rem;font-weight:600;background:{c}22;color:{c};border:1px solid {c}44;"
    return f'<span style="{style}">{e} {status.capitalize()}</span>'


def section_header(title: str, emoji: str = ""):
    p = _P
    html = (
        f'<div style="display:flex;align-items:center;gap:10px;margin:1.5rem 0 1rem 0;">'
        f'<span style="font-size:1.4rem;">{emoji}</span>'
        f'<h3 style="color:#f1f5f9;font-weight:700;margin:0;font-family:Inter,sans-serif;">{title}</h3>'
        f'<div style="flex:1;height:1px;background:linear-gradient(90deg,{p["border"]},transparent);margin-left:12px;"></div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)
