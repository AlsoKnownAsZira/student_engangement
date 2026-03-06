"""
Shared styles & theme system — dark mode with cyan-teal-violet accents.
Premium glassmorphism design with micro-animations.
All HTML is kept compact (single-line tags) to prevent Streamlit markdown parser issues.
"""

from __future__ import annotations
import streamlit as st


# ═══════════════════════════════════════════════════════════════════════════════
# PALETTE (dark mode only)
# ═══════════════════════════════════════════════════════════════════════════════

_P = {
    "bg_primary": "#06090f", "bg_secondary": "#0c1219",
    "bg_card": "rgba(15,23,42,0.65)", "bg_card_hover": "rgba(22,33,56,0.75)",
    "bg_card_solid": "#0f172a",
    "text_primary": "#e2e8f0", "text_secondary": "#94a3b8", "text_muted": "#64748b",
    "border": "rgba(56,189,248,0.10)", "border_hover": "rgba(56,189,248,0.30)",
    "accent": "#38bdf8", "accent_hover": "#67e8f9",
    "accent_subtle": "rgba(56,189,248,0.10)",
    "accent2": "#a78bfa", "accent2_hover": "#c4b5fd",
    "gradient_start": "#0c1219", "gradient_mid": "#0f172a", "gradient_end": "#1e1b4b",
    "success": "#34d399", "warning": "#fbbf24", "danger": "#f87171",
    "shadow": "rgba(0,0,0,0.5)",
    "glass_border": "rgba(56,189,248,0.12)",
    "glow_accent": "rgba(56,189,248,0.15)",
    "glow_violet": "rgba(167,139,250,0.12)",
    "chart_bg": "rgba(0,0,0,0)", "chart_grid": "rgba(148,163,184,0.08)",
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
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

/* ── Animated background orbs ─────────────────────────────────────────── */
@keyframes float-orb-1 {{ 0%,100% {{ transform:translate(0,0) scale(1); }} 50% {{ transform:translate(40px,-30px) scale(1.1); }} }}
@keyframes float-orb-2 {{ 0%,100% {{ transform:translate(0,0) scale(1); }} 50% {{ transform:translate(-30px,40px) scale(1.15); }} }}
@keyframes fade-in {{ from {{ opacity:0;transform:translateY(12px); }} to {{ opacity:1;transform:translateY(0); }} }}
@keyframes shimmer {{ 0% {{ background-position: -200% center; }} 100% {{ background-position: 200% center; }} }}
@keyframes glow-pulse {{ 0%,100% {{ box-shadow: 0 0 15px {p["glow_accent"]}; }} 50% {{ box-shadow: 0 0 30px {p["glow_accent"]}, 0 0 60px {p["glow_violet"]}; }} }}
@keyframes gradient-shift {{ 0% {{ background-position: 0% 50%; }} 50% {{ background-position: 100% 50%; }} 100% {{ background-position: 0% 50%; }} }}

/* ── Base ──────────────────────────────────────────────────────────────── */
.stApp, .stApp > header {{ background-color: {p["bg_primary"]} !important; color: {p["text_primary"]} !important; font-family: 'Inter', sans-serif !important; }}
.stApp::before {{ content:''; position:fixed; top:-50%; left:-50%; width:200%; height:200%; background: radial-gradient(ellipse at 20% 50%, {p["glow_accent"]} 0%, transparent 50%), radial-gradient(ellipse at 80% 20%, {p["glow_violet"]} 0%, transparent 50%), radial-gradient(ellipse at 50% 80%, rgba(52,211,153,0.06) 0%, transparent 50%); animation: float-orb-1 20s ease-in-out infinite; pointer-events:none; z-index:0; }}

/* ── Sidebar ───────────────────────────────────────────────────────────── */
.stApp [data-testid="stSidebar"] {{ background: linear-gradient(180deg, {p["bg_primary"]} 0%, {p["bg_secondary"]} 100%) !important; border-right: 1px solid {p["border"]} !important; backdrop-filter: blur(20px); }}

/* ── Metrics ────────────────────────────────────────────────────────────── */
[data-testid="stMetric"] {{ background: {p["bg_card"]}; border: 1px solid {p["border"]}; border-radius: 16px; padding: 20px 24px; backdrop-filter: blur(12px); box-shadow: 0 4px 16px {p["shadow"]}, inset 0 1px 0 rgba(255,255,255,0.03); transition: all 0.3s cubic-bezier(0.4,0,0.2,1); animation: fade-in 0.5s ease-out both; }}
[data-testid="stMetric"]:hover {{ transform: translateY(-3px); box-shadow: 0 8px 32px {p["shadow"]}, 0 0 20px {p["glow_accent"]}; border-color: {p["border_hover"]}; }}
[data-testid="stMetric"] label {{ color: {p["text_secondary"]} !important; font-weight: 500 !important; font-family: 'Inter', sans-serif !important; letter-spacing: 0.02em; }}
[data-testid="stMetric"] [data-testid="stMetricValue"] {{ color: {p["text_primary"]} !important; font-weight: 700 !important; font-family: 'Outfit', sans-serif !important; }}

/* ── Buttons ────────────────────────────────────────────────────────────── */
.stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"] {{ background: linear-gradient(135deg, {p["accent"]} 0%, {p["accent2"]} 100%) !important; background-size: 200% 200% !important; border: none !important; color: #0f172a !important; font-weight: 700 !important; font-family: 'Outfit', sans-serif !important; letter-spacing: 0.03em; border-radius: 12px !important; padding: 0.65rem 1.8rem !important; transition: all 0.3s cubic-bezier(0.4,0,0.2,1) !important; box-shadow: 0 4px 16px rgba(56,189,248,0.25), 0 0 20px rgba(56,189,248,0.10) !important; }}
.stButton > button[kind="primary"]:hover, .stButton > button[data-testid="stBaseButton-primary"]:hover {{ transform: translateY(-2px) scale(1.01) !important; box-shadow: 0 8px 32px rgba(56,189,248,0.35), 0 0 40px rgba(167,139,250,0.15) !important; animation: gradient-shift 3s ease infinite !important; }}
.stButton > button[kind="secondary"], .stButton > button[data-testid="stBaseButton-secondary"] {{ background: {p["bg_card"]} !important; border: 1px solid {p["border"]} !important; color: {p["text_primary"]} !important; font-weight: 500 !important; font-family: 'Inter', sans-serif !important; border-radius: 12px !important; backdrop-filter: blur(8px) !important; transition: all 0.3s cubic-bezier(0.4,0,0.2,1) !important; }}
.stButton > button[kind="secondary"]:hover, .stButton > button[data-testid="stBaseButton-secondary"]:hover {{ background: {p["bg_card_hover"]} !important; border-color: {p["accent"]} !important; box-shadow: 0 0 15px {p["glow_accent"]} !important; }}
.stButton > button {{ color: {p["text_primary"]} !important; border: 1px solid {p["border"]} !important; border-radius: 12px !important; font-weight: 500 !important; backdrop-filter: blur(8px) !important; transition: all 0.25s ease !important; }}
.stButton > button:hover {{ border-color: {p["border_hover"]} !important; }}

/* ── Tabs ────────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {{ gap: 6px; background: {p["bg_card"]} !important; border-radius: 14px; padding: 5px; backdrop-filter: blur(12px); border: 1px solid {p["border"]}; }}
.stTabs [data-baseweb="tab"] {{ border-radius: 10px !important; color: {p["text_secondary"]} !important; font-weight: 500 !important; font-family: 'Inter', sans-serif !important; transition: all 0.25s ease !important; }}
.stTabs [aria-selected="true"] {{ background: linear-gradient(135deg, {p["accent"]}, {p["accent2"]}) !important; color: #0f172a !important; font-weight: 600 !important; box-shadow: 0 2px 12px rgba(56,189,248,0.25) !important; }}

/* ── File uploader ──────────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {{ border: 2px dashed {p["border_hover"]} !important; border-radius: 16px !important; background: {p["bg_card"]} !important; backdrop-filter: blur(12px) !important; transition: all 0.3s ease !important; }}
[data-testid="stFileUploader"]:hover {{ border-color: {p["accent"]} !important; box-shadow: 0 0 20px {p["glow_accent"]} !important; }}

/* ── DataFrame ──────────────────────────────────────────────────────────── */
.stDataFrame {{ border: 1px solid {p["border"]} !important; border-radius: 14px !important; overflow: hidden !important; backdrop-filter: blur(8px); }}

/* ── Inputs ──────────────────────────────────────────────────────────────── */
hr {{ border-color: {p["border"]} !important; opacity: 0.4; }}
.stTextInput > div > div > input {{ background: {p["bg_card"]} !important; border: 1px solid {p["border"]} !important; border-radius: 10px !important; color: {p["text_primary"]} !important; backdrop-filter: blur(8px); transition: all 0.25s ease !important; }}
.stTextInput > div > div > input:focus {{ border-color: {p["accent"]} !important; box-shadow: 0 0 0 3px {p["accent_subtle"]}, 0 0 15px {p["glow_accent"]} !important; }}

/* ── Alerts ──────────────────────────────────────────────────────────────── */
.stAlert {{ border-radius: 12px !important; border-left: 4px solid !important; backdrop-filter: blur(8px); }}

/* ── Progress bar ───────────────────────────────────────────────────────── */
.stProgress > div > div > div {{ background: linear-gradient(90deg, {p["accent"]}, {p["accent2"]}) !important; border-radius: 6px !important; box-shadow: 0 0 12px rgba(56,189,248,0.3); }}

/* ── Scrollbar ──────────────────────────────────────────────────────────── */
::-webkit-scrollbar {{ width: 5px; height: 5px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: {p["text_muted"]}; border-radius: 10px; }}
::-webkit-scrollbar-thumb:hover {{ background: {p["accent"]}; }}

/* ── Smooth scroll & general animations ─────────────────────────────────── */
html {{ scroll-behavior: smooth; }}
.stMarkdown, .stMetric, .stButton {{ animation: fade-in 0.4s ease-out both; }}
</style>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# HTML HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def hero_section(title: str, subtitle: str, emoji: str = "🎓"):
    p = _P
    hero_style = (
        f"position:relative;overflow:hidden;background:linear-gradient(135deg,{p['gradient_start']} 0%,{p['gradient_mid']} 40%,{p['gradient_end']} 100%);"
        f"border-radius:20px;padding:3rem 2.5rem;margin-bottom:2rem;text-align:center;"
        f"box-shadow:0 8px 32px {p['shadow']},0 0 60px {p['glow_accent']};border:1px solid {p['glass_border']};"
    )
    # Decorative animated orbs inside hero
    orb1 = f"position:absolute;top:-30%;right:-10%;width:300px;height:300px;border-radius:50%;background:radial-gradient(circle,rgba(56,189,248,0.12),transparent 70%);animation:float-orb-1 8s ease-in-out infinite;pointer-events:none;"
    orb2 = f"position:absolute;bottom:-20%;left:-5%;width:250px;height:250px;border-radius:50%;background:radial-gradient(circle,rgba(167,139,250,0.10),transparent 70%);animation:float-orb-2 10s ease-in-out infinite;pointer-events:none;"
    html = (
        f'<div style="{hero_style}">'
        f'<div style="{orb1}"></div>'
        f'<div style="{orb2}"></div>'
        f'<div style="position:relative;z-index:1;">'
        f'<div style="font-size:3rem;margin-bottom:0.5rem;filter:drop-shadow(0 0 12px rgba(56,189,248,0.4));">{emoji}</div>'
        f'<h1 style="color:#67e8f9;font-size:2.2rem;font-weight:800;margin:0 0 0.5rem 0;font-family:Outfit,sans-serif;letter-spacing:-0.02em;text-shadow:0 0 30px rgba(56,189,248,0.3);">{title}</h1>'
        f'<p style="color:#94a3b8;font-size:1.05rem;margin:0;font-family:Inter,sans-serif;max-width:600px;margin:0 auto;">{subtitle}</p>'
        f'</div></div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def card(content_html: str, extra_style: str = ""):
    p = _P
    style = (
        f"background:{p['bg_card']};border:1px solid {p['border']};border-radius:16px;"
        f"padding:1.5rem;margin-bottom:1rem;box-shadow:0 4px 16px {p['shadow']};"
        f"backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);"
        f"transition:all 0.3s cubic-bezier(0.4,0,0.2,1);{extra_style}"
    )
    hover_id = f"card-{hash(content_html) % 100000}"
    st.markdown(f'<div class="glass-card" style="{style}">{content_html}</div>', unsafe_allow_html=True)


def feature_card(emoji: str, title: str, description: str, accent: str = ""):
    p = _P
    ac = accent or p["accent"]
    outer = (
        f"background:{p['bg_card']};border:1px solid {p['border']};"
        f"border-radius:16px;padding:1.6rem;margin-bottom:0.5rem;box-shadow:0 4px 16px {p['shadow']};"
        f"backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);"
        f"transition:all 0.3s cubic-bezier(0.4,0,0.2,1);position:relative;overflow:hidden;"
    )
    # Top accent glow line
    accent_bar = f"position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,transparent,{ac},{p['accent2']},transparent);border-radius:16px 16px 0 0;"
    html = (
        f'<div style="{outer}">'
        f'<div style="{accent_bar}"></div>'
        f'<div style="font-size:1.8rem;margin-bottom:0.6rem;filter:drop-shadow(0 0 8px {ac}40);">{emoji}</div>'
        f'<div style="color:{p["text_primary"]};font-size:1.1rem;font-weight:700;margin-bottom:0.4rem;font-family:Outfit,sans-serif;letter-spacing:-0.01em;">{title}</div>'
        f'<div style="color:{p["text_secondary"]};font-size:0.88rem;line-height:1.5;font-family:Inter,sans-serif;">{description}</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def status_badge(status: str) -> str:
    colors = {"completed": _P["success"], "processing": _P["warning"], "failed": _P["danger"], "uploading": _P["accent"]}
    emojis = {"completed": "✅", "processing": "⏳", "failed": "❌", "uploading": "📤"}
    c = colors.get(status, _P["text_muted"])
    e = emojis.get(status, "❔")
    style = (
        f"display:inline-flex;align-items:center;gap:6px;padding:4px 12px;border-radius:8px;"
        f"font-size:0.82rem;font-weight:600;font-family:Inter,sans-serif;"
        f"background:{c}15;color:{c};border:1px solid {c}30;"
        f"backdrop-filter:blur(4px);"
    )
    return f'<span style="{style}">{e} {status.capitalize()}</span>'


def section_header(title: str, emoji: str = ""):
    p = _P
    html = (
        f'<div style="display:flex;align-items:center;gap:12px;margin:2rem 0 1.2rem 0;">'
        f'<span style="font-size:1.4rem;filter:drop-shadow(0 0 6px {p["glow_accent"]});">{emoji}</span>'
        f'<h3 style="color:{p["text_primary"]};font-weight:700;margin:0;font-family:Outfit,sans-serif;letter-spacing:-0.01em;">{title}</h3>'
        f'<div style="flex:1;height:1px;background:linear-gradient(90deg,{p["accent"]}40,{p["accent2"]}20,transparent);margin-left:12px;"></div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)
