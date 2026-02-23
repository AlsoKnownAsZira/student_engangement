"""
Shared styles & theme system — dark / light mode with navy-purple accents.
"""

from __future__ import annotations
import streamlit as st


# ═══════════════════════════════════════════════════════════════════════════════
# THEME INITIALISATION
# ═══════════════════════════════════════════════════════════════════════════════

def init_theme():
    """Ensure a theme mode exists in session state."""
    if "theme_mode" not in st.session_state:
        st.session_state["theme_mode"] = "dark"


def toggle_theme():
    """Flip between dark and light mode."""
    st.session_state["theme_mode"] = (
        "light" if st.session_state["theme_mode"] == "dark" else "dark"
    )


def is_dark() -> bool:
    init_theme()
    return st.session_state["theme_mode"] == "dark"


# ═══════════════════════════════════════════════════════════════════════════════
# CSS VARIABLES — palette per mode
# ═══════════════════════════════════════════════════════════════════════════════

_DARK = {
    "bg_primary":     "#0b1120",
    "bg_secondary":   "#111827",
    "bg_card":        "#1e293b",
    "bg_card_hover":  "#263348",
    "text_primary":   "#f1f5f9",
    "text_secondary": "#94a3b8",
    "text_muted":     "#64748b",
    "border":         "#334155",
    "accent":         "#6366f1",
    "accent_hover":   "#818cf8",
    "accent_subtle":  "rgba(99,102,241,0.12)",
    "gradient_start": "#1e1b4b",
    "gradient_mid":   "#312e81",
    "gradient_end":   "#4338ca",
    "success":        "#2ecc71",
    "warning":        "#f39c12",
    "danger":         "#e74c3c",
    "shadow":         "rgba(0,0,0,0.4)",
    "glass_bg":       "rgba(30,41,59,0.6)",
    "glass_border":   "rgba(99,102,241,0.2)",
    "chart_bg":       "rgba(0,0,0,0)",
    "chart_grid":     "rgba(148,163,184,0.1)",
    "chart_text":     "#94a3b8",
}

_LIGHT = {
    "bg_primary":     "#f8fafc",
    "bg_secondary":   "#f1f5f9",
    "bg_card":        "#ffffff",
    "bg_card_hover":  "#f0f0ff",
    "text_primary":   "#1e293b",
    "text_secondary": "#475569",
    "text_muted":     "#94a3b8",
    "border":         "#e2e8f0",
    "accent":         "#4f46e5",
    "accent_hover":   "#6366f1",
    "accent_subtle":  "rgba(79,70,229,0.08)",
    "gradient_start": "#e0e7ff",
    "gradient_mid":   "#c7d2fe",
    "gradient_end":   "#a5b4fc",
    "success":        "#16a34a",
    "warning":        "#d97706",
    "danger":         "#dc2626",
    "shadow":         "rgba(0,0,0,0.08)",
    "glass_bg":       "rgba(255,255,255,0.7)",
    "glass_border":   "rgba(79,70,229,0.15)",
    "chart_bg":       "rgba(0,0,0,0)",
    "chart_grid":     "rgba(71,85,105,0.1)",
    "chart_text":     "#475569",
}


def _palette() -> dict:
    return _DARK if is_dark() else _LIGHT


def get_chart_colors() -> dict:
    """Return chart-specific theme colours for Plotly."""
    p = _palette()
    return {
        "bg": p["chart_bg"],
        "grid": p["chart_grid"],
        "text": p["chart_text"],
        "paper_bg": p["chart_bg"],
        "font_color": p["text_primary"],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# INJECT  CSS
# ═══════════════════════════════════════════════════════════════════════════════

def inject_global_css():
    """Call once per page to inject theme-aware CSS."""
    p = _palette()
    mode = st.session_state.get("theme_mode", "dark")

    css = f"""
    <style>
    /* ── Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* ── Root variables ── */
    :root {{
        --bg-primary: {p["bg_primary"]};
        --bg-secondary: {p["bg_secondary"]};
        --bg-card: {p["bg_card"]};
        --bg-card-hover: {p["bg_card_hover"]};
        --text-primary: {p["text_primary"]};
        --text-secondary: {p["text_secondary"]};
        --text-muted: {p["text_muted"]};
        --border: {p["border"]};
        --accent: {p["accent"]};
        --accent-hover: {p["accent_hover"]};
        --accent-subtle: {p["accent_subtle"]};
        --gradient-start: {p["gradient_start"]};
        --gradient-mid: {p["gradient_mid"]};
        --gradient-end: {p["gradient_end"]};
        --success: {p["success"]};
        --warning: {p["warning"]};
        --danger: {p["danger"]};
        --shadow: {p["shadow"]};
        --glass-bg: {p["glass_bg"]};
        --glass-border: {p["glass_border"]};
    }}

    /* ── Global body ── */
    .stApp, .stApp > header {{
        background-color: var(--bg-primary) !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif !important;
    }}

    .stApp [data-testid="stSidebar"] {{
        background: {"linear-gradient(180deg, #0b1120 0%, #131c33 100%)" if mode == "dark"
                       else "linear-gradient(180deg, #f1f5f9 0%, #e8ecf4 100%)"} !important;
        border-right: 1px solid var(--border) !important;
    }}

    /* ── Metric cards ── */
    [data-testid="stMetric"] {{
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 2px 8px var(--shadow);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}
    [data-testid="stMetric"]:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 20px var(--shadow);
    }}
    [data-testid="stMetric"] label {{
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
    }}
    [data-testid="stMetric"] [data-testid="stMetricValue"] {{
        color: var(--text-primary) !important;
        font-weight: 700 !important;
    }}

    /* ── Buttons ── */
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="stBaseButton-primary"] {{
        background: linear-gradient(135deg, var(--accent) 0%, var(--accent-hover) 100%) !important;
        border: none !important;
        color: #fff !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.5rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 8px rgba(99,102,241,0.3) !important;
    }}
    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="stBaseButton-primary"]:hover {{
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 16px rgba(99,102,241,0.4) !important;
    }}
    .stButton > button[kind="secondary"],
    .stButton > button[data-testid="stBaseButton-secondary"] {{
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        color: var(--text-primary) !important;
        font-weight: 500 !important;
        border-radius: 10px !important;
        transition: all 0.2s ease !important;
    }}
    .stButton > button[kind="secondary"]:hover,
    .stButton > button[data-testid="stBaseButton-secondary"]:hover {{
        background: var(--bg-card-hover) !important;
        border-color: var(--accent) !important;
    }}

    /* ── Expanders ── */
    .streamlit-expanderHeader {{
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        color: var(--text-primary) !important;
        font-weight: 600 !important;
    }}
    .streamlit-expanderContent {{
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-top: none !important;
        border-radius: 0 0 10px 10px !important;
    }}

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background: var(--bg-secondary) !important;
        border-radius: 12px;
        padding: 4px;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px !important;
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
    }}
    .stTabs [aria-selected="true"] {{
        background: var(--accent) !important;
        color: #fff !important;
    }}

    /* ── File uploader ── */
    [data-testid="stFileUploader"] {{
        border: 2px dashed var(--border) !important;
        border-radius: 12px !important;
        background: var(--bg-card) !important;
        transition: border-color 0.2s ease !important;
    }}
    [data-testid="stFileUploader"]:hover {{
        border-color: var(--accent) !important;
    }}

    /* ── Dataframes ── */
    .stDataFrame {{
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        overflow: hidden !important;
    }}

    /* ── Dividers ── */
    hr {{
        border-color: var(--border) !important;
        opacity: 0.5;
    }}

    /* ── Form inputs ── */
    .stTextInput > div > div > input {{
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        color: var(--text-primary) !important;
    }}
    .stTextInput > div > div > input:focus {{
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 2px var(--accent-subtle) !important;
    }}

    /* ── Success / Warning / Error alerts ── */
    .stAlert {{
        border-radius: 10px !important;
        border-left: 4px solid !important;
    }}

    /* ── Progress bar ── */
    .stProgress > div > div > div {{
        background: linear-gradient(90deg, var(--accent), var(--accent-hover)) !important;
        border-radius: 4px !important;
    }}

    /* ── Scrollbar ── */
    ::-webkit-scrollbar {{
        width: 6px;
        height: 6px;
    }}
    ::-webkit-scrollbar-track {{
        background: var(--bg-secondary);
    }}
    ::-webkit-scrollbar-thumb {{
        background: var(--text-muted);
        border-radius: 3px;
    }}
    ::-webkit-scrollbar-thumb:hover {{
        background: var(--accent);
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# REUSABLE  HTML HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def hero_section(title: str, subtitle: str, emoji: str = "🎓"):
    """Gradient hero banner at top of page."""
    p = _palette()
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {p['gradient_start']} 0%, {p['gradient_mid']} 50%, {p['gradient_end']} 100%);
        border-radius: 16px;
        padding: 2.5rem 2rem;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 20px {p['shadow']};
        border: 1px solid {p['glass_border']};
    ">
        <div style="font-size: 2.8rem; margin-bottom: 0.3rem;">{emoji}</div>
        <h1 style="
            color: #f1f5f9;
            font-size: 2rem;
            font-weight: 700;
            margin: 0 0 0.4rem 0;
            font-family: 'Inter', sans-serif;
        ">{title}</h1>
        <p style="
            color: #cbd5e1;
            font-size: 1.05rem;
            margin: 0;
            font-family: 'Inter', sans-serif;
        ">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def card(content_html: str, extra_style: str = ""):
    """Glass-morphism card container."""
    p = _palette()
    st.markdown(f"""
    <div style="
        background: {p['bg_card']};
        border: 1px solid {p['border']};
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 10px {p['shadow']};
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        {extra_style}
    ">
        {content_html}
    </div>
    """, unsafe_allow_html=True)


def feature_card(emoji: str, title: str, description: str, accent: str = ""):
    """Navigation feature card for the home page."""
    p = _palette()
    accent_color = accent or p["accent"]
    st.markdown(f"""
    <div style="
        background: {p['bg_card']};
        border: 1px solid {p['border']};
        border-left: 4px solid {accent_color};
        border-radius: 12px;
        padding: 1.4rem;
        margin-bottom: 0.5rem;
        box-shadow: 0 2px 8px {p['shadow']};
        transition: all 0.2s ease;
    ">
        <div style="font-size: 1.6rem; margin-bottom: 0.4rem;">{emoji}</div>
        <div style="
            color: {p['text_primary']};
            font-size: 1.05rem;
            font-weight: 600;
            margin-bottom: 0.3rem;
            font-family: 'Inter', sans-serif;
        ">{title}</div>
        <div style="
            color: {p['text_secondary']};
            font-size: 0.9rem;
            font-family: 'Inter', sans-serif;
        ">{description}</div>
    </div>
    """, unsafe_allow_html=True)


def status_badge(status: str) -> str:
    """Return an inline HTML badge for analysis status."""
    p = _palette()
    color_map = {
        "completed": p["success"],
        "processing": p["warning"],
        "failed": p["danger"],
        "uploading": p["accent"],
    }
    emoji_map = {
        "completed": "✅",
        "processing": "⏳",
        "failed": "❌",
        "uploading": "📤",
    }
    color = color_map.get(status, p["text_muted"])
    emoji = emoji_map.get(status, "❔")
    return (
        f'<span style="'
        f"display:inline-block; padding:3px 10px; border-radius:6px; "
        f"font-size:0.82rem; font-weight:600; "
        f"background:{color}22; color:{color}; "
        f"border:1px solid {color}44;"
        f'">{emoji} {status.capitalize()}</span>'
    )


def section_header(title: str, emoji: str = ""):
    """Styled section header."""
    p = _palette()
    st.markdown(f"""
    <div style="
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 1.5rem 0 1rem 0;
    ">
        <span style="font-size: 1.4rem;">{emoji}</span>
        <h3 style="
            color: {p['text_primary']};
            font-weight: 700;
            margin: 0;
            font-family: 'Inter', sans-serif;
        ">{title}</h3>
        <div style="
            flex: 1;
            height: 1px;
            background: linear-gradient(90deg, {p['border']}, transparent);
            margin-left: 12px;
        "></div>
    </div>
    """, unsafe_allow_html=True)
