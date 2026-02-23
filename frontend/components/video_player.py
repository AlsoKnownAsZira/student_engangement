"""
Video display component.
"""

from __future__ import annotations
import streamlit as st


def show_video(url: str | None, caption: str = "Annotated Result Video"):
    """Display a video from a signed URL. Falls back to a message."""
    if url:
        st.video(url)
        st.caption(f"🎬 {caption}")
    else:
        st.info("📽️ Annotated video is not available yet.")
