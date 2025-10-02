from __future__ import annotations

from pathlib import Path

import streamlit as st


def inject_brand_styles() -> None:
    st.markdown(
        """
        <link href="https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;600&display=swap" rel="stylesheet">
        <style>
        :root { --brand-accent: #B92D24; --brand-green-dark: #5F6E60; --brand-green-light: #A6C4A7; --brand-bg: #FFFCF2; --brand-text: #2C2626; }
        html, body, .stApp { font-family: Helvetica, Arial, sans-serif; color: var(--brand-text); }
        .stApp { background-color: var(--brand-bg) !important; }
        [data-testid="stSidebar"] { background-color: var(--brand-green-light) !important; }
        [data-testid="stSidebar"] * { color: var(--brand-text); }
        h1, h2, h3, h4, h5, h6 { font-family: 'Source Serif 4', Georgia, serif; color: var(--brand-green-dark); }
        a { color: var(--brand-accent); }
        .stButton>button { background-color: var(--brand-green-dark); color: #fff; border: 0; border-radius: 6px; }
        .stButton>button:hover { background-color: #4d5a50; }
        .stApp header { background: transparent; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_brand_header(logo_full: Path, logo_icon: Path) -> None:
    c1, c2 = st.columns([1, 5])
    with c1:
        if logo_full.exists():
            st.image(str(logo_full), use_container_width=True)
        elif logo_icon.exists():
            st.image(str(logo_icon), width=96)
    with c2:
        st.markdown(
            "<div style='padding-top:8px;'><h1 style='margin-bottom:0;'>Substack Ads ROI Simulator</h1></div>",
            unsafe_allow_html=True,
        )
    st.divider()


def format_currency(value: float) -> str:
    return f"${value:,.0f}"


def format_date_badges(dates) -> str:
    items: list[str] = []
    import pandas as pd

    for d in dates:
        try:
            dt = pd.to_datetime(d)
            label = dt.strftime("%b %d, %Y")
        except Exception:
            label = str(d)
        items.append(
            "<span style='display:inline-block;margin:2px 6px 2px 0;padding:2px 8px;border-radius:999px;border:1px solid #8e44ad;color:#8e44ad;font-size:12px;'>"
            f"{label}</span>"
        )
    return "".join(items)
