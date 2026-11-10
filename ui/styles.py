"""Small, centralized style adjustments for the Streamlit interface."""

import streamlit as st


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        /* Theme-neutral: translucent greys read correctly on both light and
           dark Streamlit themes, so text keeps the theme's own color. */
        :root {
            --accent: #c65037;
            --line: rgba(128, 128, 128, 0.25);
            --surface: rgba(128, 128, 128, 0.08);
        }
        [data-testid="stSidebar"] {
            background: var(--surface);
            border-right: 1px solid var(--line);
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 {
            font-size: 1.05rem;
        }
        .stButton > button {
            border-radius: 5px;
            min-height: 2.7rem;
            font-weight: 600;
        }
        .stButton > button[kind="primary"] {
            background: var(--accent);
            border-color: var(--accent);
        }
        [data-baseweb="tab-list"] {
            gap: 1.25rem;
            border-bottom: 1px solid var(--line);
        }
        [data-baseweb="tab"] {
            height: 3rem;
        }
        [data-baseweb="tab"][aria-selected="true"] {
            color: var(--accent);
        }
        /* History: compact, left-aligned entries */
        [data-testid="stSidebar"] .stButton > button {
            justify-content: flex-start;
            text-align: left;
            min-height: 0;
            padding: 0.45rem 0.7rem;
            font-weight: 500;
        }
        [data-testid="stSidebar"] .stButton > button p {
            font-size: 0.85rem;
            line-height: 1.35;
        }
        [data-testid="stMetric"] {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 5px;
            padding: 0.8rem 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )