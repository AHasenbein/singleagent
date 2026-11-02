"""Small, centralized style adjustments for the Streamlit interface."""

import streamlit as st


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #20302b;
            --muted: #64736d;
            --accent: #c65037;
            --line: #dce5df;
            --surface: #f3f7f3;
        }
        .stApp {
            color: var(--ink);
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