"""
streamlit_app.py - FEM Analysis Tool navigation entry point.

Defines multipage navigation with custom sidebar labels via st.navigation.
"""

import streamlit as st

st.set_page_config(page_title="FEM解析ツール", layout="wide")

pg = st.navigation([
    st.Page("pages/1_slab_analysis.py", title="矩形平板解析", default=True),
    st.Page("pages/2_アーチ解析.py", title="アーチ解析"),
])
pg.run()
