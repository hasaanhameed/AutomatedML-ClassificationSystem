from typing import Dict, Any

import pandas as pd
import streamlit as st

from utils.eda import compute_basic_metadata


def page_upload() -> None:
    st.subheader("Upload Dataset")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        with st.spinner("Loading dataset and computing dataset statistics..."):
            try:
                df = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                st.stop()

            st.session_state.df = df

            # Target selection
            target_col = st.selectbox(
                "Select target column (label)",
                options=df.columns,
                index=len(df.columns) - 1 if len(df.columns) > 0 else 0,
            )
            st.session_state.target_col = target_col

            # Basic metadata
            meta: Dict[str, Any] = compute_basic_metadata(df)
            st.markdown("### Dataset Overview")
            col1, col2, col3 = st.columns(3)
            col1.metric("Rows", meta["rows"])
            col2.metric("Columns", meta["cols"])
            col3.metric("Missing Cells", meta["missing_cells"])

            with st.expander("Column Info"):
                st.dataframe(meta["dtypes"])

            with st.expander("Summary Statistics"):
                st.dataframe(df.describe(include="all").transpose())

            with st.expander("First 5 Rows"):
                st.dataframe(df.head())

            with st.expander("Last 5 Rows"):
                st.dataframe(df.tail())

        st.success("Dataset uploaded and summarized successfully.")

        st.session_state.current_step = max(st.session_state.current_step, 1)


