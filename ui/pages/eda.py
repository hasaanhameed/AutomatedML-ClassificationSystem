import streamlit as st

from utils.eda import perform_full_eda
from utils.visualization import render_eda_tabs


def page_eda():
    if st.session_state.df is None:
        st.warning("Please upload a dataset first on the 'Upload Dataset' page.")
        return
    if st.session_state.target_col is None:
        st.warning("Please select a target column on the 'Upload Dataset' page.")
        return

    st.subheader("Automated Exploratory Data Analysis")

    if st.button("Run EDA"):
        with st.spinner("Running EDA..."):
            eda_results = perform_full_eda(
                st.session_state.df, st.session_state.target_col
            )
        st.session_state.eda_results = eda_results
        st.success("✅ EDA completed.")
        st.session_state.current_step = max(st.session_state.current_step, 2)

    if st.session_state.eda_results is not None:
        render_eda_tabs(
            st.session_state.df,
            st.session_state.target_col,
            st.session_state.eda_results,
        )
    else:
        st.info("Click **Run EDA** to generate automated analysis.")

