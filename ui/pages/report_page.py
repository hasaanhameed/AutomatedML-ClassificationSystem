import streamlit as st

from utils.report_generator import generate_report


def page_report() -> None:
    if (
        st.session_state.df is None
        or not st.session_state.evaluation_results
        or st.session_state.best_model_name is None
    ):
        st.warning(
            "Please complete previous steps (Upload, EDA, Preprocessing, Training, Results) first."
        )
        return

    st.subheader("Auto-Generated Report")

    df = st.session_state.df
    target_col = st.session_state.target_col
    eda_results = st.session_state.eda_results
    preprocessing_steps = {
        "missing": st.session_state.issue_strategies,
        "outliers": st.session_state.issue_strategies,
        "scaling": "N/A",
        "encoding": "N/A",
        "feature_eng": "N/A",
    }
    model_results = st.session_state.evaluation_results
    best_model_name = st.session_state.best_model_name

    dataset_info = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "features": [c for c in df.columns if c != target_col],
        "target": target_col,
        "class_dist": df[target_col].value_counts().to_dict()
        if target_col in df.columns
        else {},
    }

    if st.button("Generate Report"):
        with st.spinner("Generating report..."):
            pdf_report, html_report = generate_report(
                dataset_info=dataset_info,
                eda_results=eda_results or {},
                preprocessing_steps=preprocessing_steps,
                model_results=model_results,
                best_model=best_model_name,
            )

        if html_report:
            st.success("HTML report generated.")
            st.download_button(
                "Download Report (HTML)",
                html_report,
                "automl_report.html",
                "text/html",
            )


