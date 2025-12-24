import pandas as pd
import streamlit as st

from utils.models import evaluate_all_models
from utils.visualization import render_model_evaluation_tabs


def page_results() -> None:
    if not st.session_state.model_results:
        st.warning(
            "Please train the models first on the 'Model Training' page."
        )
        return
    if st.session_state.train_test_data is None:
        st.warning(
            "Train-test data not found. Please re-run preprocessing and training."
        )
        return

    st.subheader("Results & Model Comparison")

    X_train, X_test, y_train, y_test = st.session_state.train_test_data

    if not st.session_state.evaluation_results:
        with st.spinner("Evaluating models..."):
            eval_results = evaluate_all_models(
                st.session_state.model_results, X_test, y_test
            )
        st.session_state.evaluation_results = eval_results

    comparison_df: pd.DataFrame = st.session_state.evaluation_results["comparison_df"]
    st.markdown("### Model Comparison Table")
    st.dataframe(
        comparison_df.style.highlight_max(
            subset=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
            color="lightgreen",
        ).highlight_min(subset=["Training Time (s)"], color="lightblue")
    )

    sort_metric = st.selectbox(
        "Sort models by:",
        [
            "Accuracy",
            "Precision",
            "Recall",
            "F1-Score",
            "ROC-AUC",
            "Training Time (s)",
        ],
    )
    ascending = st.checkbox("Ascending order", value=False)
    sorted_df = comparison_df.sort_values(by=sort_metric, ascending=ascending)
    st.markdown("### Sorted Models")
    st.dataframe(sorted_df)

    # Visual comparisons
    render_model_evaluation_tabs(
        st.session_state.evaluation_results,
        st.session_state.model_results,
        X_test,
        y_test,
    )

    # Best model selection
    st.markdown("---")
    st.markdown("### Best Model Selection")
    best_model_name = st.selectbox(
        "Select your preferred model:",
        comparison_df["Model"].tolist(),
    )
    st.session_state.best_model_name = best_model_name
    best_model = st.session_state.model_results[best_model_name]["model"]
    st.success(f"✅ Selected: {best_model_name}")
    st.json(st.session_state.model_results[best_model_name]["best_params"])

    # Download comparison
    csv = comparison_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Comparison Table (CSV)",
        csv,
        "model_comparison.csv",
        "text/csv",
    )


