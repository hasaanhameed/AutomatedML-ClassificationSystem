from typing import Dict, Any

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st


# Global plotting style for dark, purple-themed visuals
PURPLE_PALETTE = ["#e9d5ff", "#c4b5fd", "#a855f7", "#7c3aed", "#5b21b6"]
PURPLE_CMAP = "Purples"

sns.set_theme(style="darkgrid", palette=PURPLE_PALETTE)
plt.style.use("dark_background")


def render_eda_tabs(
    df: pd.DataFrame,
    target_col: str,
    eda_results: Dict[str, Any],
) -> None:
    """Render EDA results in Streamlit tabs."""
    missing_tab, outlier_tab, corr_tab, dist_tab, cat_tab = st.tabs(
        ["Missing Values", "Outliers", "Correlations", "Distributions", "Categorical"]
    )

    with missing_tab:
        st.markdown("#### Missing Value Analysis")
        st.dataframe(eda_results["missing_summary"])

    with outlier_tab:
        st.markdown("#### Outlier Detection (IQR & Z-score)")
        st.write("**IQR-based outliers:**")
        st.dataframe(eda_results["outliers_iqr"])
        st.write("**Z-score-based outliers:**")
        st.dataframe(eda_results["outliers_z"])

    with corr_tab:
        st.markdown("#### Correlation Analysis")
        corr_matrix = eda_results["corr_matrix"]
        if corr_matrix is not None and not corr_matrix.empty:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            sns.heatmap(
                corr_matrix,
                annot=True,
                fmt=".2f",
                cmap=PURPLE_CMAP,
                ax=ax,
            )
            ax.set_title("Correlation Heatmap")
            st.pyplot(fig)
        st.markdown("**Highly correlated feature pairs (|corr| >= 0.8):**")
        st.dataframe(eda_results["high_correlations"])

    with dist_tab:
        st.markdown("#### Distributions of Numerical Features")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        stats = eda_results["dist_summary"]
        st.dataframe(stats)

        for col in numeric_cols:
            fig, ax = plt.subplots(figsize=(4, 2.5))
            sns.histplot(df[col].dropna(), kde=True, ax=ax, color=PURPLE_PALETTE[2])
            ax.set_title(f"Distribution - {col}")
            ax.axvline(df[col].mean(), color="#f97316", linestyle="--", label="Mean")
            ax.axvline(df[col].median(), color="#22c55e", linestyle=":", label="Median")
            ax.legend()
            st.pyplot(fig)

    with cat_tab:
        st.markdown("#### Categorical Feature Analysis")
        st.dataframe(eda_results["cat_cardinality"])

        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        for col in cat_cols:
            vc = df[col].value_counts()
            st.markdown(f"**{col}** (unique: {vc.shape[0]})")
            fig, ax = plt.subplots(figsize=(4, 2.5))
            sns.barplot(
                x=vc.index.astype(str),
                y=vc.values,
                ax=ax,
                palette=PURPLE_PALETTE,
            )
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            ax.set_ylabel("Count")
            st.pyplot(fig)


def render_model_evaluation_tabs(
    eval_results: Dict[str, Any],
    model_results: Dict[str, Any],
    X_test,
    y_test,
) -> None:
    """Render model evaluation plots in tabs."""
    overview_tab, cm_tab, roc_tab = st.tabs(
        ["Overview", "Confusion Matrices", "ROC Curves"]
    )

    comparison_df = eval_results["comparison_df"]
    per_model = eval_results["per_model_details"]

    with overview_tab:
        st.markdown("#### Metrics Overview")
        st.dataframe(comparison_df)

    with cm_tab:
        st.markdown("#### Confusion Matrices")
        for name, details in per_model.items():
            cm = details["confusion_matrix"]
            fig, ax = plt.subplots(figsize=(3.6, 2.6))
            sns.heatmap(cm, annot=True, fmt="d", cmap=PURPLE_CMAP, ax=ax)
            ax.set_title(f"Confusion Matrix - {name}")
            ax.set_ylabel("True Label")
            ax.set_xlabel("Predicted Label")
            st.pyplot(fig)

    with roc_tab:
        st.markdown("#### ROC Curves (Binary Classification)")
        for name, details in per_model.items():
            roc_data = details["roc"]
            if roc_data is None:
                continue
            fpr = roc_data["fpr"]
            tpr = roc_data["tpr"]
            auc_val = roc_data["auc"]
            fig, ax = plt.subplots(figsize=(4.2, 2.6))
            ax.plot(
                fpr,
                tpr,
                label=f"{name} (AUC = {auc_val:.3f})",
                color=PURPLE_PALETTE[2],
            )
            ax.plot([0, 1], [0, 1], linestyle="--", color="#9ca3af", label="Random")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curve")
            ax.legend()
            st.pyplot(fig)




