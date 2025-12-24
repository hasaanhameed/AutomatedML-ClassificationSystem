from typing import Dict, Any, Tuple

import io
import pandas as pd


def generate_html_report(sections: Dict[str, Any]) -> str:
    """Generate a simple HTML report string from structured sections."""
    html_parts = [
        "<html><head><meta charset='utf-8'><title>AutoML Report</title>",
        "<style>body{font-family:Arial, sans-serif;padding:20px;}h1,h2{color:#333;}table{border-collapse:collapse;width:100%;margin-bottom:20px;}th,td{border:1px solid #ddd;padding:8px;font-size:12px;}th{background:#f4f4f4;}</style>",
        "</head><body>",
        "<h1>AutoML Classification Report</h1>",
    ]

    for title, content in sections.items():
        html_parts.append(f"<h2>{title}</h2>")
        if isinstance(content, dict):
            html_parts.append("<ul>")
            for k, v in content.items():
                if isinstance(v, (dict, list, tuple, pd.DataFrame)):
                    html_parts.append(f"<li><b>{k}:</b></li>")
                    if isinstance(v, pd.DataFrame):
                        html_parts.append(v.to_html(index=False))
                    else:
                        html_parts.append(f"<pre>{str(v)}</pre>")
                else:
                    html_parts.append(f"<li><b>{k}:</b> {v}</li>")
            html_parts.append("</ul>")
        elif isinstance(content, pd.DataFrame):
            html_parts.append(content.to_html(index=False))
        else:
            html_parts.append(f"<p>{content}</p>")

    html_parts.append("</body></html>")
    return "\n".join(html_parts)


def convert_html_to_pdf(html: str) -> bytes:
    """
    Convert HTML string to PDF using pdfkit if available.

    Returns
    -------
    pdf_bytes : bytes or None if conversion failed.
    """
    try:
        import pdfkit  # type: ignore

        pdf_bytes = pdfkit.from_string(html, False)
        return pdf_bytes
    except Exception:
        # Gracefully degrade to no-PDF mode
        return b""


def generate_report(
    dataset_info: Dict[str, Any],
    eda_results: Dict[str, Any],
    preprocessing_steps: Dict[str, Any],
    model_results: Dict[str, Any],
    best_model: str,
) -> Tuple[bytes, str]:
    """
    Generate comprehensive AutoML report.
    """
    comparison_df = model_results.get("comparison_df", pd.DataFrame())
    best_model_reasoning = model_results.get("best_model_reasoning", "")
    per_model_details = model_results.get("per_model_details", {})

    best_model_details = per_model_details.get(best_model, {})

    sections = {
        "1. Dataset Overview": {
            "rows": dataset_info.get("rows"),
            "columns": dataset_info.get("columns"),
            "features": dataset_info.get("features"),
            "target": dataset_info.get("target"),
            "class_distribution": dataset_info.get("class_dist"),
        },
        "2. EDA Findings": {
            "missing_values": getattr(
                eda_results.get("missing_summary", pd.DataFrame()), "to_dict", lambda: {}
            )(orient="records")
            if isinstance(eda_results.get("missing_summary"), pd.DataFrame)
            else eda_results.get("missing_summary"),
            "outliers_iqr": getattr(
                eda_results.get("outliers_iqr", pd.DataFrame()), "to_dict", lambda: {}
            )(orient="records")
            if isinstance(eda_results.get("outliers_iqr"), pd.DataFrame)
            else eda_results.get("outliers_iqr"),
            "high_correlations": getattr(
                eda_results.get("high_correlations", pd.DataFrame()),
                "to_dict",
                lambda: {},
            )(orient="records")
            if isinstance(eda_results.get("high_correlations"), pd.DataFrame)
            else eda_results.get("high_correlations"),
            "data_issues": eda_results.get("flagged_issues"),
        },
        "3. Preprocessing Applied": {
            "missing_value_strategy": preprocessing_steps.get("missing"),
            "outlier_handling": preprocessing_steps.get("outliers"),
            "scaling_method": preprocessing_steps.get("scaling"),
            "encoding_method": preprocessing_steps.get("encoding"),
            "feature_engineering": preprocessing_steps.get("feature_eng"),
        },
        "4. Model Training": {
            "models_trained": list(per_model_details.keys()),
            "hyperparameter_method": "Grid/Random Search",
            "cross_validation": "5-fold CV",
        },
        "5. Model Comparison": {
            "comparison_table": comparison_df,
            "best_model": best_model,
            "justification": best_model_reasoning,
        },
        "6. Best Model Details": {
            "name": best_model,
            "hyperparameters": best_model_details.get("best_params"),
            "performance_metrics": best_model_details.get("metrics"),
            "confusion_matrix": "See confusion matrix section in app.",
            "feature_importance": best_model_details.get("feature_importance"),
        },
        "7. Recommendations": {
            "deployment_notes": "Model ready for deployment (after further validation).",
            "monitoring_suggestions": "Track prediction drift and data quality over time.",
            "improvement_ideas": "Consider advanced ensembles, feature selection, and interpretability tools (SHAP/LIME).",
        },
    }

    html_report = generate_html_report(sections)
    pdf_report = convert_html_to_pdf(html_report)

    return pdf_report, html_report




