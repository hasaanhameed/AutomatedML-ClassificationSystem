from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def compute_basic_metadata(df: pd.DataFrame) -> Dict[str, Any]:
    """Return simple metadata for uploaded dataframe."""
    rows, cols = df.shape
    missing_cells = int(df.isna().sum().sum())
    dtypes = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "non_null": df.notnull().sum(),
            "missing": df.isnull().sum(),
        }
    )
    return {
        "rows": rows,
        "cols": cols,
        "missing_cells": missing_cells,
        "dtypes": dtypes,
    }


def missing_value_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate missing values per column (count & percentage)."""
    missing_count = df.isnull().sum()
    missing_pct = missing_count / len(df) * 100
    summary = pd.DataFrame(
        {
            "Feature": df.columns,
            "Missing Count": missing_count.values,
            "Missing %": missing_pct.values,
        }
    )
    return summary


def outlier_detection_iqr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect outliers in numerical columns using IQR method.

    Returns
    -------
    pd.DataFrame
        Columns: Feature, Outlier Count, Outlier %
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    records = []
    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (series < lower) | (series > upper)
        count = int(mask.sum())
        pct = (count / len(series)) * 100
        records.append({"Feature": col, "Outlier Count": count, "Outlier %": pct})
    return pd.DataFrame(records)


def outlier_detection_zscore(df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    """
    Detect outliers in numerical columns using Z-score method.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    threshold : float
        Z-score threshold beyond which values are considered outliers.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    records = []
    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty or series.std() == 0:
            continue
        zscores = (series - series.mean()) / series.std()
        mask = zscores.abs() > threshold
        count = int(mask.sum())
        pct = (count / len(series)) * 100
        records.append({"Feature": col, "Outlier Count": count, "Outlier %": pct})
    return pd.DataFrame(records)


def correlation_analysis(
    df: pd.DataFrame, threshold: float = 0.8
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute correlation matrix and identify highly correlated feature pairs.

    Returns
    -------
    corr_matrix : pd.DataFrame
    high_corr_pairs : pd.DataFrame with columns [Feature 1, Feature 2, Correlation]
    """
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    pairs = []
    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr_matrix.iloc[i, j]
            if abs(val) >= threshold:
                pairs.append(
                    {"Feature 1": cols[i], "Feature 2": cols[j], "Correlation": val}
                )
    high_corr_pairs = pd.DataFrame(pairs)
    return corr_matrix, high_corr_pairs


def distribution_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return basic distribution stats for numerical columns."""
    numeric_df = df.select_dtypes(include=[np.number])
    stats = pd.DataFrame(
        {
            "Mean": numeric_df.mean(),
            "Median": numeric_df.median(),
            "Std": numeric_df.std(),
        }
    )
    stats.index.name = "Feature"
    stats.reset_index(inplace=True)
    return stats


def categorical_cardinality(df: pd.DataFrame) -> pd.DataFrame:
    """Compute cardinality info for categorical columns."""
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    records = []
    for col in cat_cols:
        vc = df[col].value_counts(dropna=False)
        records.append(
            {
                "Feature": col,
                "Unique Values": int(vc.shape[0]),
                "Most Frequent": vc.index[0] if not vc.empty else None,
                "Most Frequent Count": int(vc.iloc[0]) if not vc.empty else 0,
            }
        )
    return pd.DataFrame(records)


def class_balance_info(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
    """Return class distribution and imbalance ratio."""
    if target_col not in df.columns:
        return {"counts": {}, "imbalance_ratio": None, "flag": None}
    counts = df[target_col].value_counts()
    if counts.empty:
        return {"counts": {}, "imbalance_ratio": None, "flag": None}
    max_count = counts.max()
    min_count = counts.min()
    imbalance_ratio = float(max_count / max(min_count, 1))
    flag = "severe" if imbalance_ratio > 10 else "warning" if imbalance_ratio > 3 else "ok"
    return {
        "counts": counts.to_dict(),
        "imbalance_ratio": imbalance_ratio,
        "flag": flag,
    }


def additional_checks(df: pd.DataFrame) -> Dict[str, Any]:
    """Identify constant/near-constant, duplicates, high-cardinality categorical."""
    constant_cols: List[str] = []
    near_constant_cols: List[str] = []
    for col in df.columns:
        counts = df[col].value_counts(normalize=True, dropna=False)
        if len(counts) == 1:
            constant_cols.append(col)
        elif counts.iloc[0] >= 0.95:
            near_constant_cols.append(col)

    duplicate_rows = int(df.duplicated().sum())

    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    high_cardinality: List[str] = []
    for col in cat_cols:
        if df[col].nunique(dropna=False) > 50:
            high_cardinality.append(col)

    return {
        "constant_cols": constant_cols,
        "near_constant_cols": near_constant_cols,
        "duplicate_rows": duplicate_rows,
        "high_cardinality_cols": high_cardinality,
    }


def perform_full_eda(
    df: pd.DataFrame, target_col: str
) -> Dict[str, Any]:
    """Run all EDA analyses and return structured results."""
    missing_summary = missing_value_analysis(df)
    outliers_iqr = outlier_detection_iqr(df)
    outliers_z = outlier_detection_zscore(df)
    corr_matrix, high_corr_pairs = correlation_analysis(df)
    dist_summary = distribution_stats(df)
    cat_card = categorical_cardinality(df)
    class_balance = class_balance_info(df, target_col)
    checks = additional_checks(df)

    flagged_issues: List[str] = []
    # Simple high level issue summary
    if not missing_summary[missing_summary["Missing %"] > 20].empty:
        flagged_issues.append("Columns with >20% missing values.")
    if class_balance["flag"] == "severe":
        flagged_issues.append("Severe class imbalance detected.")
    if checks["high_cardinality_cols"]:
        flagged_issues.append("High cardinality categorical features.")

    return {
        "missing_summary": missing_summary,
        "outliers_iqr": outliers_iqr,
        "outliers_z": outliers_z,
        "corr_matrix": corr_matrix,
        "high_correlations": high_corr_pairs,
        "dist_summary": dist_summary,
        "cat_cardinality": cat_card,
        "class_balance": class_balance,
        "additional_checks": checks,
        "flagged_issues": flagged_issues,
    }




