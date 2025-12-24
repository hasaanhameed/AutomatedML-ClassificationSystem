from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    LabelEncoder,
    OrdinalEncoder,
    PolynomialFeatures,
)

from .eda import missing_value_analysis, outlier_detection_iqr, class_balance_info, additional_checks


def detect_issues(
    df: pd.DataFrame, target_col: str, eda_results: Optional[Dict[str, Any]] = None
):
    """
    Derive structured issues list from EDA or compute directly.

    Returns
    -------
    list of dict with keys:
        - key, message, severity, options, question
    """
    issues = []

    # Missing values
    mv_summary = (
        eda_results["missing_summary"]
        if eda_results and "missing_summary" in eda_results
        else missing_value_analysis(df)
    )
    for _, row in mv_summary.iterrows():
        col = row["Feature"]
        pct = row["Missing %"]
        if pct <= 0:
            continue
        severity = "critical" if pct > 20 else "warning"
        message = f"⚠️ {col} has {pct:.2f}% missing values"
        options = [
            "Mean Imputation",
            "Median Imputation",
            "Mode Imputation",
            "Constant Value",
            "Drop Column",
            "Keep As Is",
        ]
        issues.append(
            {
                "key": f"missing_{col}",
                "message": message,
                "severity": severity,
                "options": options,
                "question": f"How to handle missing values in {col}?",
            }
        )

    # Outliers (IQR based)
    out_summary = (
        eda_results["outliers_iqr"]
        if eda_results and "outliers_iqr" in eda_results
        else outlier_detection_iqr(df)
    )
    for _, row in out_summary.iterrows():
        col = row["Feature"]
        count = row["Outlier Count"]
        pct = row["Outlier %"]
        if count <= 0:
            continue
        severity = "warning" if pct > 5 else "info"
        message = f"⚠️ {col} has {count} outliers ({pct:.2f}% of values)"
        options = [
            "Remove Outliers",
            "Cap at IQR bounds",
            "Log Transform",
            "Keep As Is",
        ]
        issues.append(
            {
                "key": f"outlier_{col}",
                "message": message,
                "severity": severity,
                "options": options,
                "question": f"How to handle outliers in {col}?",
            }
        )

    # Class imbalance
    cb_info = (
        eda_results["class_balance"]
        if eda_results and "class_balance" in eda_results
        else class_balance_info(df, target_col)
    )
    if cb_info["imbalance_ratio"] is not None:
        ratio = cb_info["imbalance_ratio"]
        if ratio > 3:
            severity = "critical" if ratio > 10 else "warning"
            message = f"🔴 Severe class imbalance detected: {ratio:.2f}:1"
            options = [
                "SMOTE Oversampling",
                "Random Undersampling",
                "Class Weights",
                "No Action",
            ]
            issues.append(
                {
                    "key": "imbalance",
                    "message": message,
                    "severity": severity,
                    "options": options,
                    "question": "How to handle class imbalance?",
                }
            )

    # Additional checks
    checks = (
        eda_results["additional_checks"]
        if eda_results and "additional_checks" in eda_results
        else additional_checks(df)
    )
    if checks["constant_cols"]:
        issues.append(
            {
                "key": "constant_cols",
                "message": f"ℹ️ Constant columns detected: {checks['constant_cols']}",
                "severity": "info",
                "options": ["Drop", "Keep"],
                "question": "How to handle constant columns?",
            }
        )
    if checks["near_constant_cols"]:
        issues.append(
            {
                "key": "near_constant_cols",
                "message": f"ℹ️ Near-constant columns detected: {checks['near_constant_cols']}",
                "severity": "info",
                "options": ["Drop", "Keep"],
                "question": "How to handle near-constant columns?",
            }
        )
    if checks["duplicate_rows"] > 0:
        issues.append(
            {
                "key": "duplicate_rows",
                "message": f"ℹ️ {checks['duplicate_rows']} duplicate rows detected.",
                "severity": "info",
                "options": ["Drop Duplicates", "Keep"],
                "question": "How to handle duplicate rows?",
            }
        )

    return issues


def _apply_missing_value_strategy(
    df: pd.DataFrame, strategies: Dict[str, str]
) -> pd.DataFrame:
    """Apply per-column missing value strategies."""
    df = df.copy()
    for col, strat in strategies.items():
        if not col.startswith("missing_"):
            continue
        col_name = col.replace("missing_", "", 1)
        if col_name not in df.columns:
            continue
        choice = strat
        series = df[col_name]
        if "Mean" in choice and series.dtype.kind in "iufc":
            df[col_name] = series.fillna(series.mean())
        elif "Median" in choice and series.dtype.kind in "iufc":
            df[col_name] = series.fillna(series.median())
        elif "Mode" in choice:
            df[col_name] = series.fillna(series.mode().iloc[0] if not series.mode().empty else series)
        elif "Constant" in choice:
            if series.dtype.kind in "iufc":
                df[col_name] = series.fillna(0)
            else:
                df[col_name] = series.fillna("missing")
        elif "Drop Column" in choice:
            df = df.drop(columns=[col_name])
        elif "Keep As Is" in choice:
            continue
    return df


def _apply_outlier_strategy(
    df: pd.DataFrame, strategies: Dict[str, str]
) -> pd.DataFrame:
    """Apply simple outlier handling (IQR based) for numeric columns."""
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for key, choice in strategies.items():
        if not key.startswith("outlier_"):
            continue
        col = key.replace("outlier_", "", 1)
        if col not in numeric_cols:
            continue
        series = df[col]
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        if "Remove Outliers" in choice:
            mask = (series >= lower) & (series <= upper)
            df = df.loc[mask]
        elif "Cap at IQR bounds" in choice:
            df[col] = series.clip(lower=lower, upper=upper)
        elif "Log Transform" in choice:
            df[col] = np.log1p(series.clip(lower=0))
        elif "Keep As Is" in choice:
            continue
    return df


def _apply_duplicates_strategy(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    if strategy == "Drop Duplicates":
        return df.drop_duplicates()
    return df


def _scale_features(
    X: pd.DataFrame, scaler_choice: str
) -> Tuple[pd.DataFrame, Any]:
    if scaler_choice == "None":
        return X, None
    scaler_cls = {
        "StandardScaler": StandardScaler,
        "MinMaxScaler": MinMaxScaler,
        "RobustScaler": RobustScaler,
    }.get(scaler_choice, StandardScaler)
    scaler = scaler_cls()
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled, columns=X.columns, index=X.index), scaler


def _encode_categoricals(
    df: pd.DataFrame, encoding_strategies: Dict[str, str]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = df.copy()
    encoders: Dict[str, Any] = {}

    for col, enc in encoding_strategies.items():
        if col not in df.columns:
            continue
        if "One-Hot" in enc:
            dummies = pd.get_dummies(df[col], prefix=col, dummy_na=False)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
        elif "Label" in enc:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        elif "Ordinal" in enc:
            oe = OrdinalEncoder()
            df[col] = oe.fit_transform(df[[col]]).astype(int)
            encoders[col] = oe
    return df, encoders


def apply_preprocessing_from_strategies(
    df: pd.DataFrame,
    target_col: str,
    issue_strategies: Dict[str, str],
    scaler_choice: str,
    encoding_strategies: Dict[str, str],
    test_size: float,
    random_state: int,
    use_polynomial: bool,
) -> Tuple[pd.DataFrame, Tuple, Dict[str, Any]]:
    """
    Apply all preprocessing steps according to user strategies.

    Returns
    -------
    processed_df : pd.DataFrame
    train_test_data : tuple (X_train, X_test, y_train, y_test)
    preprocess_summary : dict
    """
    df_proc = df.copy()
    original_shape = df_proc.shape

    # Duplicates
    if "duplicate_rows" in issue_strategies:
        df_proc = _apply_duplicates_strategy(
            df_proc, issue_strategies["duplicate_rows"]
        )

    # Missing values
    df_proc = _apply_missing_value_strategy(df_proc, issue_strategies)

    # Outliers
    df_proc = _apply_outlier_strategy(df_proc, issue_strategies)

    # Constant / near constant
    if "constant_cols" in issue_strategies and issue_strategies["constant_cols"] == "Drop":
        from .eda import additional_checks

        checks = additional_checks(df_proc)
        df_proc = df_proc.drop(columns=checks["constant_cols"], errors="ignore")
    if (
        "near_constant_cols" in issue_strategies
        and issue_strategies["near_constant_cols"] == "Drop"
    ):
        from .eda import additional_checks

        checks = additional_checks(df_proc)
        df_proc = df_proc.drop(columns=checks["near_constant_cols"], errors="ignore")

    # Separate target
    if target_col not in df_proc.columns:
        raise ValueError("Target column not found after preprocessing.")
    y = df_proc[target_col]
    X = df_proc.drop(columns=[target_col])

    # Encode categoricals
    X, encoders = _encode_categoricals(X, encoding_strategies)

    # Polynomial features
    poly_features_added = 0
    if use_polynomial:
        num_cols = X.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            poly = PolynomialFeatures(degree=2, include_bias=False)
            X_poly = poly.fit_transform(X[num_cols])
            poly_cols = poly.get_feature_names_out(num_cols)
            X_poly_df = pd.DataFrame(X_poly, columns=poly_cols, index=X.index)
            # Replace numeric subset with polynomial expansion
            X = pd.concat([X.drop(columns=num_cols), X_poly_df], axis=1)
            poly_features_added = X_poly_df.shape[1] - len(num_cols)

    # Scale
    X_scaled, scaler = _scale_features(X, scaler_choice)

    # Train-test split
    # Use stratification only when every class has at least 2 samples
    class_counts = y.value_counts()
    use_stratify = len(class_counts) >= 2 and class_counts.min() >= 2
    stratify_y = y if use_stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_y,
    )

    processed_df = pd.concat([X_scaled, y.reset_index(drop=True)], axis=1)

    summary = {
        "original_shape": original_shape,
        "processed_shape": processed_df.shape,
        "scaler": scaler_choice,
        "polynomial_features_added": poly_features_added,
        "encoders": list(encoders.keys()),
        "strategies": issue_strategies,
        "stratified_split": use_stratify,
    }

    return processed_df, (X_train, X_test, y_train, y_test), summary



