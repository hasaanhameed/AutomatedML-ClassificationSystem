from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    auc,
)


def get_model_definitions() -> Dict[str, Dict[str, Any]]:
    """Return dictionary of models and hyperparameters."""
    models = {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000),
            "params": {
                "C": [0.001, 0.01, 0.1, 1, 10, 100],
                "penalty": ["l1", "l2", "elasticnet"],
                "solver": ["liblinear", "saga"],
            },
        },
        "K-Nearest Neighbors": {
            "model": KNeighborsClassifier(),
            "params": {
                "n_neighbors": [3, 5, 7, 9, 11, 15],
                "weights": ["uniform", "distance"],
                "metric": ["euclidean", "manhattan", "minkowski"],
            },
        },
        "Decision Tree": {
            "model": DecisionTreeClassifier(),
            "params": {
                "max_depth": [3, 5, 7, 10, 15, None],
                "min_samples_split": [2, 5, 10, 20],
                "min_samples_leaf": [1, 2, 4, 8],
                "criterion": ["gini", "entropy"],
            },
        },
        "Naive Bayes": {
            "model": GaussianNB(),
            "params": {
                "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6],
            },
        },
        "Random Forest": {
            "model": RandomForestClassifier(),
            "params": {
                "n_estimators": [50, 100, 200, 300],
                "max_depth": [5, 10, 15, 20, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2", None],
            },
        },
        "Support Vector Machine": {
            "model": SVC(probability=True),
            "params": {
                "C": [0.1, 1, 10, 100],
                "kernel": ["rbf", "poly", "sigmoid"],
                "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
            },
        },
        "Rule-Based Classifier": {
            "model": DecisionTreeClassifier(max_depth=3),
            "params": {
                "max_depth": [1, 2, 3, 4, 5],
                "min_samples_split": [2, 5, 10],
            },
        },
    }
    return models


def train_all_models(
    X_train,
    y_train,
    search_method: str,
    progress_bar,
    status_text,
) -> Dict[str, Dict[str, Any]]:
    """Train all models with hyperparameter optimization."""
    import time

    models = get_model_definitions()
    results: Dict[str, Dict[str, Any]] = {}
    total = len(models)

    for idx, (name, config) in enumerate(models.items()):
        status_text.text(f"Training {name}...")
        start_time = time.time()

        if search_method == "GridSearchCV":
            search = GridSearchCV(
                config["model"],
                config["params"],
                cv=5,
                scoring="f1_weighted",
                n_jobs=-1,
            )
        else:
            search = RandomizedSearchCV(
                config["model"],
                config["params"],
                n_iter=20,
                cv=5,
                scoring="f1_weighted",
                n_jobs=-1,
                random_state=42,
            )

        search.fit(X_train, y_train)
        training_time = time.time() - start_time

        results[name] = {
            "model": search.best_estimator_,
            "best_params": search.best_params_,
            "training_time": training_time,
            "cv_score": search.best_score_,
        }

        progress_bar.progress((idx + 1) / total)

    status_text.text("✅ All models trained successfully!")
    return results


def evaluate_model(model, X_test, y_test, model_name: str):
    """
    Evaluate a single model and return metrics plus predictions.
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }

    # Binary classification specific ROC-AUC
    if len(np.unique(y_test)) == 2 and y_pred_proba is not None:
        try:
            metrics["ROC-AUC"] = roc_auc_score(y_test, y_pred_proba[:, 1])
        except Exception:
            metrics["ROC-AUC"] = np.nan
    else:
        metrics["ROC-AUC"] = np.nan

    return metrics, y_pred, y_pred_proba


def evaluate_all_models(
    model_results: Dict[str, Dict[str, Any]],
    X_test,
    y_test,
) -> Dict[str, Any]:
    """
    Evaluate all trained models on test set and build comparison table.
    """
    all_metrics = []
    per_model_details: Dict[str, Dict[str, Any]] = {}

    for name, cfg in model_results.items():
        model = cfg["model"]
        metrics, y_pred, y_pred_proba = evaluate_model(model, X_test, y_test, name)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # ROC curve (binary only)
        roc_data = None
        if len(np.unique(y_test)) == 2 and y_pred_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
            roc_auc_val = auc(fpr, tpr)
            roc_data = {"fpr": fpr, "tpr": tpr, "auc": roc_auc_val}
        else:
            roc_data = None

        all_metrics.append(
            {
                "Model": name,
                "Accuracy": metrics["Accuracy"],
                "Precision": metrics["Precision"],
                "Recall": metrics["Recall"],
                "F1-Score": metrics["F1-Score"],
                "ROC-AUC": metrics.get("ROC-AUC", np.nan),
                "Training Time (s)": cfg["training_time"],
                "Best Params": str(cfg["best_params"]),
            }
        )

        per_model_details[name] = {
            "metrics": metrics,
            "y_pred": y_pred,
            "y_pred_proba": y_pred_proba,
            "confusion_matrix": cm,
            "roc": roc_data,
            "best_params": cfg["best_params"],
        }

    comparison_df = pd.DataFrame(all_metrics)

    # Choose best model by F1-Score
    best_idx = comparison_df["F1-Score"].idxmax()
    best_model_name = comparison_df.loc[best_idx, "Model"]
    best_model_reasoning = (
        f"Selected based on highest F1-Score: "
        f"{comparison_df.loc[best_idx, 'F1-Score']:.4f}."
    )

    return {
        "comparison_df": comparison_df,
        "per_model_details": per_model_details,
        "best_model": best_model_name,
        "best_model_reasoning": best_model_reasoning,
    }




