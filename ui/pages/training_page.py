import pandas as pd
import streamlit as st

from utils.models import train_all_models


def page_training() -> None:
    if st.session_state.train_test_data is None:
        st.warning(
            "Please apply preprocessing first on the 'Preprocessing' page "
            "to generate train-test split."
        )
        return

    st.subheader("Model Training & Hyperparameter Optimization")

    X_train, X_test, y_train, y_test = st.session_state.train_test_data

    st.markdown("### Models to be Trained")
    st.markdown(
        "- Logistic Regression\n"
        "- K-Nearest Neighbors\n"
        "- Decision Tree\n"
        "- Naive Bayes\n"
        "- Random Forest\n"
        "- Support Vector Machine\n"
        "- Rule-Based Classifier"
    )

    st.info(
        "All seven models listed above will be trained with automated hyperparameter optimization "
        "and evaluated on the held-out test set."
    )

    st.markdown("### Hyperparameter Optimization Method")
    search_method = st.radio(
        "Select how hyperparameters should be searched:",
        ["GridSearchCV", "RandomizedSearchCV"],
    )

    st.info(
        "Selecting GridSearchCV will run an exhaustive search over the full hyperparameter grid. "
        "When GridSearchCV is selected, Random Forest training in particular may take significantly longer, "
        "and this is expected behavior. For faster experimentation, you can choose RandomizedSearchCV, "
        "which samples a subset of the hyperparameter space and typically runs much faster."
    )

    train_btn = st.button("Train All Models")

    if train_btn:
        with st.spinner("Training models..."):
            progress_bar = st.progress(0)
            status_text = st.empty()

            results = train_all_models(
                X_train, y_train, search_method, progress_bar, status_text
            )

        st.session_state.model_results = results
        st.success("✅ All models trained successfully!")
        st.session_state.current_step = max(st.session_state.current_step, 5)

        st.markdown("### Cross-Validation Results")
        cv_df = pd.DataFrame(
            [
                {
                    "Model": name,
                    "CV F1-Score": cfg["cv_score"],
                    "Training Time (s)": cfg["training_time"],
                }
                for name, cfg in results.items()
            ]
        )
        st.dataframe(cv_df)


