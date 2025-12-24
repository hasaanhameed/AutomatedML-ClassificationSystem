import streamlit as st

from utils.preprocessing import (
    detect_issues,
    apply_preprocessing_from_strategies,
)


def page_preprocessing():
    if st.session_state.df is None:
        st.warning("Please upload a dataset first on the 'Upload Dataset' page.")
        return
    if st.session_state.target_col is None:
        st.warning("Please select a target column on the 'Upload Dataset' page.")
        return

    st.subheader("Data Issues & Preprocessing")

    df = st.session_state.df.copy()
    target_col = st.session_state.target_col

    # Issue detection based on EDA or direct checks
    issues = detect_issues(df, target_col, st.session_state.eda_results)

    st.markdown("### Detected Issues")
    if not issues:
        st.success("No major issues detected. You can still configure preprocessing.")
    else:
        for issue in issues:
            severity = issue["severity"]
            msg = issue["message"]
            key = issue["key"]
            options = issue["options"]

            if severity == "critical":
                st.error(msg)
            elif severity == "warning":
                st.warning(msg)
            else:
                st.info(msg)

            choice = st.radio(
                issue["question"],
                options,
                key=key,
            )
            st.session_state.issue_strategies[key] = choice

    st.markdown("---")
    st.markdown("### Global Preprocessing Settings")

    # Scaling
    scaler_choice = st.selectbox(
        "Feature Scaling Method",
        ["StandardScaler", "MinMaxScaler", "RobustScaler", "None"],
        index=0,
    )

    # Encoding
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    encoding_strategies = {}
    if len(categorical_cols) > 0:
        st.markdown("#### Categorical Encoding")
        for col in categorical_cols:
            encoding = st.radio(
                f"Encoding for {col}",
                ["One-Hot Encoding", "Label Encoding", "Ordinal Encoding"],
                key=f"encoding_{col}",
            )
            encoding_strategies[col] = encoding

    # Feature engineering
    st.markdown("#### Feature Engineering (Optional)")
    use_polynomial = st.checkbox("Create polynomial features for numeric columns")

    # Train-test split
    st.markdown("#### Train-Test Split")
    test_size_pct = st.slider("Test Set Size (%)", 10, 40, 20, 5)
    random_state = st.number_input("Random State", 0, 1000, 42)

    st.markdown("---")
    st.markdown("### Confirmation")

    if st.session_state.issue_strategies:
        st.write("**Selected issue handling strategies:**")
        st.json(st.session_state.issue_strategies)

    apply_btn = st.button("Apply All Preprocessing")
    reset_btn = st.button("Reset Selections")

    if reset_btn:
        st.session_state.issue_strategies = {}
        st.info("Selections reset. Reconfigure preprocessing options.")

    if apply_btn:
        with st.spinner("Applying preprocessing..."):
            (
                processed_df,
                train_test_data,
                preprocess_summary,
            ) = apply_preprocessing_from_strategies(
                df=df,
                target_col=target_col,
                issue_strategies=st.session_state.issue_strategies,
                scaler_choice=scaler_choice,
                encoding_strategies=encoding_strategies,
                test_size=test_size_pct / 100.0,
                random_state=int(random_state),
                use_polynomial=use_polynomial,
            )

        st.session_state.processed_df = processed_df
        st.session_state.train_test_data = train_test_data
        st.success("✅ Preprocessing applied.")
        st.session_state.current_step = max(st.session_state.current_step, 4)

        st.markdown("#### Preprocessing Summary")
        st.json(preprocess_summary)

        st.markdown("#### Processed Data Preview")
        st.write(f"Shape: {processed_df.shape}")
        st.dataframe(processed_df.head())

        csv = processed_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Processed Dataset",
            csv,
            "processed_dataset.csv",
            "text/csv",
        )

