from typing import Dict, Any, Tuple, Optional

import streamlit as st

from ui.theme import apply_dark_theme
from ui.sidebar import sidebar_navigation
from ui.header import render_header
from ui.pages.upload_page import page_upload
from ui.pages.eda_page import page_eda
from ui.pages.preprocessing_page import page_preprocessing
from ui.pages.training_page import page_training
from ui.pages.results_page import page_results
from ui.pages.report_page import page_report


st.set_page_config(
    page_title="AutoML Classification System",
    page_icon="A",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_session_state() -> None:
    if "df" not in st.session_state:
        st.session_state.df = None
    if "target_col" not in st.session_state:
        st.session_state.target_col = None
    if "eda_results" not in st.session_state:
        st.session_state.eda_results = None
    if "issue_strategies" not in st.session_state:
        st.session_state.issue_strategies: Dict[str, Any] = {}
    if "processed_df" not in st.session_state:
        st.session_state.processed_df = None
    if "train_test_data" not in st.session_state:
        st.session_state.train_test_data: Optional[Tuple] = None
    if "model_results" not in st.session_state:
        st.session_state.model_results: Dict[str, Any] = {}
    if "evaluation_results" not in st.session_state:
        st.session_state.evaluation_results: Dict[str, Any] = {}
    if "best_model_name" not in st.session_state:
        st.session_state.best_model_name = None
    if "current_step" not in st.session_state:
        st.session_state.current_step = 1


def main():
    init_session_state()
    page = sidebar_navigation()
    apply_dark_theme()
    render_header()

    if page == "Upload Dataset":
        page_upload()
    elif page == "EDA":
        page_eda()
    elif page == "Preprocessing":
        page_preprocessing()
    elif page == "Model Training":
        page_training()
    elif page == "Results":
        page_results()
    elif page == "Report":
        page_report()


if __name__ == "__main__":
    main()


