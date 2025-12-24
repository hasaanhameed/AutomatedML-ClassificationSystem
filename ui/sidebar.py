import streamlit as st


def sidebar_navigation() -> str:
    with st.sidebar:
        st.title("AutoML System")
        st.markdown("---")
        page = st.radio(
            "Navigation",
            [
                "Upload Dataset",
                "EDA",
                "Preprocessing",
                "Model Training",
                "Results",
                "Report",
            ],
        )
    return page

import streamlit as st


def sidebar_navigation() -> str:
    with st.sidebar:
        st.title("AutoML System")
        st.markdown("---")
        page = st.radio(
            "Navigation",
            [
                "Upload Dataset",
                "EDA",
                "Preprocessing",
                "Model Training",
                "Results",
                "Report",
            ],
        )
    return page

