import streamlit as st


def render_header() -> None:
    st.title("Automated Machine Learning Classification System")
    st.markdown("Upload your dataset and let the system handle the complete classification workflow.")

    if "current_step" in st.session_state:
        progress = st.session_state.current_step / 6
        st.progress(progress)

        steps = ["Upload", "EDA", "Issues", "Preprocessing", "Training", "Results"]
        current_idx = max(0, min(len(steps) - 1, st.session_state.current_step - 1))
        current = steps[current_idx]
        st.caption(f"Current Step: {current}")

import streamlit as st


def render_header():
    st.title("Automated Machine Learning Classification System")
    st.markdown("Upload your dataset and let the system handle the complete classification workflow.")

    if "current_step" in st.session_state:
        progress = st.session_state.current_step / 6
        st.progress(progress)

        steps = ["Upload", "EDA", "Issues", "Preprocessing", "Training", "Results"]
        current_idx = max(0, min(len(steps) - 1, st.session_state.current_step - 1))
        current = steps[current_idx]
        st.caption(f"Current Step: {current}")

