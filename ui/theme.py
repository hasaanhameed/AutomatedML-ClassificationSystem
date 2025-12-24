import streamlit as st


def apply_dark_theme() -> None:
    """Apply a dark theme with black and purple tones for improved readability."""
    dark_theme_css = """
    <style>
    html, body, [data-testid="stAppViewContainer"], .stApp {
        background-color: #050509 !important;
        color: #e9d5ff !important;
    }
    [data-testid="stHeader"], [data-testid="stToolbar"] {
        background-color: #050509 !important;
        box-shadow: none !important;
    }
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
    .stApp p, .stApp span, .stApp li, .stApp label {
        color: #e9d5ff !important;
    }
    .stApp a {
        color: #c4b5fd !important;
    }
    .stSidebar, [data-testid="stSidebar"] {
        background-color: #090918 !important;
    }
    .stButton>button, .stDownloadButton>button {
        background: linear-gradient(90deg, #4c1d95, #7c3aed);
        color: #f9f9ff;
        border-radius: 0.5rem;
        border: 1px solid #a855f7;
    }
    .stButton>button:hover, .stDownloadButton>button:hover {
        background: linear-gradient(90deg, #6d28d9, #a855f7);
        border-color: #c4b5fd;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.4rem;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #111827;
        color: #e5e7eb;
        border-radius: 0.5rem 0.5rem 0 0;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #1f2937;
        color: #f9fafb;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4c1d95 !important;
        color: #f9fafb !important;
    }
    .block-container {
        padding-top: 1.4rem;
    }
    /* Ensure uploader text is readable inside dropzone */
    .stFileUploader div, .stFileUploader label, .stFileUploader span {
        color: #e9d5ff !important;
    }
    /* Darken uploader dropzone */
    [data-testid="stFileUploaderDropzone"] {
        background-color: #0b0b12 !important;
        border: 1px solid #2d1a4f !important;
    }
    /* Uploaded filename visibility */
    .uploadedFileName {
        color: #f2c9ff !important;
        font-weight: 600;
    }
    /* Uploaded file size text */
    .uploadedFileName + span {
        color: #c4b5fd !important;
    }
    /* Browse files button (white background -> dark purple text) */
    [data-testid="stFileUploader"] button {
        color: #4c1d95 !important;
        border: 1px solid #c4b5fd !important;
    }
    [data-testid="stFileUploader"] button:hover {
        color: #4c1d95 !important;
        border-color: #a855f7 !important;
    }
    /* Metrics text color */
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
        color: #e9d5ff !important;
    }
    .stMetric label, .stMetric .metric-container {
        color: #e9d5ff !important;
    }
    .stMetric .metric-container .stMetric-value {
        color: #e9d5ff !important;
    }
    /* Select / dropdown menus */
    [data-baseweb="select"],
    [data-baseweb="select"] > div,
    [data-baseweb="select"] div[role="combobox"] {
        background-color: #0b0b12 !important;
        color: #e9d5ff !important;
        border-color: #2d1a4f !important;
    }
    [data-baseweb="select"] * {
        color: #e9d5ff !important;
    }
    [data-baseweb="menu"] {
        background-color: #0b0b12 !important;
        color: #e9d5ff !important;
        border: 1px solid #2d1a4f !important;
    }
    /* Expanders and accordions */
    .streamlit-expanderHeader,
    [data-testid="stExpander"] > details > summary {
        background-color: #0b0b12 !important;
        color: #e9d5ff !important;
    }
    [data-testid="stExpander"] > details > summary:hover {
        background-color: #111827 !important;
        color: #f2c9ff !important;
    }
    /* Tables and dataframes */
    [data-testid="stDataFrame"] table,
    [data-testid="stTable"] table {
        background-color: #0b0b12 !important;
        color: #e9d5ff !important;
    }
    [data-testid="stDataFrame"] th, [data-testid="stDataFrame"] td,
    [data-testid="stTable"] th, [data-testid="stTable"] td {
        background-color: #0b0b12 !important;
        color: #e9d5ff !important;
        border-color: #1f1f2b !important;
    }
    [data-testid="stDataFrame"] div, [data-testid="stTable"] div {
        color: #e9d5ff !important;
    }
    /* Pandas Styler and grid backgrounds inside dataframes */
    [data-testid="stDataFrame"] [class*="blank"], 
    [data-testid="stDataFrame"] [class*="row_heading"], 
    [data-testid="stDataFrame"] [class*="col_heading"] {
        background-color: #0b0b12 !important;
        color: #e9d5ff !important;
    }
    /* JSON / text blocks rendered via markdown or code */
    .stMarkdown pre, .stCode, code, pre {
        background-color: #0b0b12 !important;
        color: #e9d5ff !important;
        border-radius: 0.35rem;
        border: 1px solid #1f1f2b;
    }
    /* JSON viewer (e.g., preprocessing summary via st.json) */
    [data-testid="stJson"] {
        background-color: #050509 !important;
        color: #4c1d95 !important;
        border-radius: 0.35rem;
    }
    [data-testid="stJson"] * {
        color: #4c1d95 !important;
    }
    /* Success/info boxes keep dark background */
    .stAlert {
        background-color: #0b0b12 !important;
        color: #e9d5ff !important;
        border: 1px solid #2d1a4f !important;
    }
    /* Generic rule: dark backgrounds -> light purple text */
    .stApp, .stMarkdown, .stText, .stHeading, .stCaption, .stSubheader, .stHeader {
        color: #e9d5ff !important;
    }
    /* Links hover to brighter purple */
    a:hover {
        color: #f2c9ff !important;
    }
    /* Uploaded filename visibility */
    .uploadedFileName {
        color: #f2c9ff !important;
        font-weight: 600;
    }
    /* Metrics text color */
    .stMetric label, .stMetric .metric-container {
        color: #e9d5ff !important;
    }
    .stMetric .metric-container .stMetric-value {
        color: #e9d5ff !important;
    }
    /* Select / dropdown menus */
    [data-baseweb="select"] {
        background-color: #0b0b12 !important;
        color: #e9d5ff !important;
    }
    [data-baseweb="select"] * {
        color: #e9d5ff !important;
    }
    [data-baseweb="menu"] {
        background-color: #0b0b12 !important;
        color: #e9d5ff !important;
    }
    /* Expanders and accordions */
    .streamlit-expanderHeader {
        background-color: #0b0b12 !important;
        color: #e9d5ff !important;
    }
    /* Tables and dataframes */
    [data-testid="stDataFrame"] table,
    [data-testid="stTable"] table {
        background-color: #0b0b12 !important;
        color: #e9d5ff !important;
    }
    [data-testid="stDataFrame"] th, [data-testid="stDataFrame"] td,
    [data-testid="stTable"] th, [data-testid="stTable"] td {
        background-color: #0b0b12 !important;
        color: #e9d5ff !important;
        border-color: #1f1f2b !important;
    }
    [data-testid="stDataFrame"] div, [data-testid="stTable"] div {
        color: #e9d5ff !important;
    }
    /* JSON / text blocks (e.g., preprocessing summary) */
    .stMarkdown pre, .stCode, code, pre {
        background-color: #0b0b12 !important;
        color: #e9d5ff !important;
        border-radius: 0.35rem;
        border: 1px solid #1f1f2b;
    }
    /* Success/info boxes keep dark background */
    .stAlert {
        background-color: #0b0b12 !important;
        color: #e9d5ff !important;
    }
    </style>
    """
    st.markdown(dark_theme_css, unsafe_allow_html=True)

import streamlit as st


def apply_dark_theme() -> None:
    """Apply a dark theme with black and purple tones for improved readability."""
    dark_theme_css = """
    <style>
    html, body, [data-testid="stAppViewContainer"], .stApp {
        background-color: #050509 !important;
        color: #e9d5ff !important;
    }
    [data-testid="stHeader"], [data-testid="stToolbar"] {
        background-color: #050509 !important;
        box-shadow: none !important;
    }
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
    .stApp p, .stApp span, .stApp li, .stApp label {
        color: #e9d5ff !important;
    }
    .stApp a {
        color: #c4b5fd !important;
    }
    .stSidebar, [data-testid="stSidebar"] {
        background-color: #090918 !important;
    }
    .stButton>button, .stDownloadButton>button {
        background: linear-gradient(90deg, #4c1d95, #7c3aed);
        color: #f9f9ff;
        border-radius: 0.5rem;
        border: 1px solid #a855f7;
    }
    .stButton>button:hover, .stDownloadButton>button:hover {
        background: linear-gradient(90deg, #6d28d9, #a855f7);
        border-color: #c4b5fd;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.4rem;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #111827;
        color: #e5e7eb;
        border-radius: 0.5rem 0.5rem 0 0;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #1f2937;
        color: #f9fafb;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4c1d95 !important;
        color: #f9fafb !important;
    }
    .block-container {
        padding-top: 1.4rem;
    }
    /* Ensure uploader text is readable inside dropzone */
    .stFileUploader div, .stFileUploader label, .stFileUploader span {
        color: #e9d5ff !important;
    }
    /* Darken uploader dropzone */
    [data-testid="stFileUploaderDropzone"] {
        background-color: #0b0b12 !important;
        border: 1px solid #2d1a4f !important;
    }
    /* Uploaded filename visibility */
    .uploadedFileName {
        color: #f2c9ff !important;
        font-weight: 600;
    }
    /* Uploaded file size text */
    .uploadedFileName + span {
        color: #c4b5fd !important;
    }
    /* Browse files button (white background -> dark purple text) */
    [data-testid="stFileUploader"] button {
        color: #4c1d95 !important;
        border: 1px solid #c4b5fd !important;
    }
    [data-testid="stFileUploader"] button:hover {
        color: #4c1d95 !important;
        border-color: #a855f7 !important;
    }
    /* Metrics text color */
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
        color: #e9d5ff !important;
    }
    .stMetric label, .stMetric .metric-container {
        color: #e9d5ff !important;
    }
    .stMetric .metric-container .stMetric-value {
        color: #e9d5ff !important;
    }
    /* Select / dropdown menus */
    [data-baseweb="select"],
    [data-baseweb="select"] > div,
    [data-baseweb="select"] div[role="combobox"] {
        background-color: #0b0b12 !important;
        color: #e9d5ff !important;
        border-color: #2d1a4f !important;
    }
    [data-baseweb="select"] * {
        color: #e9d5ff !important;
    }
    [data-baseweb="menu"] {
        background-color: #0b0b12 !important;
        color: #e9d5ff !important;
        border: 1px solid #2d1a4f !important;
    }
    /* Expanders and accordions */
    .streamlit-expanderHeader,
    [data-testid="stExpander"] > details > summary {
        background-color: #0b0b12 !important;
        color: #e9d5ff !important;
    }
    [data-testid="stExpander"] > details > summary:hover {
        background-color: #111827 !important;
        color: #f2c9ff !important;
    }
    /* Tables and dataframes */
    [data-testid="stDataFrame"] table,
    [data-testid="stTable"] table {
        background-color: #0b0b12 !important;
        color: #e9d5ff !important;
    }
    [data-testid="stDataFrame"] th, [data-testid="stDataFrame"] td,
    [data-testid="stTable"] th, [data-testid="stTable"] td {
        background-color: #0b0b12 !important;
        color: #e9d5ff !important;
        border-color: #1f1f2b !important;
    }
    [data-testid="stDataFrame"] div, [data-testid="stTable"] div {
        color: #e9d5ff !important;
    }
    /* Pandas Styler and grid backgrounds inside dataframes */
    [data-testid="stDataFrame"] [class*="blank"], 
    [data-testid="stDataFrame"] [class*="row_heading"], 
    [data-testid="stDataFrame"] [class*="col_heading"] {
        background-color: #0b0b12 !important;
        color: #e9d5ff !important;
    }
    /* JSON / text blocks rendered via markdown or code */
    .stMarkdown pre, .stCode, code, pre {
        background-color: #0b0b12 !important;
        color: #e9d5ff !important;
        border-radius: 0.35rem;
        border: 1px solid #1f1f2b;
    }
    /* JSON viewer (e.g., preprocessing summary via st.json) */
    [data-testid="stJson"] {
        background-color: #050509 !important;
        color: #4c1d95 !important;
        border-radius: 0.35rem;
    }
    [data-testid="stJson"] * {
        color: #4c1d95 !important;
    }
    /* Success/info boxes keep dark background */
    .stAlert {
        background-color: #0b0b12 !important;
        color: #e9d5ff !important;
        border: 1px solid #2d1a4f !important;
    }
    /* Generic rule: dark backgrounds -> light purple text */
    .stApp, .stMarkdown, .stText, .stHeading, .stCaption, .stSubheader, .stHeader {
        color: #e9d5ff !important;
    }
    /* Links hover to brighter purple */
    a:hover {
        color: #f2c9ff !important;
    }
    </style>
    """
    st.markdown(dark_theme_css, unsafe_allow_html=True)

