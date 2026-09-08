import streamlit as st
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Medical Insurance Statistical Analysis",
    page_icon="📊",
    layout="wide"
)

# Load dataset
df = pd.read_csv("D:\\DS602_202618011\\insurance.csv")

# Title
st.title("📊 Medical Insurance Statistical Analysis")

st.write(
    "Interactive dashboard for exploring medical insurance data, "
    "performing hypothesis tests, and building statistical models."
)

# Create tabs
tab1, tab2, tab3 = st.tabs([
    "📊 Data Exploration",
    "🧪 Hypothesis Testing Lab",
    "📈 Live Prediction & Diagnostics"
])


# ============================================================
# TAB 1 — DATA EXPLORATION
# ============================================================

with tab1:

    st.header("📊 Data Exploration")

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    st.subheader("Dataset Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())


# ============================================================
# TAB 2 — HYPOTHESIS TESTING
# ============================================================

with tab2:

    st.header("🧪 Hypothesis Testing Lab")

    st.write(
        "Perform statistical hypothesis tests on the medical insurance dataset."
    )

    st.info(
        "Hypothesis tests will be added here in the next step."
    )


# ============================================================
# TAB 3 — PREDICTION & DIAGNOSTICS
# ============================================================

with tab3:

    st.header("📈 Live Prediction & Diagnostics")

    st.write(
        "Regression prediction and model diagnostics will appear here."
    )

    st.info(
        "OLS regression and live prediction will be added here in the next step."
    )