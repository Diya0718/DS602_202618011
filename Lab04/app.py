import streamlit as st
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Medical Insurance Statistical Analysis",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 Medical Insurance Statistical Analysis")

st.write(
    "Interactive dashboard for exploring medical insurance data, "
    "performing hypothesis tests, and building statistical models."
)

# Load dataset
df = pd.read_csv("insurance.csv")

# Display dataset
st.subheader("Dataset Preview")
st.dataframe(df.head())

# Dataset information
st.subheader("Dataset Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Rows", df.shape[0])

with col2:
    st.metric("Columns", df.shape[1])

with col3:
    st.metric("Missing Values", df.isnull().sum().sum())