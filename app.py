import streamlit as st
import pandas as pd
from model_loader import load_model_file
from debugger import analyze_model
from utils import check_data

st.set_page_config(page_title="ML Model Debugger", layout="centered")

st.title("🔍 ML Model Debugger")

# Upload dataset
data_file = st.file_uploader("Upload Dataset (CSV)", type=["csv"])

# Upload model
model_file = st.file_uploader("Upload Model (.pkl)", type=["pkl"])

if data_file is not None:
    df = pd.read_csv(data_file)
    st.write("### Dataset Preview")
    st.dataframe(df.head())

    # Data check
    st.write("### Data Issues")
    issues = check_data(df)
    for issue in issues:
        st.warning(issue)

if model_file is not None:
    model = load_model_file(model_file)
    st.success("Model Loaded Successfully!")

    if data_file is not None:
        result = analyze_model(model, df)

        st.write("### Model Analysis")
        st.write(f"Accuracy: {result['accuracy']:.2f}")

        if result["overfitting"]:
            st.error("Model may be Overfitting")

        if result["underfitting"]:
            st.warning("Model may be Underfitting")