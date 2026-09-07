# lab_eda_gui.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Page Configuration
st.set_page_config(
    page_title="EDA Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Exploratory Data Analysis Interface")

# 2. Sidebar: Dataset Ingestion
st.sidebar.title("EDA TASK")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    # Read dataset
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("File uploaded!")

    # 3. Dataset Overview
    st.subheader("Dataset Overview")
    
    # First 5 rows
    st.write("**First 5 Rows:**")
    st.dataframe(df.head())

    # Shape
    st.write(f"**Shape:** {df.shape[0]} rows, {df.shape[1]} columns")

    # Column Data Types
    st.write("**Column Data Types:**")
    dtype_df = pd.DataFrame(df.dtypes.reset_index())
    dtype_df.columns = ["Column", "Data Type"]
    st.dataframe(dtype_df, use_container_width=True)

    # Missing value summary (count + percentage)
    st.write("**Missing Values per Column:**")
    missing_count = df.isnull().sum()
    missing_percent = (missing_count / len(df)) * 100
    missing_df = pd.DataFrame({
        "Missing Count": missing_count,
        "Missing %": missing_percent.round(2)
    })
    st.dataframe(missing_df, use_container_width=True)

    # Basic statistics for numerical columns
    st.write("**Basic Numerical Statistics:**")
    numeric_cols = df.select_dtypes(include=["number"]).columns
    if len(numeric_cols) > 0:
        st.dataframe(df[numeric_cols].describe().T, use_container_width=True)
    else:
        st.info("No numerical columns found.")

    # 4. Attribute Selection
    st.sidebar.subheader("🔍 Attribute Selection")
    selected_col = st.sidebar.selectbox(
        "Choose a column to visualize:",
        df.columns.tolist()
    )

    # Detect column type
    if pd.api.types.is_numeric_dtype(df[selected_col]) and df[selected_col].nunique() < 10:
        col_type = "Categorical"
    elif pd.api.types.is_numeric_dtype(df[selected_col]):
        col_type = "Numerical"
    else:
        col_type = "Categorical"

    st.sidebar.write(f"**Type:** {col_type}")

    # 5. Visualization Rendering
    st.subheader("Visualization")

    fig, ax = plt.subplots(figsize=(8, 4))

    if col_type == "Numerical":
        sns.histplot(df[selected_col].dropna(), bins=20, kde=False, color="blue", ax=ax)
        ax.set_title(f"Distribution of {selected_col}")
        ax.set_xlabel(selected_col)
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

    else:  # Categorical
        counts = df[selected_col].value_counts()
        total = counts.sum()
        percentages = (counts / total * 100).round(1)

        bars = ax.bar(counts.index.astype(str), counts.values, color="skyblue", edgecolor="black")
        ax.set_title(f"Frequency of {selected_col}")
        ax.set_xlabel(selected_col)
        ax.set_ylabel("Count")

        for i, (idx, val) in enumerate(counts.items()):
            pct = percentages[idx]
            ax.text(i, val + 0.5, f"{pct}%", ha="center", fontsize=9)

        st.pyplot(fig)

else:
    st.info("Please upload a CSV file to start EDA.")