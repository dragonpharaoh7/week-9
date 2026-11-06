import pandas as pd
import streamlit as st

from apputil import GroupEstimate


@st.cache_data
def load_data(file):
    """Read the uploaded CSV, or fall back to a small demo table."""
    if file is not None:
        return pd.read_csv(file)

    return pd.DataFrame(
        {
            "loc_country": ["Guatemala", "Guatemala", "Mexico", "Mexico", "Mexico"],
            "roast": ["Light", "Light", "Medium", "Medium", "Dark"],
            "rating": [88.0, 88.8, 91.0, 91.0, 80.0],
        }
    )


@st.cache_resource
def fit_model(df, x_cols, y_col, estimate):
    """Fit (and cache) the model so it isn't refit on every interaction."""
    return GroupEstimate(estimate=estimate).fit(df[list(x_cols)], df[y_col])


st.write(
    """
# Week 9: Group Estimate

Pick the categorical columns to group by and a continuous column to estimate,
then choose a group to get a prediction.
"""
)

upload = st.file_uploader("Data (CSV)", type="csv")
df = load_data(upload)
st.dataframe(df.head())

estimate = st.radio("Estimate", ["mean", "median"], horizontal=True)
x_cols = st.multiselect("Group by (categorical)", df.columns)
y_col = st.selectbox("Estimate column (continuous)", df.columns, index=len(df.columns) - 1)

if x_cols and y_col not in x_cols:
    model = fit_model(df, tuple(x_cols), y_col, estimate)

    choices = [st.selectbox(col, sorted(df[col].dropna().unique())) for col in x_cols]
    prediction = model.predict([choices])[0]

    if pd.isna(prediction):
        st.warning("That combination of categories wasn't in the data.")
    else:
        st.metric(f"Estimated {y_col} ({estimate})", f"{prediction:,.2f}")
