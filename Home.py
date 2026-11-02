import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Airline Satisfaction",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_PATH = Path(__file__).parent / "cleaned_data.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

st.title("✈️ Airline Passenger Satisfaction")
st.markdown(
    "An interactive machine-learning app to explore passenger experience "
    "and predict whether a passenger is likely to be satisfied."
)

try:
    df = load_data()
    total = len(df)
    satisfied = int(df["satisfaction"].astype(bool).sum())
    satisfaction_rate = satisfied / total * 100 if total else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Total passengers", f"{total:,}")
    c2.metric("Satisfied passengers", f"{satisfied:,}")
    c3.metric("Satisfaction rate", f"{satisfaction_rate:.1f}%")
except Exception:
    st.info("Place cleaned_data.csv in the same folder as Home.py to show dataset summary.")

st.divider()
left, right = st.columns([1.2, 1])
with left:
    st.subheader("What's inside the app?")
    st.markdown("""
    - **Visualization:** explore satisfaction distribution, passenger segments,
      service ratings, and flight delays.
    - **Prediction:** enter passenger and flight details to get a satisfaction
      prediction from the trained XGBoost pipeline.
    """)
with right:
    st.subheader("How to navigate")
    st.markdown("""
    Use the pages in the left sidebar:
    1. **Visualization** to understand patterns in the dataset.
    2. **Prediction** to test a passenger profile.
    """)

st.caption("Model target: satisfaction (True = satisfied, False = neutral or dissatisfied).")
