import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Data Visualization", page_icon="📊", layout="wide")
DATA_PATH = Path(__file__).resolve().parents[1] / "cleaned_data.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

st.title("📊 Passenger Data Visualization")
st.caption("Explore the cleaned Airline Passenger Satisfaction dataset.")

try:
    df = load_data()
except FileNotFoundError:
    st.error("cleaned_data.csv was not found. Put it in the project root beside Home.py.")
    st.stop()

# Normalize target only for chart labels; the original dataset is not modified.
df["Satisfaction Label"] = df["satisfaction"].map(
    {True: "Satisfied", False: "Neutral / Dissatisfied",
     "True": "Satisfied", "False": "Neutral / Dissatisfied"}
)

total = len(df)
sat_count = int((df["Satisfaction Label"] == "Satisfied").sum())
not_sat_count = total - sat_count
sat_rate = sat_count / total * 100 if total else 0

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total passengers", f"{total:,}")
k2.metric("Satisfied", f"{sat_count:,}")
k3.metric("Neutral / dissatisfied", f"{not_sat_count:,}")
k4.metric("Satisfaction rate", f"{sat_rate:.1f}%")

st.divider()

st.subheader("Filter the dataset")
f1, f2, f3 = st.columns(3)
gender_options = ["All"] + sorted(df["Gender"].dropna().unique().tolist())
travel_options = ["All"] + sorted(df["Type of Travel"].dropna().unique().tolist())
class_options = ["All"] + sorted(df["Class"].dropna().unique().tolist())

gender_filter = f1.selectbox("Gender", gender_options)
travel_filter = f2.selectbox("Type of travel", travel_options)
class_filter = f3.selectbox("Travel class", class_options)

filtered = df.copy()
if gender_filter != "All":
    filtered = filtered[filtered["Gender"] == gender_filter]
if travel_filter != "All":
    filtered = filtered[filtered["Type of Travel"] == travel_filter]
if class_filter != "All":
    filtered = filtered[filtered["Class"] == class_filter]

st.caption(f"Showing {len(filtered):,} passengers after filtering.")

# Two-column dashboard
left, right = st.columns(2)

with left:
    st.subheader("Satisfaction distribution")
    sat_dist = (
        filtered["Satisfaction Label"]
        .value_counts()
        .rename_axis("Satisfaction")
        .reset_index(name="Passengers")
    )
    fig = px.pie(
        sat_dist, names="Satisfaction", values="Passengers",
        hole=0.58, color="Satisfaction",
        color_discrete_map={
            "Satisfied": "#16A34A",
            "Neutral / Dissatisfied": "#64748B"
        }
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(height=360, margin=dict(t=20, b=20, l=10, r=10))
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Satisfaction by travel class")
    by_class = (
        filtered.groupby("Class", observed=False)["satisfaction"]
        .mean().mul(100).reset_index(name="Satisfaction rate (%)")
    )
    fig = px.bar(
        by_class, x="Class", y="Satisfaction rate (%)",
        color="Class", text_auto=".1f"
    )
    fig.update_layout(height=360, showlegend=False,
                      yaxis_title="Satisfied passengers (%)",
                      margin=dict(t=20, b=20, l=10, r=10))
    st.plotly_chart(fig, use_container_width=True)

left, right = st.columns(2)
with left:
    st.subheader("Satisfaction by type of travel")
    by_travel = (
        filtered.groupby("Type of Travel", observed=False)["satisfaction"]
        .mean().mul(100).reset_index(name="Satisfaction rate (%)")
    )
    fig = px.bar(
        by_travel, x="Type of Travel", y="Satisfaction rate (%)",
        color="Type of Travel", text_auto=".1f"
    )
    fig.update_layout(height=350, showlegend=False,
                      yaxis_title="Satisfied passengers (%)",
                      margin=dict(t=20, b=20, l=10, r=10))
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Average service ratings")
    rating_cols = [
        "Inflight wifi service", "Departure/Arrival time convenient",
        "Ease of Online booking", "Gate location", "Food and drink",
        "Online boarding", "Seat comfort", "Inflight entertainment",
        "On-board service", "Leg room service", "Baggage handling",
        "Checkin service", "Cleanliness"
    ]
    rating_means = (
        filtered[rating_cols].mean().sort_values()
        .rename_axis("Service").reset_index(name="Average rating")
    )
    fig = px.bar(
        rating_means, x="Average rating", y="Service",
        orientation="h", color="Average rating",
        color_continuous_scale="Blues"
    )
    fig.update_layout(height=430, margin=dict(t=20, b=20, l=10, r=10))
    st.plotly_chart(fig, use_container_width=True)

st.divider()
left, right = st.columns(2)

with left:
    st.subheader("Passenger age distribution")
    fig = px.histogram(
        filtered, x="Age", color="Satisfaction Label",
        nbins=35, barmode="overlay", opacity=0.75
    )
    fig.update_layout(height=350, margin=dict(t=20, b=20, l=10, r=10))
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Flight delays")
    delay_data = filtered[[
        "Departure Delay in Minutes", "Arrival Delay in Minutes"
    ]].melt(var_name="Delay type", value_name="Minutes").dropna()
    # Cap only the chart display at the 99th percentile so extreme delays
    # do not compress the main distribution. Raw data remains unchanged.
    upper = delay_data["Minutes"].quantile(0.99)
    delay_data = delay_data[delay_data["Minutes"] <= upper]
    fig = px.histogram(
        delay_data, x="Minutes", color="Delay type",
        nbins=40, barmode="overlay", opacity=0.65
    )
    fig.update_layout(height=350, margin=dict(t=20, b=20, l=10, r=10))
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Dataset preview")
st.dataframe(filtered.drop(columns=["Satisfaction Label"]).head(20), use_container_width=True)
