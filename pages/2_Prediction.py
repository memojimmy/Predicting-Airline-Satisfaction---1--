import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="Satisfaction Prediction", page_icon="🔮", layout="wide")

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "xgboost_model.pkl"

st.title("🔮 Passenger Satisfaction Prediction")
st.caption("Complete the passenger and flight details, then run the trained XGBoost pipeline.")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Copy xgboost_model.pkl (created by joblib.dump "
        "in your notebook) into the project root beside Home.py."
    )
    st.stop()

try:
    model = load_model()
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.stop()

# These are the 21 input columns used by the notebook's XGBoost pipeline.
# id is excluded (it was loaded as the dataframe index), and satisfaction is the target.
with st.form("prediction_form"):
    st.subheader("Passenger information")
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        age = st.number_input("Age", min_value=7, max_value=85, value=35, step=1)
    with p2:
        gender = st.selectbox("Gender", ["Female", "Male"])
    with p3:
        customer_type = st.selectbox("Customer Type", ["Loyal Customer", "disloyal Customer"])
    with p4:
        travel_type = st.selectbox("Type of Travel", ["Business travel", "Personal Travel"])

    st.subheader("Flight information")
    f1, f2 = st.columns(2)
    with f1:
        flight_distance = st.number_input(
            "Flight Distance", min_value=67, max_value=4983, value=1000, step=1
        )
    with f2:
        travel_class = st.selectbox("Class", ["Business", "Eco", "Eco Plus"])

    st.subheader("Service ratings")
    st.caption("Ratings use the values available in the dataset (0–5, except Baggage handling: 1–5).")
    rating_cols = [
        ("Inflight wifi service", "wifi"),
        ("Departure/Arrival time convenient", "time_convenient"),
        ("Ease of Online booking", "online_booking"),
        ("Gate location", "gate_location"),
        ("Food and drink", "food_drink"),
        ("Online boarding", "online_boarding"),
        ("Seat comfort", "seat_comfort"),
        ("Inflight entertainment", "entertainment"),
        ("On-board service", "onboard_service"),
        ("Leg room service", "leg_room"),
        ("Baggage handling", "baggage"),
        ("Checkin service", "checkin"),
        ("Cleanliness", "cleanliness"),
    ]
    rating_values = {}
    for start in range(0, len(rating_cols), 3):
        cols = st.columns(3)
        for col, (label, key) in zip(cols, rating_cols[start:start+3]):
            with col:
                min_rating = 1 if key == "baggage" else 0
                rating_values[label] = st.select_slider(
                    label, options=list(range(min_rating, 6)), value=4
                )

    st.subheader("Flight delays")
    d1, d2 = st.columns(2)
    with d1:
        departure_delay = st.number_input(
            "Departure Delay in Minutes", min_value=0, max_value=489, value=0, step=1
        )
    with d2:
        arrival_delay = st.number_input(
            "Arrival Delay in Minutes", min_value=0.0, max_value=491.0,
            value=0.0, step=1.0
        )

    submitted = st.form_submit_button("Predict satisfaction", type="primary", use_container_width=True)

if submitted:
    input_data = {
        "Age": age,
        "Flight Distance": flight_distance,
        "Inflight wifi service": rating_values["Inflight wifi service"],
        "Departure/Arrival time convenient": rating_values["Departure/Arrival time convenient"],
        "Ease of Online booking": rating_values["Ease of Online booking"],
        "Gate location": rating_values["Gate location"],
        "Food and drink": rating_values["Food and drink"],
        "Online boarding": rating_values["Online boarding"],
        "Seat comfort": rating_values["Seat comfort"],
        "Inflight entertainment": rating_values["Inflight entertainment"],
        "On-board service": rating_values["On-board service"],
        "Leg room service": rating_values["Leg room service"],
        "Baggage handling": rating_values["Baggage handling"],
        "Checkin service": rating_values["Checkin service"],
        "Cleanliness": rating_values["Cleanliness"],
        "Departure Delay in Minutes": departure_delay,
        "Arrival Delay in Minutes": arrival_delay,
        "Gender": gender,
        "Customer Type": customer_type,
        "Type of Travel": travel_type,
        "Class": travel_class,
    }

    # Preserve the exact training feature order.
    feature_order = [
        "Age", "Flight Distance", "Inflight wifi service",
        "Departure/Arrival time convenient", "Ease of Online booking",
        "Gate location", "Food and drink", "Online boarding", "Seat comfort",
        "Inflight entertainment", "On-board service", "Leg room service",
        "Baggage handling", "Checkin service", "Cleanliness",
        "Departure Delay in Minutes", "Arrival Delay in Minutes",
        "Gender", "Customer Type", "Type of Travel", "Class"
    ]
    input_df = pd.DataFrame([input_data], columns=feature_order)

    try:
        prediction = model.predict(input_df)[0]
        classes = getattr(model, "classes_", None)
        probability_satisfied = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            if classes is None:
                classes = model.named_steps[list(model.named_steps.keys())[-1]].classes_
            true_indices = [i for i, value in enumerate(classes) if str(value).lower() == "true" or value == 1]
            if true_indices:
                probability_satisfied = float(probabilities[true_indices[0]])

        is_satisfied = str(prediction).lower() == "true" or prediction == 1
        st.divider()
        if is_satisfied:
            st.success("Prediction: Satisfied")
        else:
            st.warning("Prediction: Neutral or Dissatisfied")

        if probability_satisfied is not None:
            st.metric("Estimated probability of satisfaction", f"{probability_satisfied:.1%}")
            st.progress(min(max(probability_satisfied, 0.0), 1.0))
        st.caption("This is a model prediction, not a guarantee of the passenger's actual experience.")
        with st.expander("Review submitted inputs"):
            st.dataframe(input_df, use_container_width=True)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.info("Check that this model was trained with the same 21 feature columns and preprocessing pipeline.")
