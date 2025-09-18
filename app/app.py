import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("models/flight_price_model.pkl")

st.title("✈️ Flight Price Prediction App")

st.write("Enter flight details below to predict the price:")

# Example input fields (modify as per dataset features)
airline = st.selectbox("Airline", ["IndiGo", "Air India", "SpiceJet", "Vistara"])
source = st.selectbox("Source", ["Delhi", "Mumbai", "Kolkata", "Chennai"])
destination = st.selectbox("Destination", ["Cochin", "Bangalore", "Delhi", "Hyderabad"])
duration = st.number_input("Duration (in hours)", min_value=1.0, max_value=30.0, step=0.5)
total_stops = st.selectbox("Total Stops", [0, 1, 2, 3, 4])

if st.button("Predict Price"):
    # Convert input into dataframe for prediction
    input_dict = {
        "Duration": [duration],
        "Total_Stops": [total_stops],
        f"Airline_{airline}": [1],
        f"Source_{source}": [1],
        f"Destination_{destination}": [1],
    }

    # Fill missing dummy variables
    input_df = pd.DataFrame(input_dict)
    all_features = model.feature_names_in_
    # Ensure all features are present (missing ones will be filled with 0)
    input_df = input_df.reindex(columns=all_features, fill_value=0)


    # Predict
    prediction = model.predict(input_df)[0]
    st.success(f"Estimated Flight Price: ₹ {int(prediction):,}")
