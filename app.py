



import streamlit as st
import pandas as pd
import joblib
import os

if not os.path.exists('housing_model.pkl') or not os.path.exists('scaler.pkl'):
    st.error("Model or scaler file not found. Please ensure 'housing_model.pkl' and 'scaler.pkl' are in the same directory as this script.")
else:
    model = joblib.load('housing_model.pkl')
    scaler = joblib.load('scaler.pkl')

    st.title('California Housing Price Predictor')

    # User choice: Upload CSV or Input Manually
    input_method = st.radio("Choose input method:", ('Upload CSV File', 'Input Features Manually'))

    if input_method == 'Upload CSV File':
        # File upload
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

        if uploaded_file:
            # Read and preprocess
            new_data = pd.read_csv(uploaded_file)

            # Drop columns not used during training
            if 'median_house_value' in new_data.columns:
                new_data = new_data.drop(columns=['median_house_value'])

            # Ensure the columns match the expected features
            expected_features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'ocean_proximity']

            # Reindex to match expected features
            new_data = new_data.reindex(columns=expected_features, fill_value=0)

            # Handle categorical feature (ensure consistency with training)
            if 'ocean_proximity' in new_data.columns:
                # Map categories to integers if this was done during training
                category_mapping = {'<1H OCEAN': 0, 'INLAND': 1, 'ISLAND': 2, 'NEAR BAY': 3, 'NEAR OCEAN': 4}
                new_data['ocean_proximity'] = new_data['ocean_proximity'].map(category_mapping).fillna(0)

            # Scale the data
            scaled_data = scaler.transform(new_data)

            # Predict
            predictions = model.predict(scaled_data)
            new_data['Predicted_Price'] = predictions

            # Display
            st.write("Predictions:")
            st.write(new_data)

            # Download
            st.download_button(
                label="Download Predictions",
                data=new_data.to_csv(index=False),
                file_name='predictions.csv'
            )
    elif input_method == 'Input Features Manually':
        # Manual input
        st.subheader("Enter the features manually:")
        longitude = st.number_input("Longitude", value=-120.0)
        latitude = st.number_input("Latitude", value=35.0)
        housing_median_age = st.number_input("Housing Median Age", value=25)
        total_rooms = st.number_input("Total Rooms", value=1000)
        total_bedrooms = st.number_input("Total Bedrooms", value=200)
        population = st.number_input("Population", value=500)
        households = st.number_input("Households", value=150)
        median_income = st.number_input("Median Income", value=3.0)
        ocean_proximity = st.selectbox("Ocean Proximity", options=['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'])

        # Map ocean proximity to integers
        category_mapping = {'<1H OCEAN': 0, 'INLAND': 1, 'ISLAND': 2, 'NEAR BAY': 3, 'NEAR OCEAN': 4}
        ocean_proximity = category_mapping[ocean_proximity]

        # Create a DataFrame for prediction
        manual_data = pd.DataFrame([{
            'longitude': longitude,
            'latitude': latitude,
            'housing_median_age': housing_median_age,
            'total_rooms': total_rooms,
            'total_bedrooms': total_bedrooms,
            'population': population,
            'households': households,
            'median_income': median_income,
            'ocean_proximity': ocean_proximity
        }])

        # Scale the data
        scaled_data = scaler.transform(manual_data)

        # Predict
        prediction = model.predict(scaled_data)[0]
        st.write(f"Predicted Price: ${prediction:,.2f}")

    # Save artifacts
    joblib.dump(model, 'housing_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    #----Run in terminal-----
    #cd "C:\Users\Ahmed\Desktop\linear regression project"
    #add