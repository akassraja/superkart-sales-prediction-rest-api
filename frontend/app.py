import streamlit as st
import pandas as pd
import requests

# Base URL of the Flask backend
BACKEND_URL = "http://backend:7860"

# Set the title of the Streamlit app
st.title("SuperKart Sales Prediction")

# Section for online prediction
st.subheader("Online Prediction")

# Collect user input for product and store features (based on the SuperKart model)
product_weight = st.number_input("Product Weight (kg)", min_value=4.0, max_value=22.0, value=12.0, step=0.1)
product_sugar_content = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
product_allocated_area = st.number_input("Product Allocated Area (ratio)", min_value=0.004, max_value=0.298, value=0.05, step=0.001, format="%.3f")
product_mrp = st.number_input("Product MRP (Maximum Retail Price)", min_value=31.0, max_value=266.0, value=150.0, step=0.01)
store_id = st.selectbox("Store ID", ['OUT004', 'OUT003', 'OUT001', 'OUT002'])
store_size = st.selectbox("Store Size", ["Medium", "High", "Small"])
store_location_city_type = st.selectbox("Store Location City Type", ["Tier 2", "Tier 1", "Tier 3"])
store_type = st.selectbox("Store Type", ["Supermarket Type2", "Departmental Store", "Supermarket Type1", "Food Mart"])
store_age_years = st.number_input("Store Age (Years)", min_value=0, max_value=22, value=10, step=1)
product_id_prefix = st.selectbox("Product ID Prefix", ['FD', 'NC', 'DR'])
perishable = st.selectbox("Perishable (1=Yes, 0=No)", [1, 0])

# Convert user input into a dictionary for the payload
payload = {
    'Product_Weight': product_weight,
    'Product_Sugar_Content': product_sugar_content,
    'Product_Allocated_Area': product_allocated_area,
    'Product_MRP': product_mrp,
    'Store_Id': store_id,
    'Store_Size': store_size,
    'Store_Location_City_Type': store_location_city_type,
    'Store_Type': store_type,
    'Store_Age_Years': store_age_years,
    'Product_Id_Prefix': product_id_prefix,
    'Perishable': perishable
}

# Make prediction when the "Predict" button is clicked
if st.button("Predict Sales", type="primary"):
    response = requests.post(f"{BACKEND_URL}/v1/predict", json=payload)  # Send data to Flask API
    if response.status_code == 200:
        prediction = response.json()["Predicted Product Store Sales Total (in dollars)"]
        st.success(f"Predicted Product Store Sales Total (in dollars): {prediction:.2f}")
    else:
        st.error(f"Unable to connect to the prediction API. Status Code: {response.status_code}")
        st.error(response.text)

# Section for batch prediction
st.subheader("Batch Prediction")

# Allow users to upload a CSV file for batch prediction
uploaded_file = st.file_uploader("Upload CSV file for batch prediction", type=["csv"])

# Make batch prediction when the "Predict Batch" button is clicked
if uploaded_file is not None:
    if st.button("Predict Batch Sales", type="primary"):
        files = {"file": uploaded_file.getvalue()} # Get file content as bytes
        response = requests.post(f"{BACKEND_URL}/v1/predictbatch", files=files)  # Send file to Flask API
        if response.status_code == 200:
            predictions = response.json()
            st.success("Batch predictions completed!")
            st.write(predictions)  # Display the predictions
        else:
            st.error(f"Unable to connect to the prediction API. Status Code: {response.status_code}")
            st.error(response.text)
