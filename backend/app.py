from pathlib import Path

import numpy as np
import joblib  # For loading the serialized model
import pandas as pd  # For data manipulation
from flask import Flask, request, jsonify  # For creating the Flask API

# Initialize the Flask application
superkart_sales_predictor_api = Flask("SuperKart Sales Predictor")

# Load the trained machine learning model stored beside this application.
model = joblib.load(Path(__file__).resolve().parent / "superkart_model.joblib")

# Helper function for feature engineering (consistent with notebook)
def get_perishable_category(product_type_val):
    perishable_types = ['Dairy', 'Fruits and Vegetables', 'Meat', 'Seafood', 'Frozen Foods']
    return 1 if product_type_val in perishable_types else 0

# Define a route for the home page (GET request)
@superkart_sales_predictor_api.get('/')
def home():
    """
    This function handles GET requests to the root URL ('/') of the API.
    It returns a simple welcome message.
    """
    return "Welcome to the SuperKart Sales Prediction API!"

# Define an endpoint for single product prediction (POST request)
@superkart_sales_predictor_api.post('/v1/predict')
def predict_sales():
    """
    This function handles POST requests to the '/v1/predict' endpoint.
    It expects a JSON payload containing raw product and store details and returns
    the predicted sales as a JSON response.
    """
    product_data = request.get_json(silent=True)

    # Define raw required fields that the API client should provide
    required_raw_fields = [
        'Product_Id', 'Product_Weight', 'Product_Sugar_Content', 'Product_Allocated_Area',
        'Product_Type', 'Product_MRP', 'Store_Id', 'Store_Establishment_Year',
        'Store_Size', 'Store_Location_City_Type', 'Store_Type'
    ]

    missing_fields = [
        field for field in required_raw_fields
        if not isinstance(product_data, dict) or field not in product_data
    ]
    if missing_fields:
        return jsonify({
            'error': 'Missing required fields in input payload',
            'fields': missing_fields
        }), 400

    # --- Feature Engineering within the API ---
    # These transformations mirror those performed in the notebook

    # 1. Replace 'reg' with 'Regular' in Product_Sugar_Content
    processed_product_sugar_content = product_data['Product_Sugar_Content'].replace('reg', 'Regular')

    # 2. Calculate Store_Age_Years
    current_year = 2009 # Keep consistent with notebook's training logic
    store_age_years = current_year - product_data['Store_Establishment_Year']

    # 3. Extract Product_Id_Prefix
    product_id_prefix = product_data['Product_Id'][:2]

    # 4. Determine Perishable status
    perishable = get_perishable_category(product_data['Product_Type'])

    # --- End Feature Engineering ---

    # Construct the final sample with engineered features, matching model's expected input
    sample = {
        'Product_Weight': product_data['Product_Weight'],
        'Product_Sugar_Content': processed_product_sugar_content,
        'Product_Allocated_Area': product_data['Product_Allocated_Area'],
        'Product_MRP': product_data['Product_MRP'],
        'Store_Id': product_data['Store_Id'],
        'Store_Size': product_data['Store_Size'],
        'Store_Location_City_Type': product_data['Store_Location_City_Type'],
        'Store_Type': product_data['Store_Type'],
        'Store_Age_Years': store_age_years,
        'Product_Id_Prefix': product_id_prefix,
        'Perishable': perishable
    }

    # Convert the extracted data into a Pandas DataFrame
    input_data = pd.DataFrame([sample])

    # Make prediction
    predicted_sales = model.predict(input_data)[0]

    # Convert predicted_sales to Python float and round it
    predicted_sales = round(float(predicted_sales), 2)

    # Return the predicted sales
    return jsonify({'Predicted Product Store Sales Total (in dollars)': predicted_sales})


# Define an endpoint for batch prediction (POST request)
@superkart_sales_predictor_api.post('/v1/predictbatch')
def predict_sales_batch():
    """
    This function handles POST requests to the '/v1/predictbatch' endpoint.
    It expects a CSV file containing raw product details for multiple products
    and returns the predicted sales as a list in the JSON response.
    """
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file provided'}), 400

    raw_input_df = pd.read_csv(file)

    # Define raw required fields that the API client should provide in the CSV
    required_raw_fields_for_batch = [
        'Product_Id', 'Product_Weight', 'Product_Sugar_Content', 'Product_Allocated_Area',
        'Product_Type', 'Product_MRP', 'Store_Id', 'Store_Establishment_Year',
        'Store_Size', 'Store_Location_City_Type', 'Store_Type'
    ]
    missing_cols = [col for col in required_raw_fields_for_batch if col not in raw_input_df.columns]
    if missing_cols:
        return jsonify({
            'error': 'Missing columns in batch CSV',
            'missing_columns': missing_cols
        }), 400

    # --- Feature Engineering for batch data ---
    processed_df = raw_input_df.copy()

    # 1. Replace 'reg' with 'Regular' in Product_Sugar_Content
    processed_df['Product_Sugar_Content'] = processed_df['Product_Sugar_Content'].replace('reg', 'Regular')

    # 2. Calculate Store_Age_Years
    current_year = 2009 # Keep consistent with notebook's training logic
    processed_df['Store_Age_Years'] = current_year - processed_df['Store_Establishment_Year']

    # 3. Extract Product_Id_Prefix
    processed_df['Product_Id_Prefix'] = processed_df['Product_Id'].apply(lambda x: x[:2])

    # 4. Determine Perishable status
    processed_df['Perishable'] = processed_df['Product_Type'].apply(get_perishable_category)
    # --- End Feature Engineering ---

    # Select only the features that the model expects for prediction
    features_for_model = [
        'Product_Weight', 'Product_Sugar_Content', 'Product_Allocated_Area',
        'Product_MRP', 'Store_Id', 'Store_Size',
        'Store_Location_City_Type', 'Store_Type', 'Store_Age_Years',
        'Product_Id_Prefix', 'Perishable'
    ]
    input_data_for_model = processed_df[features_for_model]

    # Make predictions for all products in the DataFrame
    predicted_sales = model.predict(input_data_for_model).tolist()

    # Round the predicted sales values
    predicted_sales = [round(float(sales), 2) for sales in predicted_sales]

    # Return the predictions list as a JSON response
    return jsonify({'Predicted Product Store Sales Total (in dollars)': predicted_sales})

# Run the Flask application in debug mode if this script is executed directly
if __name__ == '__main__':
    superkart_sales_predictor_api.run(debug=True)
