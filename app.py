import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- PAGE SETUP ---
st.set_page_config(page_title="Sales Predictor AI", layout="centered")
st.title("📊 AI Sales Forecasting Tool")
st.markdown("Enter product details below to predict monthly sales volume.")

# --- LOAD MODEL & FEATURES ---
@st.cache_resource
def load_assets():
    model = joblib.load('sales_model.pkl')
    features = joblib.load('feature_names.pkl')
    return model, features

try:
    model, model_features = load_assets()
except:
    st.error("Model files not found. Please run the 'joblib.dump' code in your notebook first!")
    st.stop()

# --- USER INPUTS ---
st.sidebar.header("Product Specifications")
price = st.sidebar.number_input("Unit Price ($)", min_value=1.0, max_value=500.0, value=25.0)
rating = st.sidebar.slider("Customer Rating", 1.0, 5.0, 4.0)
stock_group = st.sidebar.radio("Stock Category", ["Limited Quantity", "Vast Quantity"])

# --- PREDICTION LOGIC ---
if st.button("Generate Forecast"):
    # 1. Prepare a base dataframe with zeros for all features
    input_data = pd.DataFrame(0, index=[0], columns=model_features)
    
    # 2. Fill in the basic numerical values
    # Note: Use the exact names your model expects from Step 3
    if 'price' in input_data.columns: input_data['price'] = price
    if 'rating' in input_data.columns: input_data['rating'] = rating
    if 'group' in input_data.columns: 
        input_data['group'] = 1 if stock_group == "Limited Quantity" else 0
    
    # 3. Predict (Remember to reverse the log if you used log_sales)
    prediction_log = model.predict(input_data)
    final_prediction = np.expm1(prediction_log)[0] # Back to normal units
    
    # --- DISPLAY RESULTS ---
    st.success(f"### Predicted Sales: {int(final_prediction)} units / month")
    
    # Show confidence context
    st.metric(label="Inventory Impact", value=stock_group, delta="Scarcity Effect Active" if "Limited" in stock_group else None)

st.divider()
st.info("💡 **Data Scientist Note:** This model uses a Random Forest Regressor. Feature importance shows that Price and Stock Status are the strongest predictors for this dataset.")
