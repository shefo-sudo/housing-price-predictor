import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --- Page Config ---
st.set_page_config(page_title="Housing Price Predictor", page_icon="🏠", layout="centered")

# --- Custom CSS ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #ffffff;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Header ---
st.title('🏠 Housing Price Prediction')
st.markdown("Predict the market value of a house using Machine Learning.")
st.markdown("--- ")

# --- Load Resources ---
@st.cache_resource
def load_assets():
    model = pickle.load(open('housing_model.pkl', 'rb'))
    model_columns = pickle.load(open('model_columns.pkl', 'rb'))
    return model, model_columns

try:
    model, model_columns = load_assets()
except Exception as e:
    st.error("Error loading model files. Please ensure .pkl files are in the directory.")
    st.stop()

# --- Input Form ---
with st.container():
    st.subheader("Enter Property Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        overall_qual = st.slider('Overall Quality (1-10)', 1, 10, 6)
        total_sf = st.number_input('Total Square Footage', value=2000, step=50)
    
    with col2:
        house_age = st.number_input('House Age (Years)', value=10, min_value=0)
        remodel_age = st.number_input('Years Since Remodel', value=5, min_value=0)

    st.markdown("### Location & Zoning")
    ms_zoning = st.selectbox('Zoning Classification (MS Zoning)', ['RL', 'RM', 'FV', 'RH', 'C (all)'])

# --- Prediction Logic ---
if st.button('Calculate Estimated Price'):
    # Prepare Input DataFrame
    input_df = pd.DataFrame(np.zeros((1, len(model_columns))), columns=model_columns)
    
    # Mapping manual inputs
    mappings = {
        'Overall Qual': overall_qual,
        'Total SF': total_sf,
        'House Age': house_age,
        'Years Since Remodel': remodel_age
    }
    
    for col, val in mappings.items():
        if col in input_df.columns: input_df[col] = val
    
    # Handle Categorical Encoding
    zone_col = f'MS Zoning_{ms_zoning}'
    if zone_col in input_df.columns: input_df[zone_col] = 1

    # Predict
    prediction = model.predict(input_df)[0]
    
    # Display Result
    st.markdown("--- ")
    st.markdown(f"""
        <div class='result-box'>
            <h3>Predicted Market Price</h3>
            <h1 style='color: #2e7d32;'>${prediction:,.2f}</h1>
        </div>
    """, unsafe_allow_html=True)
    st.balloons()

st.markdown("--- ")
st.caption("Note: This model is for educational purposes based on historical data.")
