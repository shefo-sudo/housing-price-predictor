import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.title('Housing Price Prediction App')

# Load model and columns
model = pickle.load(open('housing_model.pkl', 'rb'))
model_columns = pickle.load(open('model_columns.pkl', 'rb'))

st.write("Enter house details to predict the price:")

# Example inputs (simplified for demo)
overall_qual = st.slider('Overall Quality', 1, 10, 5)
total_sf = st.number_input('Total Square Footage', value=2000)
house_age = st.number_input('House Age', value=10)

if st.button('Predict'):
    # Create a template dataframe with zeros
    input_df = pd.DataFrame(np.zeros((1, len(model_columns))), columns=model_columns)
    
    # Fill the inputs we have
    if 'Overall Qual' in input_df.columns: input_df['Overall Qual'] = overall_qual
    if 'Total SF' in input_df.columns: input_df['Total SF'] = total_sf
    if 'House Age' in input_df.columns: input_df['House Age'] = house_age
    
    prediction = model.predict(input_df)
    st.success(f'Predicted Sale Price: ${prediction[0]:,.2f}')
