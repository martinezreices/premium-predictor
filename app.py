import streamlit as st
import pandas as pd
import pickle
import numpy as np 

st.set_page_config(
    page_title='Insurance Premium Predictor', 
    page_icon='⚕️',
    layout='centered',
    initial_sidebar_state='expanded'
)

# model loading
@st.cache_resource
def load_model(): 
    with open('src/premium_predictor_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# Interface 

st.title('⚕️ Insurance Premium Predictor')
st.write(
    'This application predicts an estimated insurance premium based on individual details'
    'please enter your information in the sidebar to get your estimate'
)

# User Input
st.sidebar.header('Enter Your Details')

# Sidebar Fields
age = st.sidebar.slider('Age', min_value=18, max_value=100, value=30)
sex = st.sidebar.selectbox('Sex',('male', 'female'))

st.sidebar.subheader("Height & Weight")
feet = st.sidebar.number_input("Feet", min_value=1, max_value=8, value=5)
inches = st.sidebar.number_input("Inches", min_value=0, max_value=11, value=10)
weight_lbs = st.sidebar.number_input("Weight (lbs)", min_value=50, max_value=500, value=160)

# --- BMI Calculation ---
# Convert height to total inches
total_height_in = (feet * 12) + inches

# Set a default BMI and calculate only if height is not zero to prevent errors
bmi = 0.0
if total_height_in > 0:
    # BMI formula for Imperial units
    bmi = (weight_lbs / (total_height_in ** 2)) * 703

st.sidebar.metric(label="Calculated BMI", value=f"{bmi:.1f}")
st.sidebar.write("---") # Adds a visual separator

children = st.sidebar.slider('Number of Children', min_value=0, max_value=5, value=0)
smoker = st.sidebar.selectbox('Do you use Tobacco?',('yes', 'no'))



state_to_region_map = {
    'Alabama': 'southeast', 'Alaska': 'northwest', 'Arizona': 'southwest', 'Arkansas': 'southeast',
    'California': 'southwest', 'Colorado': 'northwest', 'Connecticut': 'northeast', 'Delaware': 'northeast',
    'Florida': 'southeast', 'Georgia': 'southeast', 'Hawaii': 'southwest', 'Idaho': 'northwest',
    'Illinois': 'northeast', 'Indiana': 'northeast', 'Iowa': 'northwest', 'Kansas': 'northwest',
    'Kentucky': 'southeast', 'Louisiana': 'southeast', 'Maine': 'northeast', 'Maryland': 'northeast',
    'Massachusetts': 'northeast', 'Michigan': 'northeast', 'Minnesota': 'northwest', 'Mississippi': 'southeast',
    'Missouri': 'northwest', 'Montana': 'northwest', 'Nebraska': 'northwest', 'Nevada': 'southwest',
    'New Hampshire': 'northeast', 'New Jersey': 'northeast', 'New Mexico': 'southwest', 'New York': 'northeast',
    'North Carolina': 'southeast', 'North Dakota': 'northwest', 'Ohio': 'northeast', 'Oklahoma': 'southwest',
    'Oregon': 'northwest', 'Pennsylvania': 'northeast', 'Rhode Island': 'northeast', 'South Carolina': 'southeast',
    'South Dakota': 'northwest', 'Tennessee': 'southeast', 'Texas': 'southwest', 'Utah': 'northwest',
    'Vermont': 'northeast', 'Virginia': 'southeast', 'Washington': 'northwest', 'West Virginia': 'southeast',
    'Wisconsin': 'northwest', 'Wyoming': 'northwest'
}

state_list = sorted(list(state_to_region_map.keys()))
state = st.sidebar.selectbox("State", state_list)
# Perform the lookup to find the region
region = state_to_region_map[state]
# Display the resulting region to the user
st.sidebar.write(f"Model Region: **{region.title()}**")




# Prediction Logic
if st.sidebar.button('Submit'):

    input_data = {
        'age': [age],
        'sex': [sex],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker],
        'region': [region]
    }
    input_df = pd.DataFrame(input_data)

    #Prediction
    prediction = model.predict(input_df)
    predicted_premium = prediction[0]
    monthly_premium = predicted_premium/12

    # Results
    st.success(
        f"""
        **Prediction Results:**
        * **Predicted Annual Premium:** ${predicted_premium:,.2f}
        * **Predicted Monthly Premium:** ${monthly_premium:,.2f}
        """
    )

    st.write(
        'This is an estimated premium. Actual prices may vary based on detailed information and underwriting.'
    )