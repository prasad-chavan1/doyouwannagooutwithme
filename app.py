import streamlit as st
import pandas as pd
import joblib

# Load the model
model = joblib.load('https://github.com/prasad-chavan1/doyouwannagooutwithme/blob/main/model.pkl')

# Define the categorical columns and their possible values
categorical_features = {
    "Willing_to_Commit": ["mid", "strong"],
    "Relationship_Type": ["online", "offline"],
    "Share_Social_Media_ID_Pass": ["No", "Yes"],
    "Horniness_Level": ["low", "mid", "strong"],
    "Emotional_Involvement": ["low", "mid", "strong"],
    "Appreciate_What": ["mentalHealth", "looks"]
}

# Define the continuous columns
continuous_features = ["Past_Relations", "Breakups", "Flirt_Count", "Dating_Apps_Used", "Time_Given_Hours"]

# Streamlit app
st.title("Play Boy/ Girl Prediction App")

# Input fields for continuous features (integers)
inputs = {}
for feature in continuous_features:
    inputs[feature] = st.number_input(f"Enter {feature}", min_value=0, format="%d")

# Radio buttons for categorical features
for feature, options in categorical_features.items():
    inputs[feature] = st.radio(f"Select {feature}", options)

# Convert inputs to DataFrame
input_df = pd.DataFrame([inputs])

# One-hot encode the categorical features
input_df = pd.get_dummies(input_df, columns=categorical_features.keys())

# Ensure that all the dummy columns are in the same order as in training
model_columns = model.feature_names_in_  # Get feature names from the model
input_df = input_df.reindex(columns=model_columns, fill_value=0)

# Predict
if st.button("Predict"):
    try:
        prediction = model.predict(input_df)[0]
        st.write("Prediction:", "They are Playboy!!" if prediction == 1 else "You're lukcy! You are in right hands :)")
    except ValueError as e:
        st.error(f"Error: {e}")
