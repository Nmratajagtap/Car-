@streamlit
import streamlit as st
import pandas as pd
import numpy as np

# Assuming you have 'best_model', 'X_test', 'y_test', and 'mapping_dict' defined from your previous code.
# Replace these with your actual data and model.

# Example mapping_dict (replace with your actual mapping)
mapping_dict = {
    'buying': {'vhigh': 0, 'high': 1, 'med': 2, 'low': 3},
    'maint': {'vhigh': 0, 'high': 1, 'med': 2, 'low': 3},
    'doors': {'2': 0, '3': 1, '4': 2, '5more': 3},
    'persons': {'2': 0, '4': 1, 'more': 2},
    'lug_boot': {'small': 0, 'med': 1, 'big': 2},
    'safety': {'low': 0, 'med': 1, 'high': 2},
    'class': {'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3}
}

# Example best_model and data (REPLACE WITH YOUR ACTUAL MODEL AND DATA)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Sample data (replace with your actual data loading)
data = pd.DataFrame({
    'buying': [0, 1, 2, 3, 0],
    'maint': [0, 1, 2, 3, 0],
    'doors': [0, 1, 2, 3, 0],
    'persons': [0, 1, 2, 0, 1],
    'lug_boot': [0, 1, 2, 0, 1],
    'safety': [0, 1, 2, 0, 1],
    'class': [0, 1, 2, 0, 1]
})

X = data.drop('class', axis=1)
y = data['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
best_model = LogisticRegression(max_iter=1000)
best_model.fit(X_train, y_train)



st.title("Car Evaluation Model")

# Input features
buying = st.selectbox("Buying Price", list(mapping_dict['buying'].keys()))
maint = st.selectbox("Maintenance Price", list(mapping_dict['maint'].keys()))
doors = st.selectbox("Number of Doors", list(mapping_dict['doors'].keys()))
persons = st.selectbox("Number of Persons", list(mapping_dict['persons'].keys()))
lug_boot = st.selectbox("Luggage Boot Size", list(mapping_dict['lug_boot'].keys()))
safety = st.selectbox("Safety", list(mapping_dict['safety'].keys()))


# Create input dataframe
input_data = pd.DataFrame({
    'buying': [mapping_dict['buying'][buying]],
    'maint': [mapping_dict['maint'][maint]],
    'doors': [mapping_dict['doors'][doors]],
    'persons': [mapping_dict['persons'][persons]],
    'lug_boot': [mapping_dict['lug_boot'][lug_boot]],
    'safety': [mapping_dict['safety'][safety]]
})

# Make prediction
if st.button("Predict"):
    prediction = best_model.predict(input_data)[0]
    predicted_class = list(mapping_dict['class'].keys())[int(prediction)]
    st.write(f"Predicted Car Class: {predicted_class}")

    # Display classification report (optional)
    y_pred = best_model.predict(X_test)
    st.text(classification_report(y_test, y_pred))
