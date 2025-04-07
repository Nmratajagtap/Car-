# car_eval_app.py

import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Title and subtitle
st.title("🚗 Car Evaluation Classifier using Random Forest & Streamlit")
st.markdown("Predict the car condition using machine learning based on various features.")

# Load dataset
df = pd.read_csv('car.csv')

# Mapping categorical features to numerical
mapping_dict = {
    'buying': {'vhigh': 0, 'high': 1, 'med': 2, 'low': 3},
    'doors': {'2': 0, '3': 1, '4': 2, '5more': 3},
    'lug_boot': {'small': 0, 'med': 1, 'big': 2},
    'class': {'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3},
    'maint': {'vhigh': 0, 'high': 1, 'med': 2, 'low': 3},
    'persons': {'2': 0, '4': 1, 'more': 2},
    'safety': {'low': 0, 'med': 1, 'high': 2}
}

for col, mapping in mapping_dict.items():
    df[col] = df[col].map(mapping).fillna(-1)

# Split data
X = df.drop('class', axis=1)
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
best_model = RandomForestClassifier(random_state=42)
best_model.fit(X_train, y_train)

# User input interface
st.header("📥 Enter Car Details to Predict Class")

buying = st.selectbox("Buying Price", list(mapping_dict['buying'].keys()))
maint = st.selectbox("Maintenance Price", list(mapping_dict['maint'].keys()))
doors = st.selectbox("Number of Doors", list(mapping_dict['doors'].keys()))
persons = st.selectbox("Number of Persons", list(mapping_dict['persons'].keys()))
lug_boot = st.selectbox("Luggage Boot Size", list(mapping_dict['lug_boot'].keys()))
safety = st.selectbox("Safety", list(mapping_dict['safety'].keys()))

# Prediction
if st.button("Predict"):
    input_data = pd.DataFrame({
        'buying': [mapping_dict['buying'][buying]],
        'maint': [mapping_dict['maint'][maint]],
        'doors': [mapping_dict['doors'][doors]],
        'persons': [mapping_dict['persons'][persons]],
        'lug_boot': [mapping_dict['lug_boot'][lug_boot]],
        'safety': [mapping_dict['safety'][safety]]
    })

    prediction = best_model.predict(input_data)[0]
    class_reverse_mapping = {v: k for k, v in mapping_dict['class'].items()}
    st.success(f"✅ Predicted Car Class: **{class_reverse_mapping[prediction]}**")

    # Optional: show performance of model
    y_pred = best_model.predict(X_test)
    st.subheader("📊 Model Performance on Test Set")
    st.text(classification_report(y_test, y_pred))

# Footer
st.markdown("---")
st.markdown("#### 👩‍💻 Made with ❤️ by **Namu**")

