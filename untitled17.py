import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="🚗 Car Evaluation Classifier", layout="centered")

st.title("🚗 Car Evaluation Classifier using Random Forest & Streamlit")
st.write("Predict the car condition using Machine Learning based on various features.")
st.markdown("**👩‍💻 Made by: Namu**")

# Load dataset
df = pd.read_csv("/mnt/data/car.csv")

# Encode categorical variables
le_dict = {}
for column in df.columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    le_dict[column] = le

# Split data
X = df.drop('class', axis=1)
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.success(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

st.subheader("🔍 Predict Car Condition")
buying = st.selectbox("Buying", le_dict['buying'].classes_)
maint = st.selectbox("Maintenance", le_dict['maint'].classes_)
doors = st.selectbox("Doors", le_dict['doors'].classes_)
persons = st.selectbox("Persons", le_dict['persons'].classes_)
lug_boot = st.selectbox("Lug Boot", le_dict['lug_boot'].classes_)
safety = st.selectbox("Safety", le_dict['safety'].classes_)

if st.button("Predict"):
    input_data = [[
        le_dict['buying'].transform([buying])[0],
        le_dict['maint'].transform([maint])[0],
        le_dict['doors'].transform([doors])[0],
        le_dict['persons'].transform([persons])[0],
        le_dict['lug_boot'].transform([lug_boot])[0],
        le_dict['safety'].transform([safety])[0]
    ]]

    input_df = pd.DataFrame(input_data, columns=X.columns)
    prediction = model.predict(input_df)
    predicted_class = le_dict['class'].inverse_transform(prediction)[0]

    st.success(f"Predicted Car Condition: {predicted_class}")

st.markdown("---")
st.caption("Made with ❤️ by Namu")





