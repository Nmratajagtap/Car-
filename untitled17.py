import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 🚗 Title and Description
st.title("🚗 Car Evaluation Classifier using Random Forest & Streamlit")
st.write("Predict the car condition using Machine Learning based on various features.")
st.markdown("👩‍💻 **Made by: Namu**")

# 📂 Load dataset from uploaded path
path = "/mnt/data/car.csv"  # Uploaded path
try:
    df = pd.read_csv(path)
    st.success("✅ Dataset loaded successfully!")
    st.dataframe(df.head())
except FileNotFoundError:
    st.error(f"❌ File not found at path: {path}")
    st.stop()

# 🎯 Encode categorical variables
le = LabelEncoder()
for col in df.columns:
    df[col] = le.fit_transform(df[col])

# 🔄 Split data
X = df.drop("class", axis=1)
y = df["class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🌳 Train Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# 📊 Show Accuracy
accuracy = accuracy_score(y_test, y_pred)
st.success(f"✅ Model Accuracy: {accuracy:.5f} ({accuracy:.2%})")

# 🧠 Predict from user input
st.subheader("🔎 Try a prediction:")

user_input = []
for col in X.columns:
    unique_vals = df[col].unique()
    val = st.selectbox(f"Select {col}", options=unique_vals)
    user_input.append(val)

if st.button("Predict"):
    prediction = rf.predict([user_input])[0]
    st.info(f"📈 Predicted Class: {prediction}")

# ❤️ Footer
st.markdown("---")
st.markdown("Made with ❤️ by **Namu**")


else:
    st.warning("⚠️ Please upload a valid `car.csv` file to proceed.")

