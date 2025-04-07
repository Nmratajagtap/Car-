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

try:
    # 🔽 Read the dataset from a fixed path
    df = pd.read_csv(r'/content/car.csv')
    st.success("✅ File loaded successfully from /content/car.csv")
    st.dataframe(df.head())

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
        unique_vals = sorted(df[col].unique())
        val = st.selectbox(f"Select value for {col}", options=unique_vals)
        user_input.append(val)

    if st.button("Predict"):
        prediction = rf.predict([user_input])[0]
        st.info(f"📈 Predicted Class: {prediction}")
    
    # Footer
    st.markdown("---")
    st.markdown("Made with ❤️ by **Namu**")

except FileNotFoundError:
    st.error("❌ File not found. Please make sure the path '/content/car.csv' is correct.")




