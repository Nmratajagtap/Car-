import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 🚗 App Title
st.set_page_config(page_title="Car Evaluation Classifier")
st.title("🚗 Car Evaluation Classifier using Random Forest & Streamlit")
st.write("Predict the car condition using Machine Learning based on various features.")
st.markdown("👩‍💻 **Made by: Namu**")

# ✅ Read dataset directly (no uploader!)
df = pd.read_csv("car.csv")

# 🔍 Dataset Preview
st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# 🔄 Encode categorical features
df_encoded = df.apply(lambda col: pd.factorize(col)[0])

# 🧪 Train-test split
X = df_encoded.iloc[:, :-1]
y = df_encoded.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🌲 Train Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 🎯 Show accuracy
accuracy = model.score(X_test, y_test)
st.success(f"🎯 Model Accuracy: {accuracy * 100:.2f}%")

# 🧪 Prediction Section
st.subheader("🔍 Predict Car Condition")

# User input widgets
input_data = []
for col in df.columns[:-1]:
    value = st.selectbox(f"Select value for **{col}**", df[col].unique())
    input_data.append(value)

# Encode user input
input_encoded = [
    pd.Series(df[col].unique()).tolist().index(val)
    for col, val in zip(df.columns[:-1], input_data)
]

# Make prediction
prediction = model.predict([input_encoded])[0]

# Decode prediction
decoded = df[df.columns[-1]].unique()[prediction]
st.success(f"✅ Predicted Car Condition: **{decoded}**")

# Footer
st.markdown("---")
st.markdown("❤️ Made with love by **Namu**")



