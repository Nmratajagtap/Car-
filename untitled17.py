import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Title and description
st.title("🚗 Car Evaluation Classifier using Random Forest & Streamlit")
st.write("Predict the car condition using Machine Learning based on various features.")
st.markdown("👩‍💻 **Made by: Namu**")

# Load the dataset from uploaded path
df = pd.read_csv('/mnt/data/car.csv')

# Show dataset
st.subheader("📊 Car Dataset")
st.dataframe(df)

# Encode categorical variables
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in df.columns:
    df[col] = le.fit_transform(df[col])

# Split the data
X = df.drop('class', axis=1)
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict and calculate accuracy
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# Show accuracy
st.subheader("🎯 Model Accuracy")
st.success(f"The Random Forest model accuracy is: **{acc * 100:.2f}%**")

# Optional: User input prediction
st.subheader("🔍 Try Your Own Prediction")
buying = st.selectbox("Buying", df.columns[0:1])
maint = st.selectbox("Maint", df.columns[1:2])
doors = st.selectbox("Doors", df.columns[2:3])
persons = st.selectbox("Persons", df.columns[3:4])
lug_boot = st.selectbox("Lug Boot", df.columns[4:5])
safety = st.selectbox("Safety", df.columns[5:6])

# Prepare input
input_data = pd.DataFrame([[buying, maint, doors, persons, lug_boot, safety]],
                          columns=X.columns)

# Encode input
for col in input_data.columns:
    input_data[col] = le.fit(df[col]).transform(input_data[col])

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)
    st.success(f"✅ The predicted car condition is: **{le.inverse_transform(prediction)[0]}**")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by **Namu**")





