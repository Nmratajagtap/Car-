import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

st.title("🚗 Car Evaluation Classifier using Random Forest & Streamlit")
st.write("Predict the car condition using Machine Learning based on various features.")
st.markdown("👩‍💻 **Made by: Namu**")

# Upload CSV file
uploaded_file = st.file_uploader("📁 Upload your car.csv file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Car Dataset")
    st.dataframe(df)

    # Label Encoding
    le_dict = {}
    for col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le  # Store for later inverse transform

    X = df.drop("class", axis=1)
    y = df["class"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.subheader("🎯 Model Accuracy")
    st.success(f"The Random Forest model accuracy is: **{acc * 100:.2f}%**")

    # Prediction input form
    st.subheader("🔮 Predict Car Evaluation")
    col1, col2 = st.columns(2)

    with col1:
        buying = st.selectbox("Buying Price", le_dict['buying'].classes_)
        maint = st.selectbox("Maintenance Price", le_dict['maint'].classes_)
        doors = st.selectbox("Number of Doors", le_dict['doors'].classes_)
        persons = st.selectbox("Persons Capacity", le_dict['persons'].classes_)

    with col2:
        lug_boot = st.selectbox("Luggage Boot Size", le_dict['lug_boot'].classes_)
        safety = st.selectbox("Safety Level", le_dict['safety'].classes_)

    input_data = [[
        le_dict['buying'].transform([buying])[0],
        le_dict['maint'].transform([maint])[0],
        le_dict['doors'].transform([doors])[0],
        le_dict['persons'].transform([persons])[0],
        le_dict['lug_boot'].transform([lug_boot])[0],
        le_dict['safety'].transform([safety])[0]
    ]]

    prediction = model.predict(input_data)
    predicted_class = le_dict['class'].inverse_transform(prediction)[0]

    st.subheader("🧠 Prediction Result")
    st.success(f"The predicted car evaluation is: **{predicted_class}**")

    st.markdown("---")
    st.markdown("Made with ❤️ by **Namu**")
else:
    st.warning("Please upload a `car.csv` file to continue.")






