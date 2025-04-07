# car_app_path.py
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 🎉 Title and Credits
st.title("🚗 Car Evaluation Classifier using Random Forest & Streamlit")
st.markdown("Predict the **car condition** using Machine Learning based on various features.")
st.markdown("#### 👩‍💻 Made by: Namu")
uploaded_file = st.file_uploader("Upload your car.csv file", type=['csv'])

uploaded_file = st.file_uploader("Upload car.csv", type=['csv'])

try:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv(r"C:\Users\Admin\Downloads\car.csv")  # Fallback for local run
except FileNotFoundError:
    st.error("❌ File not found. Please check the path or upload the file manually.")


    # Mapping categorical values
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

    X = df.drop('class', axis=1)
    y = df['class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # 🚘 Prediction Input
    st.subheader("🔍 Predict Car Condition")
    buying = st.selectbox("Buying Price", list(mapping_dict['buying'].keys()))
    maint = st.selectbox("Maintenance Price", list(mapping_dict['maint'].keys()))
    doors = st.selectbox("Number of Doors", list(mapping_dict['doors'].keys()))
    persons = st.selectbox("Seating Capacity", list(mapping_dict['persons'].keys()))
    lug_boot = st.selectbox("Luggage Boot Size", list(mapping_dict['lug_boot'].keys()))
    safety = st.selectbox("Safety Level", list(mapping_dict['safety'].keys()))

    input_data = pd.DataFrame({
        'buying': [mapping_dict['buying'][buying]],
        'maint': [mapping_dict['maint'][maint]],
        'doors': [mapping_dict['doors'][doors]],
        'persons': [mapping_dict['persons'][persons]],
        'lug_boot': [mapping_dict['lug_boot'][lug_boot]],
        'safety': [mapping_dict['safety'][safety]]
    })

    if st.button("Predict"):
        pred = model.predict(input_data)[0]
        class_names = {v: k for k, v in mapping_dict['class'].items()}
        st.success(f"🚘 Predicted Car Class: **{class_names[pred]}**")

        # 📋 Evaluation
        y_pred = model.predict(X_test)
        st.subheader("📋 Classification Report")
        st.text(classification_report(y_test, y_pred))

except FileNotFoundError:
    st.error("❌ File not found. Please check the path: " + path)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by **Namu**")

