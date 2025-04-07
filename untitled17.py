import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="🚗 Car Evaluation Classifier", layout="centered")

# Title and branding
st.title("🚗 Car Evaluation Classifier using Random Forest & Streamlit")
st.markdown("Predict the car condition using Machine Learning based on various features.")
st.markdown("👩‍💻 Made by: **Namu**")
st.write("📤 Upload your `car.csv` file below:")

# File uploader
uploaded_file = st.file_uploader("Upload car.csv", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        # Show data
        st.subheader("📊 Uploaded Dataset Preview")
        st.dataframe(df)

        # Encode categorical columns
        st.subheader("🔁 Label Encoding Features")
        label_encoders = {}
        for column in df.columns:
            if df[column].dtype == 'object':
                le = LabelEncoder()
                df[column] = le.fit_transform(df[column])
                label_encoders[column] = le

        # Splitting
        X = df.drop('class', axis=1)
        y = df['class']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluation
        acc = accuracy_score(y_test, y_pred)
        st.subheader("✅ Model Accuracy")
        st.success(f"Accuracy: {acc:.2f}")

        st.subheader("📋 Classification Report")
        st.text(classification_report(y_test, y_pred))

        st.subheader("📉 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

        st.subheader("🌟 Feature Importance")
        importances = model.feature_importances_
        features = X.columns
        importance_df = pd.DataFrame({"Feature": features, "Importance": importances})
        importance_df = importance_df.sort_values(by="Importance", ascending=False)

        fig2, ax2 = plt.subplots()
        sns.barplot(data=importance_df, x="Importance", y="Feature", ax=ax2)
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"❌ Error reading file: {e}")

else:
    st.warning("👈 Please upload a `car.csv` file to continue.")

st.markdown("---")
st.markdown("Made with ❤️ by **Namu**")
