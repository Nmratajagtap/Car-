import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# App title
st.title("🚗 Car Evaluation Classifier using Random Forest & Streamlit")
st.markdown("Predict the car condition using Machine Learning based on various features.")
st.markdown("👩‍💻 Made by: **Namu**")

# File uploader
uploaded_file = st.file_uploader("📤 Upload your `car.csv` file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Preview of Uploaded Data")
    st.write(df.head())

    # Encode categorical variables
    label_encoders = {}
    for column in df.columns:
        if df[column].dtype == 'object':
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            label_encoders[column] = le

    # Split into features and target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Random Forest model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Model accuracy
    accuracy = accuracy_score(y_test, y_pred)
    st.success(f"✅ Model Accuracy: {accuracy:.2%}")

    # Classification report
    st.subheader("📄 Classification Report")
    st.text(classification_report(y_test, y_pred))

    # Confusion Matrix
    st.subheader("📉 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    st.pyplot(plt)
#Accuracy
    st.success(f"✅ Model Accuracy: {accuracy:.2%}")


    # Feature Importance
    st.subheader("💡 Feature Importance")
    importances = model.feature_importances_
    features = X.columns
    importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    st.pyplot(plt)

    st.markdown("---")
    st.markdown("✅ **App completed successfully!**")
    st.markdown("❤️ Made with love by **Namu**")

else:
    st.warning("⚠️ Please upload a valid `car.csv` file to proceed.")

