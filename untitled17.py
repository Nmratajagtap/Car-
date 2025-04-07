import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ------------------------
# App Info Section
# ------------------------
st.title("🌸 Iris Flower Classification App")
st.markdown("""
### About the App  
This app uses **Multinomial Logistic Regression (Softmax)** trained on the Iris dataset to classify iris flowers into three species:  
- *Setosa*  
- *Versicolor*  
- *Virginica*  

Made by **Namu** 😀
""")

# ------------------------
# Load and display data
# ------------------------
df = sns.load_dataset('iris')
st.subheader("📊 Iris Dataset Preview")
st.dataframe(df.head())

# ------------------------
# Model training
# ------------------------
X = df.drop('species', axis=1)
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
st.subheader("✅ Model Accuracy")
st.write(f"Accuracy: **{acc:.2f}**")

# ------------------------
# Prediction
# ------------------------
st.subheader("🌼 Predict Species")
sepal_length = st.slider('Sepal Length', float(df['sepal_length'].min()), float(df['sepal_length'].max()))
sepal_width = st.slider('Sepal Width', float(df['sepal_width'].min()), float(df['sepal_width'].max()))
petal_length = st.slider('Petal Length', float(df['petal_length'].min()), float(df['petal_length'].max()))
petal_width = st.slider('Petal Width', float(df['petal_width'].min()), float(df['petal_width'].max()))

if st.button("Predict"):
    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    st.success(f"The predicted species is: **{prediction[0].capitalize()}**")
