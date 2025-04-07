import streamlit as st
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sidebar
st.sidebar.title("🌸 Iris Classifier")
st.sidebar.markdown("**Made by Namu 😀**")

# Title and info
st.title("Iris Flower Classification App 🌼")
st.markdown("""
This app uses **Multinomial Logistic Regression (Softmax)** trained on the Iris dataset  
to classify iris flowers into three species: **Setosa, Versicolor, and Virginica.**  
**Made by Namu 😀**
""")

# Load dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200)
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.subheader("Model Accuracy")
st.write(f"✅ Accuracy: **{accuracy * 100:.2f}%**")

# Input sliders
st.subheader("Predict Flower Type")
sepal_length = st.slider("Sepal Length (cm)", float(X[:, 0].min()), float(X[:, 0].max()))
sepal_width = st.slider("Sepal Width (cm)", float(X[:, 1].min()), float(X[:, 1].max()))
petal_length = st.slider("Petal Length (cm)", float(X[:, 2].min()), float(X[:, 2].max()))
petal_width = st.slider("Petal Width (cm)", float(X[:, 3].min()), float(X[:, 3].max()))

# Prediction button
if st.button("🔍 Predict Flower Species"):
    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(input_data)[0]
    prediction_name = target_names[prediction]
    st.success(f"🌸 The predicted Iris species is: **{prediction_name}**")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by **Namu**")
