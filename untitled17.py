import streamlit as st
import pandas as pd
pip install seaborn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Car Data Analysis App", layout="wide")
st.title("🚗 Car Data Analysis App")

uploaded_file = st.file_uploader("📂 Upload your Car Dataset (CSV)", type=['csv'], key="car_file")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("🔍 Data Preview")
    st.dataframe(df.head())

    st.subheader("🧾 Data Types and Null Check")
    st.write(df.dtypes)
    st.write("Missing values:")
    st.write(df.isnull().sum())

    # Encode Categorical Columns
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

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Logistic Regression
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.subheader("📊 Logistic Regression Metrics")
    st.write(f"**Accuracy**: {accuracy_score(y_test, y_pred):.2f}")
    st.write(f"**Precision**: {precision_score(y_test, y_pred, average='weighted'):.2f}")
    st.write(f"**Recall**: {recall_score(y_test, y_pred, average='weighted'):.2f}")

    st.subheader("📈 Feature Importance (Logistic Regression Coefficients)")
    coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_[0]})
    coef_df['Abs'] = coef_df['Coefficient'].abs()
    st.dataframe(coef_df.sort_values(by='Abs', ascending=False))

    st.subheader("🧠 Model Comparison")
    models = {
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42)
    }

    for name, clf in models.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        st.markdown(f"**{name}**")
        st.text(classification_report(y_test, y_pred))

    st.subheader("📌 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    st.subheader("📦 Boxplots for Features vs Class")
    for col in X.columns:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.boxplot(x='class', y=col, data=df, ax=ax2)
        st.pyplot(fig2)
else:
    st.warning("Please upload a CSV file to continue.")
