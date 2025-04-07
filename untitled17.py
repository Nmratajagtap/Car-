# car_data_analysis_app.py

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import statsmodels.api as sm

st.title("🚗 Car Data Analysis App")

uploaded_file = st.file_uploader("Upload your Car Dataset (CSV)", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Data")
    st.dataframe(df.head())

    st.write("### Data Types")
    st.text(df.dtypes)

    st.write("### Missing Values")
    st.write(df.isnull().sum())

    # Mapping categorical columns
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

    # Features and Target
    X = df.drop('class', axis=1)
    y = df['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Logistic Regression
    st.write("## Logistic Regression Model")
    log_model = LogisticRegression()
    log_model.fit(X_train, y_train)
    y_pred = log_model.predict(X_test)

    st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    st.write(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.2f}")
    st.write(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.2f}")

    # Feature Coefficients
    coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': log_model.coef_[0]})
    coef_df['AbsCoef'] = coef_df['Coefficient'].abs()
    coef_df = coef_df.sort_values(by='AbsCoef', ascending=False)

    st.write("### Feature Importance (Logistic Regression Coefficients)")
    st.dataframe(coef_df[['Feature', 'Coefficient']])

    # Model Comparison
    st.write("## Model Comparison")

    models = {
        "KNN": KNeighborsClassifier(),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        st.write(f"### {name}")
        st.text(classification_report(y_test, y_pred))

    # Correlation Heatmap
    st.write("## Correlation Heatmap")
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    st.pyplot(plt)

    # Boxplots
    st.write("## Boxplots of Features vs Class")
    for col in X.columns:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x='class', y=col, data=df)
        plt.title(f'{col} vs Class')
        st.pyplot(plt)



