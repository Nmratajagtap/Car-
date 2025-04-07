# Streamlit app
st.title("Car Evaluation Model")

# Input features (replace with actual input fields)
buying = st.selectbox("Buying Price", list(mapping_dict['buying'].keys()))
maint = st.selectbox("Maintenance Price", list(mapping_dict['maint'].keys()))
doors = st.selectbox("Number of Doors", list(mapping_dict['doors'].keys()))
persons = st.selectbox("Number of Persons", list(mapping_dict['persons'].keys()))
lug_boot = st.selectbox("Luggage Boot Size", list(mapping_dict['lug_boot'].keys()))
safety = st.selectbox("Safety", list(mapping_dict['safety'].keys()))


# Create input dataframe
input_data = pd.DataFrame({
    'buying': [mapping_dict['buying'][buying]],
    'maint': [mapping_dict['maint'][maint]],
    'doors': [mapping_dict['doors'][doors]],
    'persons': [mapping_dict['persons'][persons]],
    'lug_boot': [mapping_dict['lug_boot'][lug_boot]],
    'safety': [mapping_dict['safety'][safety]]
})

# Make prediction
if st.button("Predict"):
    prediction = best_model.predict(input_data)[0]
    st.write(f"Predicted Car Class: {list(mapping_dict['class'].keys())[int(prediction)]}")

    # Display classification report (optional)
    y_pred = best_model.predict(X_test)
    st.text(classification_report(y_test, y_pred))
