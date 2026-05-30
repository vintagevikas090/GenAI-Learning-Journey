from tensorflow.keras.models import load_model
import pickle
import pandas as pd
import streamlit as st
import os, time

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="centered")

# LOAD MODELS AND ENCODERS
model_folder_path = r'D:\AI\genai-learning-journey\02_deep_learning_for_NLP\ANN Projects\Customer Churn Classification\models'

@st.cache_resource
def load_models_and_encoders(model_folder_path):
    model = load_model(os.path.join(model_folder_path, "ann_classification_model.h5"))

    with open(os.path.join(model_folder_path, "geography_encoder_model.pkl"), "rb") as file:
        label_encoder_geo = pickle.load(file)

    with open(os.path.join(model_folder_path, "gender_encoder_model.pkl"), "rb") as file:
        label_encoder_gender = pickle.load(file)

    with open(os.path.join(model_folder_path, "Standard_Scaler_model.pkl"), "rb") as file:
        scaler = pickle.load(file)
    
    return model, label_encoder_geo, label_encoder_gender, scaler

with st.spinner(text = 'Loading Models'):
    model, label_encoder_geo, label_encoder_gender, scaler = load_models_and_encoders(model_folder_path)
    st.success('Models Loaded Successfully')


# MAIN UI
st.title("📊 Customer Churn Prediction")
st.write("Enter customer details to predict churn probability.")


# INPUT FORM
with st.form("input_data", clear_on_submit=True):

    geography = st.selectbox("Geography", label_encoder_geo.categories_[0])
    gender = st.selectbox("Gender", label_encoder_gender.classes_)

    age = st.slider("Age", 18, 100, 25)

    credit_score = st.number_input("Credit Score", min_value=100, max_value=900, value=650, step = 10)
    balance = st.number_input("Balance", min_value=0.0, value=0.0, step=100.0)
    estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=10000.0, step=1000.0)

    tenure = st.slider("Tenure", 0, 10, 3)
    num_of_products = st.slider("Number of Products", 1, 4, 1)

    has_cr_card = st.selectbox("Has Credit Card", ["No", "Yes"])
    is_active_member = st.selectbox("Is Active Member", ["No", "Yes"])

    submit = st.form_submit_button("Predict Churn")


# PREDICTION
if submit:
    with st.spinner(text = 'Wait for some time'):
        time.sleep(5)
        input_data = {
            "CreditScore": credit_score,
            "Gender": label_encoder_gender.transform([gender])[0],
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": num_of_products,
            "HasCrCard": int(has_cr_card == "Yes"),
            "IsActiveMember": int(is_active_member == "Yes"),
            "EstimatedSalary": estimated_salary
        }

        input_df = pd.DataFrame([input_data])

        geo_encoded = label_encoder_geo.transform([[geography]]).toarray()
        geo_encoded_df = pd.DataFrame(geo_encoded, columns=label_encoder_geo.get_feature_names_out())

        input_df = pd.concat([input_df.reset_index(drop=True), geo_encoded_df], axis=1)

        input_df_scaled = scaler.transform(input_df)

        prediction_proba = float(model.predict(input_df_scaled, verbose=0)[0][0])

    st.divider()

    st.metric("Churn Probability", f"{prediction_proba:.2%}")

    if prediction_proba > 0.5:
        st.error("⚠️ Customer is likely to churn")
    else:
        st.success("✅ Customer is likely to stay")