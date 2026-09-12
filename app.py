import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load('hcc_voting_model.pkl')
scaler = joblib.load('hcc_scaler.pkl')
features = joblib.load('hcc_features.pkl')

st.title("HCC Patient Survival Prediction Tool")
st.write("Please enter the patient's clinical and laboratory parameters below:")

st.sidebar.header("Patient Parameters")

age = st.sidebar.number_input("Age (years)", min_value=18.0, max_value=100.0, value=35.0)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
symptoms = st.sidebar.selectbox("Symptoms", ["No", "Yes"], index=0)
alcohol = st.sidebar.selectbox("Alcohol", ["No", "Yes"])
smoking = st.sidebar.selectbox("Smoking", ["No", "Yes"])
diabetes = st.sidebar.selectbox("Diabetes", ["No", "Yes"])
obesity = st.sidebar.selectbox("Obesity", ["No", "Yes"])
hepb = st.sidebar.selectbox("Hepatitis B", ["No", "Yes"])
hepc = st.sidebar.selectbox("Hepatitis C", ["No", "Yes"])
cirrhosis = st.sidebar.selectbox("Cirrhosis", ["No", "Yes"])
endemic = st.sidebar.selectbox("Endemic Calcification", ["No", "Yes"])
fatty_liver = st.sidebar.selectbox("Non-alcoholic Steatohepatitis (NASH)", ["No", "Yes"])

total_bil = st.sidebar.number_input("Total Bilirubin (mg/dL)", min_value=0.0, max_value=50.0, value=0.6)
direct_bil = st.sidebar.number_input("Direct Bilirubin (mg/dL)", min_value=0.0, max_value=30.0, value=0.2)
ast = st.sidebar.number_input("AST (U/L)", min_value=0.0, max_value=2000.0, value=25.0)
alt = st.sidebar.number_input("ALT (U/L)", min_value=0.0, max_value=2000.0, value=25.0)
alp = st.sidebar.number_input("ALP (U/L)", min_value=0.0, max_value=2000.0, value=70.0)
afp = st.sidebar.number_input("AFP (ng/mL)", min_value=0.0, max_value=100000.0, value=5.0)
albumin = st.sidebar.number_input("Albumin (g/dL)", min_value=0.0, max_value=10.0, value=4.2)
hemoglobin = st.sidebar.number_input("Hemoglobin (g/dL)", min_value=0.0, max_value=25.0, value=13.5)
platelets = st.sidebar.number_input("Platelets (10^3/uL)", min_value=0.0, max_value=1000.0, value=250.0)
inr = st.sidebar.number_input("INR", min_value=0.0, max_value=10.0, value=0.9)
creatinine = st.sidebar.number_input("Creatinine (mg/dL)", min_value=0.0, max_value=15.0, value=0.7)
ferritin = st.sidebar.number_input("Ferritin (ng/mL)", min_value=0.0, max_value=10000.0, value=80.0)

nodules = st.sidebar.selectbox("Number of Nodules", ["Single", "Multiple"])
major_dim = st.sidebar.number_input("Major Dimension (cm)", min_value=0.0, max_value=30.0, value=1.5)
encap = st.sidebar.selectbox("Encasement / Tumor Encapsulation", ["No", "Yes"])
encephalopathy = st.sidebar.selectbox("Encephalopathy Grade", ["None", "Grade 1-2", "Grade 3-4"])
ascites = st.sidebar.selectbox("Ascites", ["None", "Mild", "Moderate-Severe"])
portal_htn = st.sidebar.selectbox("Portal Hypertension", ["No", "Yes"])
metastasis = st.sidebar.selectbox("Metastasis", ["No", "Yes"])

ast_alt_ratio = ast / alt if alt != 0 else 0.0
albi_score = (np.log10(total_bil * 17.1) * 0.66) + (albumin * -0.085)
fib4_score = (age * ast) / (platelets * 1000) * np.sqrt(alt) if platelets != 0 else 0.0

cp_score = 0
if total_bil > 3.0: cp_score += 3
elif total_bil >= 2.0: cp_score += 2
else: cp_score += 1

if albumin > 3.5: cp_score += 1
elif albumin >= 2.8: cp_score += 2
else: cp_score += 3

if inr > 2.3: cp_score += 3
elif inr >= 1.7: cp_score += 2
else: cp_score += 1

if ascites == "Moderate-Severe": cp_score += 3
elif ascites == "Mild": cp_score += 2
else: cp_score += 1

if encephalopathy == "Grade 3-4": cp_score += 3
elif encephalopathy == "Grade 1-2": cp_score += 2
else: cp_score += 1

raw_data = {
    'Age': age,
    'Gender': 1 if gender == "Male" else 0,
    'Symptoms': 1 if symptoms == "Yes" else 0,
    'Alcohol': 1 if alcohol == "Yes" else 0,
    'Smoking': 1 if smoking == "Yes" else 0,
    'Diabetes': 1 if diabetes == "Yes" else 0,
    'Obesity': 1 if obesity == "Yes" else 0,
    'Hepatitis_B': 1 if hepb == "Yes" else 0,
    'Hepatitis_C': 1 if hepc == "Yes" else 0,
    'Cirrhosis': 1 if cirrhosis == "Yes" else 0,
    'Endemic_Calcification': 1 if endemic == "Yes" else 0,
    'NASH': 1 if fatty_liver == "Yes" else 0,
    'Total_Bilirubin': total_bil,
    'Direct_Bilirubin': direct_bil,
    'AST': ast,
    'ALT': alt,
    'ALP': alp,
    'AFP': afp,
    'Albumin': albumin,
    'Hemoglobin': hemoglobin,
    'Platelets': platelets,
    'INR': inr,
    'Creatinine': creatinine,
    'Ferritin': ferritin,
    'Number_of_Nodules': 1 if nodules == "Single" else 2,
    'Major_Dimension': major_dim,
    'Tumor_Encapsulation': 1 if encap == "Yes" else 0,
    'Encephalopathy_Grade': 0 if encephalopathy == "None" else (1 if encephalopathy == "Grade 1-2" else 2),
    'Ascites': 0 if ascites == "None" else (1 if ascites == "Mild" else 2),
    'Portal_Hypertension': 1 if portal_htn == "Yes" else 0,
    'Metastasis': 1 if metastasis == "Yes" else 0,
    'AST_ALT_Ratio': ast_alt_ratio,
    'ALBI_Score': albi_score,
    'FIB4_Score': fib4_score,
    'Child_Pugh_Score': cp_score
}

input_df = pd.DataFrame([raw_data])

for col in features:
    if col not in input_df.columns:
        input_df[col] = 0.0

input_df = input_df[features]

if st.button("Predict Survival", key="predict_btn"):
    input_df = input_df[features]
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)

    st.subheader("Prediction Result:")
    if prediction[0] == 1:
        st.success("🟢 Prediction: Favorable Outcome / Likely to Survive")
    else:
        st.error("🔴 Prediction: Unfavorable Outcome / High Mortality Risk")