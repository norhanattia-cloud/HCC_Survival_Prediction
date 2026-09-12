import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

model = joblib.load('hcc_voting_model.pkl')

try:
    scaler = joblib.load('hcc_scaler.pkl')
    # اختبار لو الscaler متوقع عدد مختلف
    if hasattr(scaler, "n_features_in_") and scaler.n_features_in_ != 46:
        scaler = StandardScaler()
except:
    scaler = StandardScaler()

features = joblib.load('hcc_features.pkl')

st.title("HCC Patient Survival Prediction Tool")
st.sidebar.header("Patient Clinical Data Input")

gender_val = st.sidebar.selectbox("Gender", ["Female", "Male"])
gender = 1.0 if gender_val == "Male" else 0.0
age = st.sidebar.number_input("Age (Years)", value=50.0)
ps = st.sidebar.selectbox("Performance Status (PS)", [0.0, 1.0, 2.0, 3.0, 4.0])

hcv_val = st.sidebar.selectbox("HCV Ab", ["Negative", "Positive"])
hcv = 1.0 if hcv_val == "Positive" else 0.0

hbsag_val = st.sidebar.selectbox("HBsAg", ["Negative", "Positive"])
hbsag = 1.0 if hbsag_val == "Positive" else 0.0

hbeag_val = st.sidebar.selectbox("HBeAg", ["Negative", "Positive"])
hbeag = 1.0 if hbeag_val == "Positive" else 0.0

hbcab_val = st.sidebar.selectbox("HBcAb", ["Negative", "Positive"])
hbcab = 1.0 if hbcab_val == "Positive" else 0.0

cirrhosis_val = st.sidebar.selectbox("Cirrhosis", ["No", "Yes"])
cirrhosis = 1.0 if cirrhosis_val == "Yes" else 0.0

alcohol_val = st.sidebar.selectbox("Alcohol Consumption", ["No", "Yes"])
alcohol = 1.0 if alcohol_val == "Yes" else 0.0
alcohol_grams = st.sidebar.number_input("Alcohol Grams/day", value=0.0) if alcohol == 1.0 else 0.0

smoking_val = st.sidebar.selectbox("Smoking", ["No", "Yes"])
smoking = 1.0 if smoking_val == "Yes" else 0.0
packs_year = st.sidebar.number_input("Packs/Year", value=0.0) if smoking == 1.0 else 0.0

symptoms_val = st.sidebar.selectbox("Symptoms", ["Absent", "Present"])
symptoms = 1.0 if symptoms_val == "Present" else 0.0

endemic_val = st.sidebar.selectbox("Endemic Country", ["No", "Yes"])
endemic = 1.0 if endemic_val == "Yes" else 0.0

diabetes = 1.0 if st.sidebar.selectbox("Diabetes", ["No", "Yes"]) == "Yes" else 0.0
obesity = 1.0 if st.sidebar.selectbox("Obesity", ["No", "Yes"]) == "Yes" else 0.0
htn = 1.0 if st.sidebar.selectbox("Arterial Hypertension", ["No", "Yes"]) == "Yes" else 0.0
renal = 1.0 if st.sidebar.selectbox("Chronic Renal Insufficiency", ["No", "Yes"]) == "Yes" else 0.0
hiv = 1.0 if st.sidebar.selectbox("HIV", ["No", "Yes"]) == "Yes" else 0.0
nash = 1.0 if st.sidebar.selectbox("NASH", ["No", "Yes"]) == "Yes" else 0.0
hemochromatosis = 1.0 if st.sidebar.selectbox("Hemochromatosis", ["No", "Yes"]) == "Yes" else 0.0

comorbidities_count = diabetes + obesity + htn + renal + hiv + nash + hemochromatosis

total_bil = st.sidebar.number_input("Total Bilirubin (mg/dL)", value=1.0)
direct_bil = st.sidebar.number_input("Direct Bilirubin", value=0.3)
albumin = st.sidebar.number_input("Albumin (g/dL)", value=3.5)
alt = st.sidebar.number_input("ALT (U/L)", value=40.0)
ast = st.sidebar.number_input("AST (U/L)", value=40.0)
alp = st.sidebar.number_input("ALP (U/L)", value=100.0)
ggt = st.sidebar.number_input("GGT (U/L)", value=50.0)
total_protein = st.sidebar.number_input("Total Protein", value=7.0)
platelets = st.sidebar.number_input("Platelets (10^3/uL)", value=150.0)
leucocytes = st.sidebar.number_input("Leucocytes", value=5.0)
hemoglobin = st.sidebar.number_input("Hemoglobin (g/dL)", value=12.0)
mcv = st.sidebar.number_input("MCV", value=90.0)
inr = st.sidebar.number_input("INR", value=1.0)
afp = st.sidebar.number_input("AFP (ng/mL)", value=10.0)
creatinine = st.sidebar.number_input("Creatinine (mg/dL)", value=1.1)
iron = st.sidebar.number_input("Iron", value=80.0)
ferritin = st.sidebar.number_input("Ferritin", value=200.0)
oxygen_sat = st.sidebar.number_input("Oxygen Saturation", value=30.0)

nodule = st.sidebar.number_input("Number of Nodules", value=1.0)
major_dim = st.sidebar.number_input("Major Dimension (cm)", value=3.0)
ascites = st.sidebar.selectbox("Ascites Grade", [0.0, 1.0, 2.0, 3.0])
encephalopathy = st.sidebar.selectbox("Encephalopathy Grade", [0.0, 1.0, 2.0, 3.0])
varices = 1.0 if st.sidebar.selectbox("Varices", ["No", "Yes"]) == "Yes" else 0.0
splenomegaly = 1.0 if st.sidebar.selectbox("Splenomegaly", ["No", "Yes"]) == "Yes" else 0.0
pht = 1.0 if st.sidebar.selectbox("Portal Hypertension (PHT)", ["No", "Yes"]) == "Yes" else 0.0
pvt = 1.0 if st.sidebar.selectbox("Portal Vein Thrombosis (PVT/PVTT)", ["No", "Yes"]) == "Yes" else 0.0
metastasis = 1.0 if st.sidebar.selectbox("Metastasis", ["No", "Yes"]) == "Yes" else 0.0
hallmark = 1.0 if st.sidebar.selectbox("Hallmark", ["No", "Yes"]) == "Yes" else 0.0

ast_alt_ratio = ast / alt if alt > 0 else 0.0
albi_score = (np.log10(total_bil * 17.1) * 0.66) - ((albumin * 10) * 0.085)
fib4_score = (age * ast) / ((platelets / 1000) * np.sqrt(alt)) if alt > 0 else 0.0

cp_score = 0
cp_score += 1 if total_bil < 2.0 else (2 if 2.0 <= total_bil <= 3.0 else 3)
cp_score += 1 if albumin > 3.5 else (2 if 2.8 <= albumin <= 3.5 else 3)
cp_score += 1 if inr < 1.7 else (2 if 1.7 <= inr <= 2.3 else 3)
cp_score += 1 if ascites == 1.0 else (2 if ascites == 2.0 else 3)
cp_score += 1 if encephalopathy == 1.0 else (2 if encephalopathy == 2.0 else 3)

if st.button("Predict Survival"):
    raw_data = {
        'Gender': gender, 'Symptoms': symptoms, 'Alcohol': alcohol, 'HBsAg': hbsag, 'HBeAg': hbeag, 
        'HBcAb': hbcab, 'HCVAb': hcv, 'Cirrhosis': cirrhosis, 'Endemic Country': endemic, 'Smoking': smoking, 
        'Diabetes': diabetes, 'Obesity': obesity, 'Hemochromatosis': hemochromatosis, 'Arterial Hypertension': htn, 
        'Chronic Renal Insufficency': renal, 'HIV': hiv, 'NASH': nash, 'Varices': varices, 'Splenomegaly': splenomegaly, 
        'PHT': pht, 'PVT': pvt, 'Metastasis': metastasis, 'Hallmark': hallmark, 'Age': age, 'Alcohol Grams_day': alcohol_grams, 
        'Packs_year': packs_year, 'PS': ps, 'Encephalopathy': encephalopathy, 'Ascites': ascites, 'INR': inr, 
        'AFP': afp, 'Hemoglobin': hemoglobin, 'MCV': mcv, 'Leucocytes': leucocytes, 'Platelets': platelets, 
        'Total_Bil': total_bil, 'ALT': alt, 'AST': ast, 'GGT': ggt, 'ALP': alp, 'Total Protein': total_protein, 
        'Creatinine': creatinine, 'Nodule': nodule, 'Major_Dim': major_dim, 'Iron': iron, 
        'Oxygen Saturation': oxygen_sat, 'Ferritin': ferritin,
        'AST_ALT_Ratio': ast_alt_ratio, 'ALBI_Score': albi_score, 'FIB4_Score': fib4_score, 
        'Child_Pugh_Score': cp_score, 'Comorbidities_Count': comorbidities_count
    }
    
    input_df = pd.DataFrame([raw_data])
    
    for col in features:
        if col not in input_df.columns:
            input_df[col] = 0.0
            
    input_df = input_df[features]
    
    
    if st.button("Predict Survival", key="predict_btn"):
        input_scaled = scaler.transform(input_df.values)
        prediction = model.predict(input_scaled)

    st.subheader("Prediction Result:")
    if prediction[0] == 1:
        st.success("🟢 Prediction: Favorable Outcome / Likely to Survive")
    else:
        st.error("🔴 Prediction: Unfavorable Outcome / High Mortality Risk")