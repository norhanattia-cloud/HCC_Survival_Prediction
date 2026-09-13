# 🩺 Predictive Modeling for Hepatocellular Carcinoma (HCC) Patient Survival

## 📌 Project Overview
This project develops a Machine Learning-based clinical decision-support system to predict survival outcomes for patients with Hepatocellular Carcinoma (HCC). By integrating clinical domain knowledge with advanced ensemble learning, the tool provides real-time, highly accurate prognostic assessments to assist healthcare professionals.

## 🚀 Key Features
* **Clinical Feature Engineering:** Dynamically calculates established medical indices (Child-Pugh Score, ALBI Score, FIB-4 Index, AST/ALT ratio) directly from raw laboratory inputs.
* **Robust Preprocessing Pipeline:** Implements KNN Imputation for missing values, strategically drops highly correlated features to prevent multicollinearity, and applies **SMOTE** to balance the training dataset without causing data leakage.
* **Ensemble Machine Learning:** Utilizes a custom **Soft Voting Classifier** (combining Logistic Regression and Gradient Boosting) to maximize generalization, achieving an overall accuracy of **81.8%** with a balanced recall for both survival and mortality classes.
* **Interactive Deployment:** Features a fully functional **Streamlit** web application with dynamic dimensionality alignment to ensure seamless real-time predictions.

## 📁 Repository Structure
* `Project Notebook.ipynb`: The complete ML pipeline including EDA, Preprocessing, Hyperparameter Tuning (GridSearchCV), and Model Evaluation.
* `app.py`: The Streamlit application script for the graphical user interface.
* `hcc_voting_model.pkl`: The serialized final ensemble model.
* `hcc_scaler.pkl` & `hcc_features.pkl`: Serialized artifacts for data standardization and dimensionality mapping.
* `requirements.txt`: Required Python dependencies.

## ⚙️ How to Run Locally
1. Clone this repository to your local machine.
2. Install the required dependencies using:
   ```bash
   pip install -r requirements.txt
