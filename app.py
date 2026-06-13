import streamlit as st
import pickle
import pandas as pd

# Load model and scaler
with open("loan_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("🏦 CreditWise Loan Approval Prediction")

# Numeric Inputs
Applicant_Income = st.number_input("Applicant Income", min_value=0.0)
Coapplicant_Income = st.number_input("Coapplicant Income", min_value=0.0)
Age = st.number_input("Age", min_value=18)
Dependents = st.number_input("Dependents", min_value=0)
Existing_Loans = st.number_input("Existing Loans", min_value=0)
Savings = st.number_input("Savings", min_value=0.0)
Collateral_Value = st.number_input("Collateral Value", min_value=0.0)
Loan_Amount = st.number_input("Loan Amount", min_value=0.0)
Loan_Term = st.number_input("Loan Term (Months)", min_value=1)
Credit_Score = st.number_input("Credit Score", min_value=0.0)
DTI_Ratio = st.number_input("DTI Ratio", min_value=0.0)

# Categorical Inputs
Education_Level = st.selectbox(
    "Education Level",
    ["Graduate", "High School", "Postgraduate"]
)

Gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

Employment_Status = st.selectbox(
    "Employment Status",
    ["Salaried", "Self-employed", "Unemployed"]
)

Marital_Status = st.selectbox(
    "Marital Status",
    ["Married", "Single"]
)

Loan_Purpose = st.selectbox(
    "Loan Purpose",
    ["Business", "Car", "Education", "Home", "Personal"]
)

Property_Area = st.selectbox(
    "Property Area",
    ["Rural", "Semiurban", "Urban"]
)

Employer_Category = st.selectbox(
    "Employer Category",
    ["Government", "MNC", "Private", "Unemployed"]
)

if st.button("Predict"):

    data = {
        "Applicant_Income": Applicant_Income,
        "Coapplicant_Income": Coapplicant_Income,
        "Age": Age,
        "Dependents": Dependents,
        "Existing_Loans": Existing_Loans,
        "Savings": Savings,
        "Collateral_Value": Collateral_Value,
        "Loan_Amount": Loan_Amount,
        "Loan_Term": Loan_Term,

        "Education_Level": 0 if Education_Level == "Graduate" else 1,
        "Gender": 1 if Gender == "Male" else 0,

        "Employment_Status_Salaried": 1 if Employment_Status == "Salaried" else 0,
        "Employment_Status_Self-employed": 1 if Employment_Status == "Self-employed" else 0,
        "Employment_Status_Unemployed": 1 if Employment_Status == "Unemployed" else 0,

        "Marital_Status_Single": 1 if Marital_Status == "Single" else 0,

        "Loan_Purpose_Car": 1 if Loan_Purpose == "Car" else 0,
        "Loan_Purpose_Education": 1 if Loan_Purpose == "Education" else 0,
        "Loan_Purpose_Home": 1 if Loan_Purpose == "Home" else 0,
        "Loan_Purpose_Personal": 1 if Loan_Purpose == "Personal" else 0,

        "Property_Area_Semiurban": 1 if Property_Area == "Semiurban" else 0,
        "Property_Area_Urban": 1 if Property_Area == "Urban" else 0,

        "Employer_Category_Government": 1 if Employer_Category == "Government" else 0,
        "Employer_Category_MNC": 1 if Employer_Category == "MNC" else 0,
        "Employer_Category_Private": 1 if Employer_Category == "Private" else 0,
        "Employer_Category_Unemployed": 1 if Employer_Category == "Unemployed" else 0,

        "DTI_Ratio_sq": DTI_Ratio ** 2,
        "Credit_Score_sq": Credit_Score ** 2,
    }

    feature_order = [
        "Applicant_Income",
        "Coapplicant_Income",
        "Age",
        "Dependents",
        "Existing_Loans",
        "Savings",
        "Collateral_Value",
        "Loan_Amount",
        "Loan_Term",
        "Education_Level",
        "Gender",
        "Employment_Status_Salaried",
        "Employment_Status_Self-employed",
        "Employment_Status_Unemployed",
        "Marital_Status_Single",
        "Loan_Purpose_Car",
        "Loan_Purpose_Education",
        "Loan_Purpose_Home",
        "Loan_Purpose_Personal",
        "Property_Area_Semiurban",
        "Property_Area_Urban",
        "Employer_Category_Government",
        "Employer_Category_MNC",
        "Employer_Category_Private",
        "Employer_Category_Unemployed",
        "DTI_Ratio_sq",
        "Credit_Score_sq"
    ]

    df = pd.DataFrame([data])
    df = df.reindex(columns=feature_order)

    df_scaled = scaler.transform(df)

    prediction = model.predict(df_scaled)

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")
