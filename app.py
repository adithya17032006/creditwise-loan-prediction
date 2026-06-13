```python
import streamlit as st
import pandas as pd
import pickle

# Page Config
st.set_page_config(
    page_title="CreditWise Loan Prediction",
    page_icon="🏦",
    layout="wide"
)

# Load Model and Scaler
with open("loan_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Header
st.title("🏦 CreditWise Loan Approval Prediction")

st.markdown("""
Predict whether a loan application is likely to be approved using Machine Learning.
""")

# Sidebar
st.sidebar.title("🏦 CreditWise")

st.sidebar.info(
    """
    ### Loan Approval Predictor

    Enter applicant details and get an instant prediction.

    **Model:** Logistic Regression
    **Accuracy:** 87.5%
    """
)

# -------------------------
# Numeric Inputs
# -------------------------

col1, col2 = st.columns(2)

with col1:
    Applicant_Income = st.number_input(
        "Applicant Income",
        min_value=0.0,
        value=10000.0
    )

    Coapplicant_Income = st.number_input(
        "Coapplicant Income",
        min_value=0.0,
        value=2000.0
    )

    Age = st.number_input(
        "Age",
        min_value=18,
        value=30
    )

    Dependents = st.number_input(
        "Dependents",
        min_value=0,
        value=0
    )

    Credit_Score = st.number_input(
        "Credit Score",
        min_value=300.0,
        max_value=900.0,
        value=650.0
    )

with col2:

    Existing_Loans = st.number_input(
        "Existing Loans",
        min_value=0,
        value=0
    )

    DTI_Ratio = st.number_input(
        "DTI Ratio",
        min_value=0.0,
        max_value=2.0,
        value=0.30
    )

    Savings = st.number_input(
        "Savings",
        min_value=0.0,
        value=10000.0
    )

    Collateral_Value = st.number_input(
        "Collateral Value",
        min_value=0.0,
        value=50000.0
    )

    Loan_Amount = st.number_input(
        "Loan Amount",
        min_value=0.0,
        value=15000.0
    )

Loan_Term = st.number_input(
    "Loan Term (Months)",
    min_value=1,
    value=60
)

# -------------------------
# Categorical Inputs
# -------------------------

Education_Level = st.selectbox(
    "Education Level",
    ["Graduate", "Not Graduate"]
)

Gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
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

# -------------------------
# Prediction
# -------------------------

if st.button("Predict Loan Approval"):

    education_encoded = 0 if Education_Level == "Graduate" else 1
    gender_encoded = 1 if Gender == "Male" else 0

    data = {
        "Applicant_Income": Applicant_Income,
        "Coapplicant_Income": Coapplicant_Income,
        "Age": Age,
        "Dependents": Dependents,
        "Credit_Score": Credit_Score,
        "Existing_Loans": Existing_Loans,
        "DTI_Ratio": DTI_Ratio,
        "Savings": Savings,
        "Collateral_Value": Collateral_Value,
        "Loan_Amount": Loan_Amount,
        "Loan_Term": Loan_Term,
        "Education_Level": education_encoded,
        "Gender": gender_encoded,

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
    }

    feature_order = [
        "Applicant_Income",
        "Coapplicant_Income",
        "Age",
        "Dependents",
        "Credit_Score",
        "Existing_Loans",
        "DTI_Ratio",
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
        "Employer_Category_Unemployed"
    ]

    df = pd.DataFrame([data])
    df = df[feature_order]

    try:
        df_scaled = scaler.transform(df)

        prediction = model.predict(df_scaled)[0]

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(df_scaled)[0][1]

            st.metric(
                "Approval Probability",
                f"{probability * 100:.2f}%"
            )

        if prediction == 1:
            st.success("✅ Loan Approved")
            st.balloons()
        else:
            st.error("❌ Loan Rejected")

    except Exception as e:
        st.error(f"Prediction Error: {e}")

st.markdown("---")

st.caption(
    "Developed by Adithya H S | CreditWise Loan Approval Prediction"
)
```
