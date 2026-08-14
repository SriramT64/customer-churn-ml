import pandas as pd
import joblib
import shap

# Load model and preprocessor
model = joblib.load("src/churn_model.pkl")
preprocessor = joblib.load("src/preprocessor.pkl")

# Create one sample customer
customer = pd.DataFrame([{
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 5,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 90.0,
    "TotalCharges": 450.0
}])

# Transform customer using our trained preprocessor
customer_processed = preprocessor.transform(customer)

# Create SHAP explainer
explainer = shap.TreeExplainer(model)

# Calculate SHAP values
shap_values = explainer.shap_values(customer_processed)

print("SHAP explanation generated successfully!")

print("\nSHAP values:")
print(shap_values)