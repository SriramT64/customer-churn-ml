import pandas as pd
import joblib

# Load model and preprocessor
model = joblib.load("src/churn_model.pkl")
preprocessor = joblib.load("src/preprocessor.pkl")

print("\n===== CUSTOMER CHURN PREDICTION =====\n")

customer = {
    "gender": input("Gender (Male/Female): "),
    "SeniorCitizen": int(input("Senior Citizen (0/1): ")),
    "Partner": input("Partner (Yes/No): "),
    "Dependents": input("Dependents (Yes/No): "),
    "tenure": int(input("Tenure (months): ")),
    "PhoneService": input("Phone Service (Yes/No): "),
    "MultipleLines": input("Multiple Lines (Yes/No/No phone service): "),
    "InternetService": input("Internet Service (DSL/Fiber optic/No): "),
    "OnlineSecurity": input("Online Security (Yes/No/No internet service): "),
    "OnlineBackup": input("Online Backup (Yes/No/No internet service): "),
    "DeviceProtection": input("Device Protection (Yes/No/No internet service): "),
    "TechSupport": input("Tech Support (Yes/No/No internet service): "),
    "StreamingTV": input("Streaming TV (Yes/No/No internet service): "),
    "StreamingMovies": input("Streaming Movies (Yes/No/No internet service): "),
    "Contract": input("Contract (Month-to-month/One year/Two year): "),
    "PaperlessBilling": input("Paperless Billing (Yes/No): "),
    "PaymentMethod": input(
        "Payment Method "
        "(Electronic check/Mailed check/Bank transfer (automatic)/Credit card (automatic)): "
    ),
    "MonthlyCharges": float(input("Monthly Charges: ")),
    "TotalCharges": float(input("Total Charges: "))
}

# Convert input into DataFrame
customer_df = pd.DataFrame([customer])

# Apply the SAME preprocessing used during training
customer_processed = preprocessor.transform(customer_df)

# Prediction
prediction = model.predict(customer_processed)[0]

# Probability
probability = model.predict_proba(customer_processed)[0][1]

print("\n===== RESULT =====")

if prediction == 1:
    print("Prediction: CUSTOMER WILL CHURN")
else:
    print("Prediction: CUSTOMER WILL STAY")

print(f"Churn Probability: {probability:.2%}")