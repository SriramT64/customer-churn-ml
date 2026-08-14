import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Load trained model and preprocessor
model = joblib.load("src/churn_model.pkl")
preprocessor = joblib.load("src/preprocessor.pkl")
explainer = shap.TreeExplainer(model)

# Page configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊"
)

st.title("📊 Customer Churn Predictor")
st.write("Predict whether a customer is likely to churn.")

st.divider()

st.subheader("Customer Information")

# Basic information
gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

senior_citizen = st.selectbox(
    "Senior Citizen",
    [0, 1]
)

partner = st.selectbox(
    "Partner",
    ["Yes", "No"]
)

dependents = st.selectbox(
    "Dependents",
    ["Yes", "No"]
)

tenure = st.number_input(
    "Tenure (months)",
    min_value=0,
    max_value=100,
    value=12
)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=70.0
)

total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=800.0
)
phone_service = st.selectbox(
    "Phone Service",
    ["Yes", "No"]
)

multiple_lines = st.selectbox(
    "Multiple Lines",
    ["Yes", "No", "No phone service"]
)

internet_service = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

online_security = st.selectbox(
    "Online Security",
    ["Yes", "No", "No internet service"]
)

online_backup = st.selectbox(
    "Online Backup",
    ["Yes", "No", "No internet service"]
)

device_protection = st.selectbox(
    "Device Protection",
    ["Yes", "No", "No internet service"]
)

tech_support = st.selectbox(
    "Tech Support",
    ["Yes", "No", "No internet service"]
)

streaming_tv = st.selectbox(
    "Streaming TV",
    ["Yes", "No", "No internet service"]
)

streaming_movies = st.selectbox(
    "Streaming Movies",
    ["Yes", "No", "No internet service"]
)

contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

paperless_billing = st.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

payment_method = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)
st.divider()

# Threshold selected by user
threshold = st.slider(
    "Churn Decision Threshold",
    min_value=0.30,
    max_value=0.70,
    value=0.50,
    step=0.05
)

# Predict button
if st.button("🔮 Predict Churn"):

    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }])

    customer_processed = preprocessor.transform(customer)

    probability = model.predict_proba(customer_processed)[0][1]
    shap_values = explainer.shap_values(customer_processed)

    st.session_state["probability"] = probability
    st.session_state["shap_values"] = shap_values
    st.session_state["customer"] = customer


# Show result
if "probability" in st.session_state:

    probability = st.session_state["probability"]

    # Compare probability with CURRENT slider value
    if probability >= threshold:
        prediction = "CHURN"
    else:
        prediction = "STAY"

    st.subheader("Prediction Result")

    st.metric(
        "Churn Probability",
        f"{probability:.2%}"
    )

    st.metric(
        "Decision Threshold",
        f"{threshold:.0%}"
    )

    # Show the comparison directly
    st.write(
        f"**{probability:.2%} {'≥' if probability >= threshold else '<'} "
        f"{threshold:.0%}**"
    )

    if prediction == "CHURN":
        st.error("⚠️ Customer is likely to CHURN")
    else:
        st.success("✅ Customer is likely to STAY")


# SHAP explanation
if "shap_values" in st.session_state:

    st.subheader("🔍 Why this prediction?")

    shap_values = st.session_state["shap_values"]

    # Get feature names after preprocessing
    feature_names = preprocessor.get_feature_names_out()

    # Convert SHAP output to numpy array
    shap_array = shap_values

    # Get SHAP values for churn class
    if shap_array.ndim == 3:
        values = shap_array[0, :, 1]

    elif shap_array.ndim == 2:
        values = shap_array[0]

    else:
        values = shap_array

    # Make sure values are 1-dimensional
    values = values.flatten()

    # Create explanation dataframe
    explanation = pd.DataFrame({
        "Feature": feature_names,
        "SHAP Value": values
    })

    # Importance
    explanation["Importance"] = explanation["SHAP Value"].abs()

    # Sort by strongest influence
    explanation = explanation.sort_values(
        "Importance",
        ascending=False
    ).head(10)

    # Make feature names easier to understand
    def clean_feature_name(feature):

        feature = feature.replace("cat__", "")
        feature = feature.replace("num__", "")
        feature = feature.replace("_", " ")

        return feature

    explanation["Feature"] = explanation["Feature"].apply(
        clean_feature_name
    )

    # Add direction of influence
    explanation["Impact"] = explanation["SHAP Value"].apply(
        lambda x: "🔴 Increases churn risk"
        if x > 0
        else "🟢 Reduces churn risk"
    )

    # Display explanation
    st.dataframe(
        explanation[["Feature", "SHAP Value", "Impact"]],
        use_container_width=True
    )

    st.subheader("📊 Feature Impact on Churn")

    # Sort features by SHAP value
    chart_data = explanation.sort_values("SHAP Value")

    # Create colors based on impact
    bar_colors = [
        "green" if value < 0 else "red"
        for value in chart_data["SHAP Value"]
    ]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.barh(
        chart_data["Feature"],
        chart_data["SHAP Value"],
        color=bar_colors
    )

    # Zero line
    ax.axvline(0, linewidth=1)

    ax.set_xlabel("SHAP Value")
    ax.set_ylabel("Feature")
    ax.set_title("Factors Influencing Churn Prediction")

    plt.tight_layout()

    st.pyplot(fig)