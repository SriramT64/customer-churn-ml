# Customer Churn Prediction & Explainable ML

An end-to-end machine learning application that predicts whether a customer is likely to churn and explains the factors influencing the prediction.

## 🚀 Live Demo

[Open the deployed Streamlit application](https://customer-churn-ml-bejbx4mwtbeu5pqlxawinc.streamlit.app/)

## 📌 Project Overview

Customer churn prediction helps businesses identify customers who may leave their service.

This project builds a machine learning system that:

- Cleans and preprocesses customer data
- Performs exploratory data analysis
- Trains multiple classification models
- Compares Logistic Regression and Random Forest
- Uses GridSearchCV for hyperparameter tuning
- Evaluates the model using classification metrics and ROC-AUC
- Allows businesses to adjust the churn decision threshold
- Uses SHAP to explain individual predictions
- Provides an interactive Streamlit interface
- Is deployed as a web application

## 🏗️ Project Architecture

```text
Customer Data
      ↓
Data Cleaning
      ↓
Feature Preprocessing
      ↓
Train / Test Split
      ↓
Logistic Regression ──┐
                      ├── Model Evaluation
Random Forest ────────┘
                      ↓
              GridSearchCV
                      ↓
             Best Random Forest
                      ↓
              Churn Probability
                      ↓
           Decision Threshold
                      ↓
             Churn / Stay
                      ↓
              SHAP Explanation
                      ↓
              Streamlit App
