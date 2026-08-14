import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

df = pd.read_csv("data/customer_churn.csv")

print(df.head())
print(df.shape)
print(df.info())

print("\nChurn Distribution:")
print(df["Churn"].value_counts())

print("\nChurn Percentage:")
print(df["Churn"].value_counts(normalize=True)*100)

print("\nMissing values:")
print(df.isnull().sum())

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"],errors="coerce")
print("\nAfter converting Totalcharges:")
print(df["TotalCharges"].dtype)
print("\nMissing values after conversion:")
print(df["TotalCharges"].isnull().sum())

df["TotalCharges"] = df["TotalCharges"].fillna(0)
print("\nMissing values after cleaning")
print(df.isnull().sum().sum())

print("\nChurn by Contract")
print(pd.crosstab(df["Contract"],df["Churn"]))

churn_rate = (df.groupby("Contract")["Churn"].value_counts(normalize=True).unstack())
print(churn_rate)

churn_rate.plot(kind="bar")
plt.title("churn rate by contract type")
plt.xlabel("contract type")
plt.ylabel("churn rate")
plt.legend(title="Churn")
plt.tight_layout()
plt.show()

internet_churn = (df.groupby("InternetService")["Churn"].value_counts(normalize=True).unstack())
print(internet_churn)

internet_churn.plot(kind="bar")
plt.title("churn rate by internet service")
plt.xlabel("internet type")
plt.ylabel("churn rate")
plt.legend(title="Churn")
plt.tight_layout()
plt.show()

print("\nChurn by tenure")
tenure_churn = (df.groupby("Churn")["tenure"].mean())
print(tenure_churn)

print("\nChurn by Monthly charges")
monthly_charges = (df.groupby("Churn")["MonthlyCharges"].mean())
print(monthly_charges)

print("\nData types")
print(df.dtypes)

df = df.drop("customerID",axis=1)

df["Churn"] = df["Churn"].map({"No":0,"Yes":1})
print("\nAfter preproccessing target")
print(df.head())
print(df.dtypes)

X = df.drop("Churn",axis=1)
y = df["Churn"]
print("\nFeatures (X):")
print(X.head())
print("\nFeatures (y):")
print(y.head())
print("\nX shape:",X.shape)
print("\ny shape:",y.shape)

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42,test_size=0.2,stratify=y)
print("\nTrain/Test split")
print("X_train:",X_train.shape)
print("X_test:",X_test.shape)
print("y_train:",y_train.shape)
print("y_test:",y_test.shape)

numerical_features = ["SeniorCitizen","tenure","MonthlyCharges","TotalCharges"]
categorial_features = [col for col in X.columns
                       if col not in numerical_features]
print("\nnumerical_features")
print(numerical_features)
print("\ncategorial_features")
print(categorial_features)

preprocessor = ColumnTransformer(transformers=[("num",StandardScaler(),numerical_features),("cat",OneHotEncoder(handle_unknown="ignore"),categorial_features)])
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)
print("\nProcessed data")
print("X_train_processed:",X_train_processed)
print("X_test_processed:",X_test_processed)


model = LogisticRegression(max_iter=2000,class_weight="balanced")
model.fit(X_train_processed,y_train)
print("\nmodel training completed!")

y_pred = model.predict(X_test_processed)
print("\nfirst 20 predictions:",y_pred[:20])
print("\nActual values:",y_test.to_numpy()[:20])

print("accuracy:",accuracy_score(y_test,y_pred))
print("confusuon matrix:",confusion_matrix(y_test,y_pred))
print("classification:",classification_report(y_test,y_pred))

rf_model = RandomForestClassifier(n_estimators=200,random_state=42,class_weight="balanced")
rf_model.fit(X_train_processed,y_train)
print("\nRandom forest training completed!")

rf_pred = rf_model.predict(X_test_processed)
print("\nfirst 20 predictions:",rf_pred[:20])

print("rf_accuracy:",accuracy_score(y_test,rf_pred))
print("rf_confusuon matrix:",confusion_matrix(y_test,rf_pred))
print("rf_classification:",classification_report(y_test,rf_pred))

importances = rf_model.feature_importances_
feature_names = preprocessor.get_feature_names_out()
feature_importance = pd.DataFrame({"feature": feature_names,"importance": importances})
feature_importance = feature_importance.sort_values(by="importance",ascending=False)
print("\nTop 15 important features")
print(feature_importance.head(15))

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5],
    "class_weight": [None, "balanced"]}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1)

grid_search.fit(X_train_processed, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best CV score:", grid_search.best_score_)

best_rf_model = grid_search.best_estimator_

tuned_rf_pred = best_rf_model.predict(X_test_processed)

print("tuned_rf_accuracy:",accuracy_score(y_test,tuned_rf_pred))
print("tuned_rf_confusuon matrix:",confusion_matrix(y_test,tuned_rf_pred))
print("tuned_rf_classification:",classification_report(y_test,tuned_rf_pred))

# Probability of class 1 (churn)
y_prob_lr = model.predict_proba(X_test_processed)[:, 1]
y_prob_rf = best_rf_model.predict_proba(X_test_processed)[:, 1]

print("Logistic Regression ROC-AUC:",
      roc_auc_score(y_test, y_prob_lr))

print("Tuned Random Forest ROC-AUC:",
      roc_auc_score(y_test, y_prob_rf))

import joblib

joblib.dump(best_rf_model, "src/churn_model.pkl")
joblib.dump(preprocessor, "src/preprocessor.pkl")

print("Model and preprocessor saved successfully!")


import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

# Get churn probabilities
y_prob = best_rf_model.predict_proba(X_test_processed)[:, 1]

thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

print("\n===== THRESHOLD ANALYSIS =====")

for threshold in thresholds:

    y_pred_threshold = (y_prob >= threshold).astype(int)

    precision = precision_score(y_test, y_pred_threshold)
    recall = recall_score(y_test, y_pred_threshold)
    f1 = f1_score(y_test, y_pred_threshold)

    print(f"\nThreshold: {threshold}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 Score:  {f1:.3f}")