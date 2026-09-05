import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ========================================
# LOAD DATA
# ========================================

data = pd.read_csv("../data/action_outcomes.csv")

print("========================================")
print("REVERSA MODEL TRAINING")
print("========================================")

print("Dataset shape:", data.shape)


# ========================================
# TARGET
# ========================================

y = data["recovered"]


# ========================================
# FEATURES
# ========================================

feature_columns = [
    "amount",
    "failure_reason",
    "attempt_number",
    "subscription_months",
    "successful_payments",
    "failed_payments",
    "avg_delay_days",
    "reminder_success_rate",
    "action"
]

X = data[feature_columns]


# ========================================
# FEATURE TYPES
# ========================================

categorical_features = [
    "failure_reason",
    "action"
]

numeric_features = [
    "amount",
    "attempt_number",
    "subscription_months",
    "successful_payments",
    "failed_payments",
    "avg_delay_days",
    "reminder_success_rate"
]


# ========================================
# PREPROCESSOR
# ========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# ========================================
# GROUPED TRAIN / TEST SPLIT
# ========================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(
        X,
        y,
        groups=data["payment_id"]
    )
)


X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]


# ========================================
# INFORMATION
# ========================================

print()
print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print(
    "Training payment cases:",
    data.iloc[train_idx]["payment_id"].nunique()
)

print(
    "Testing payment cases:",
    data.iloc[test_idx]["payment_id"].nunique()
)


# ========================================
# MODEL
# ========================================

model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)


# ========================================
# PIPELINE
# ========================================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# ========================================
# TRAIN
# ========================================

pipeline.fit(
    X_train,
    y_train
)

print()
print("Model trained successfully!")


# ========================================
# MODEL PREDICTION
# ========================================

y_probability = pipeline.predict_proba(
    X_test
)[:, 1]

y_prediction = (
    y_probability >= 0.5
).astype(int)


# ========================================
# MODEL PERFORMANCE
# ========================================

accuracy = accuracy_score(
    y_test,
    y_prediction
)

precision = precision_score(
    y_test,
    y_prediction
)

recall = recall_score(
    y_test,
    y_prediction
)

f1 = f1_score(
    y_test,
    y_prediction
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


print()
print("MODEL PERFORMANCE")
print("--------------------------------")

print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))
print("ROC-AUC  :", round(roc_auc, 4))


# ========================================
# CREATE HELD-OUT TEST DATA
# ========================================

test_data = data.iloc[test_idx].copy()

test_data["predicted_probability"] = y_probability


# ========================================
# ACTION SELECTION DATA
# ========================================

print()
print("========================================")
print("HELD-OUT DATA CREATED")
print("========================================")

print(
    "Unique test payments:",
    test_data["payment_id"].nunique()
)

print(
    "Test action records:",
    len(test_data)
)


# ========================================
# EXPORT TEST DATA
# ========================================

test_data.to_csv(
    "../data/test_action_outcomes.csv",
    index=False
)

print()
print(
    "Saved: ../data/test_action_outcomes.csv"
)


# ========================================
# EXPORT MODEL
# ========================================

import joblib

joblib.dump(
    pipeline,
    "../data/reversa_model.pkl"
)

print(
    "Saved: ../data/reversa_model.pkl"
)


# ========================================
# EXPORT SPLIT INFORMATION
# ========================================

split_info = pd.DataFrame({
    "payment_id": data["payment_id"].iloc[test_idx].unique()
})

split_info.to_csv(
    "../data/test_payment_ids.csv",
    index=False
)

print(
    "Saved: ../data/test_payment_ids.csv"
)

print()
print("========================================")
print("TRAINING COMPLETE")
print("========================================")