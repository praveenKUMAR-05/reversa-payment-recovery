from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

from policy_engine import check_policy


# ========================================
# REVERSA API
# ========================================

app = FastAPI(
    title="REVERSA API",
    description="Intelligent failed-payment recovery decision engine",
    version="1.0"
)


# ========================================
# LOAD MODEL
# ========================================

pipeline = joblib.load(
    "../data/reversa_model.pkl"
)


# ========================================
# LOAD PAYMENT DATA
# ========================================

payments = pd.read_csv(
    "../data/failed_payments.csv"
)


# ========================================
# ACTION COSTS
# ========================================

ACTION_COSTS = {
    "WAIT": 0,
    "RETRY": 5,
    "REMINDER": 1,
    "PAYMENT_LINK": 2
}


ACTIONS = [
    "WAIT",
    "RETRY",
    "REMINDER",
    "PAYMENT_LINK"
]


# ========================================
# INPUT MODEL
# ========================================

class PaymentRequest(BaseModel):

    payment_id: str
    customer_id: str
    amount: float
    failure_reason: str
    attempt_number: int
    subscription_months: int
    successful_payments: int
    failed_payments: int
    avg_delay_days: float
    reminder_success_rate: float


# ========================================
# HOME
# ========================================

@app.get("/")
def home():

    return {
        "system": "REVERSA",
        "status": "running",
        "message": "Failed payment recovery decision engine"
    }


# ========================================
# CORE DECISION FUNCTION
# ========================================

def make_decision(payment):

    candidates = []

    # ------------------------------------
    # Create candidate for every action
    # ------------------------------------

    for action in ACTIONS:

        candidates.append({

            "amount":
                payment["amount"],

            "failure_reason":
                payment["failure_reason"],

            "attempt_number":
                payment["attempt_number"],

            "subscription_months":
                payment["subscription_months"],

            "successful_payments":
                payment["successful_payments"],

            "failed_payments":
                payment["failed_payments"],

            "avg_delay_days":
                payment["avg_delay_days"],

            "reminder_success_rate":
                payment["reminder_success_rate"],

            "action":
                action
        })


    candidate_df = pd.DataFrame(
        candidates
    )


    # ------------------------------------
    # ML prediction
    # ------------------------------------

    probabilities = pipeline.predict_proba(
        candidate_df
    )[:, 1]


    candidate_df[
        "recovery_probability"
    ] = probabilities


    # ------------------------------------
    # Action cost
    # ------------------------------------

    candidate_df[
        "action_cost"
    ] = candidate_df[
        "action"
    ].map(ACTION_COSTS)


    # ------------------------------------
    # Natural recovery baseline
    # ------------------------------------

    wait_probability = candidate_df.loc[
        candidate_df["action"] == "WAIT",
        "recovery_probability"
    ].iloc[0]


    # ------------------------------------
    # Incremental recovery
    # ------------------------------------

    candidate_df[
        "incremental_recovery"
    ] = (
        candidate_df["recovery_probability"]
        - wait_probability
    )


    # ------------------------------------
    # Expected additional revenue
    # ------------------------------------

    candidate_df[
        "incremental_revenue"
    ] = (
        payment["amount"]
        *
        candidate_df["incremental_recovery"]
    )


    # ------------------------------------
    # Net incremental value
    # ------------------------------------

    candidate_df[
        "net_value"
    ] = (
        candidate_df["incremental_revenue"]
        -
        candidate_df["action_cost"]
    )


    # ------------------------------------
    # Policy filter
    # ------------------------------------

    allowed = []

    for _, row in candidate_df.iterrows():

        result = check_policy(

            action=row["action"],

            recovery_probability=
                row["recovery_probability"],

            attempt_number=
                payment["attempt_number"],

            payment_recovered=False,

            amount=payment["amount"]
        )


        if result["allowed"]:

            allowed.append(row)


    # ------------------------------------
    # No allowed action
    # ------------------------------------

    if not allowed:

        return {

            "payment_id":
                payment["payment_id"],

            "customer_id":
                payment["customer_id"],

            "amount":
                float(payment["amount"]),

            "failure_reason":
                payment["failure_reason"],

            "recommended_action":
                "NO_ACTION",

            "natural_recovery_probability":
                round(
                    float(wait_probability),
                    4
                ),

            "reason":
                "No action passed policy checks."
        }


    # ------------------------------------
    # Select best action
    # ------------------------------------

    allowed_df = pd.DataFrame(
        allowed
    )


    allowed_df = allowed_df.sort_values(
        "net_value",
        ascending=False
    )


    best = allowed_df.iloc[0]


    # ------------------------------------
    # Final result
    # ------------------------------------

    return {

        "payment_id":
            payment["payment_id"],

        "customer_id":
            payment["customer_id"],

        "amount":
            float(payment["amount"]),

        "failure_reason":
            payment["failure_reason"],

        "recommended_action":
            best["action"],

        "recovery_probability":
            round(
                float(
                    best["recovery_probability"]
                ),
                4
            ),

        "natural_recovery_probability":
            round(
                float(wait_probability),
                4
            ),

        "expected_additional_revenue":
            round(
                float(
                    best["incremental_revenue"]
                ),
                2
            ),

        "action_cost":
            float(
                best["action_cost"]
            ),

        "net_incremental_value":
            round(
                float(
                    best["net_value"]
                ),
                2
            ),

        "reason":
            "Highest net incremental value among policy-allowed actions."
    }


# ========================================
# MANUAL PREDICTION ENDPOINT
# ========================================

@app.post("/predict")
def predict(payment: PaymentRequest):

    payment_data = payment.model_dump()

    return make_decision(
        payment_data
    )


# ========================================
# PAYMENT ID RECOMMENDATION
# ========================================

@app.post("/recommend/{payment_id}")
def recommend(payment_id: str):

    # ------------------------------------
    # Find payment
    # ------------------------------------

    matching_payments = payments[
        payments["payment_id"] == payment_id
    ]


    # ------------------------------------
    # Payment not found
    # ------------------------------------

    if matching_payments.empty:

        raise HTTPException(
            status_code=404,
            detail=f"Payment {payment_id} not found."
        )


    # ------------------------------------
    # Get payment
    # ------------------------------------

    payment = matching_payments.iloc[0]


    # ------------------------------------
    # Make decision
    # ------------------------------------

    return make_decision(
        payment.to_dict()
    )