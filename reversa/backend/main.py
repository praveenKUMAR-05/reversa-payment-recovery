from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import os


# ========================================
# REVERSA BACKEND
# ========================================

app = FastAPI(
    title="REVERSA API",
    description="Failed payment recovery decision engine",
    version="1.0.0"
)


# ========================================
# CORS
# ========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========================================
# PATHS
# ========================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "data",
    "reversa_model.pkl"
)


# ========================================
# LOAD MODEL
# ========================================

pipeline = joblib.load(
    MODEL_PATH
)


# ========================================
# REVERSA CONFIGURATION
# ========================================

ACTIONS = [
    "WAIT",
    "RETRY",
    "REMINDER",
    "PAYMENT_LINK"
]

ACTION_COSTS = {
    "WAIT": 0,
    "RETRY": 5,
    "REMINDER": 1,
    "PAYMENT_LINK": 2
}

MIN_RECOVERY_PROBABILITY = 0.60
MAX_RETRIES = 2


# ========================================
# REQUEST MODEL
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
# HEALTH CHECK
# ========================================

@app.get("/")
def root():

    return {
        "system": "REVERSA",
        "status": "running",
        "message": "Failed payment recovery decision engine"
    }


# ========================================
# ANALYZE PAYMENT
# ========================================

@app.post("/api/analyze")
def analyze_payment(
    payment: PaymentRequest
):

    # ------------------------------------
    # CREATE ACTION CANDIDATES
    # ------------------------------------

    candidates = []

    for action in ACTIONS:

        candidates.append({

            "amount":
                payment.amount,

            "failure_reason":
                payment.failure_reason,

            "attempt_number":
                payment.attempt_number,

            "subscription_months":
                payment.subscription_months,

            "successful_payments":
                payment.successful_payments,

            "failed_payments":
                payment.failed_payments,

            "avg_delay_days":
                payment.avg_delay_days,

            "reminder_success_rate":
                payment.reminder_success_rate,

            "action":
                action
        })


    candidate_df = pd.DataFrame(
        candidates
    )


    # ------------------------------------
    # ML PREDICTIONS
    # ------------------------------------

    probabilities = pipeline.predict_proba(
        candidate_df
    )[:, 1]


    candidate_df[
        "recovery_probability"
    ] = probabilities


    # ------------------------------------
    # ACTION COST
    # ------------------------------------

    candidate_df[
        "action_cost"
    ] = candidate_df[
        "action"
    ].map(
        ACTION_COSTS
    )


    # ------------------------------------
    # NATURAL RECOVERY
    # ------------------------------------

    wait_probability = candidate_df.loc[
        candidate_df["action"] == "WAIT",
        "recovery_probability"
    ].iloc[0]


    # ------------------------------------
    # INCREMENTAL RECOVERY
    # ------------------------------------

    candidate_df[
        "incremental_recovery"
    ] = (
        candidate_df[
            "recovery_probability"
        ]
        -
        wait_probability
    )


    # ------------------------------------
    # EXPECTED ADDITIONAL REVENUE
    # ------------------------------------

    candidate_df[
        "incremental_revenue"
    ] = (
        payment.amount
        *
        candidate_df[
            "incremental_recovery"
        ]
    )


    # ------------------------------------
    # NET VALUE
    # ------------------------------------

    candidate_df[
        "net_incremental_value"
    ] = (
        candidate_df[
            "incremental_revenue"
        ]
        -
        candidate_df[
            "action_cost"
        ]
    )


    # ------------------------------------
    # APPLY POLICY
    # ------------------------------------

    allowed = []

    for _, row in candidate_df.iterrows():

        action = row["action"]

        probability = row[
            "recovery_probability"
        ]


        # Probability safety rule

        if probability < MIN_RECOVERY_PROBABILITY:

            continue


        # Retry limit

        if (
            action == "RETRY"
            and
            payment.attempt_number >= MAX_RETRIES
        ):

            continue


        allowed.append(row)


    # ------------------------------------
    # NO SAFE ACTION
    # ------------------------------------

    if len(allowed) == 0:

        return {

            "payment_id":
                payment.payment_id,

            "customer_id":
                payment.customer_id,

            "amount":
                payment.amount,

            "failure_reason":
                payment.failure_reason,

            "selected_action":
                "NO_ACTION",

            "recovery_probability":
                round(
                    float(wait_probability),
                    4
                ),

            "natural_recovery_probability":
                round(
                    float(wait_probability),
                    4
                ),

            "expected_additional_revenue":
                0,

            "action_cost":
                0,

            "net_incremental_value":
                0,

            "reason":
                "No action passed the policy checks."
        }


    # ------------------------------------
    # SELECT BEST ACTION
    # ------------------------------------

    allowed_df = pd.DataFrame(
        allowed
    )


    best = allowed_df.loc[
        allowed_df[
            "net_incremental_value"
        ].idxmax()
    ]


    # ------------------------------------
    # POSITIVE VALUE CHECK
    # ------------------------------------

    if best[
        "net_incremental_value"
    ] <= 0:

        selected_action = "NO_ACTION"

        expected_revenue = 0

        action_cost = 0

        net_value = 0

    else:

        selected_action = best[
            "action"
        ]

        expected_revenue = float(
            best[
                "incremental_revenue"
            ]
        )

        action_cost = float(
            best[
                "action_cost"
            ]
        )

        net_value = float(
            best[
                "net_incremental_value"
            ]
        )


    # ====================================
    # RETURN RESULT
    # ====================================

    return {

        "payment_id":
            payment.payment_id,

        "customer_id":
            payment.customer_id,

        "amount":
            payment.amount,

        "failure_reason":
            payment.failure_reason,

        "selected_action":
            selected_action,

        "recovery_probability":
            round(
                float(
                    best[
                        "recovery_probability"
                    ]
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
                expected_revenue,
                2
            ),

        "action_cost":
            round(
                action_cost,
                2
            ),

        "net_incremental_value":
            round(
                net_value,
                2
            ),

        "reason":
            "Highest net incremental value among policy-allowed actions."
    }