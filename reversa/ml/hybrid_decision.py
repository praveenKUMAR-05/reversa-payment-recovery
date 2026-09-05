import pandas as pd
import joblib

from policy_engine import check_policy


# ========================================
# REVERSA HYBRID DECISION ENGINE
# ========================================

pipeline = joblib.load(
    "../data/reversa_model.pkl"
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


# ========================================
# FAILURE-REASON POLICY
# ========================================

REASON_POLICY = {

    "authentication_failed":
        "PAYMENT_LINK",

    "bank_error":
        "RETRY",

    "insufficient_funds":
        "WAIT",

    "payment_method_issue":
        "PAYMENT_LINK"
}


# ========================================
# LOAD PAYMENTS
# ========================================

payments = pd.read_csv(
    "../data/failed_payments.csv"
)


payment = payments.iloc[0]


# ========================================
# DISPLAY PAYMENT
# ========================================

print("========================================")
print("REVERSA HYBRID DECISION ENGINE")
print("========================================")

print(
    "Payment ID:",
    payment["payment_id"]
)

print(
    "Customer ID:",
    payment["customer_id"]
)

print(
    "Amount: ₹",
    payment["amount"]
)

print(
    "Failure:",
    payment["failure_reason"]
)


# ========================================
# CREATE ACTION CANDIDATES
# ========================================

actions = [
    "WAIT",
    "RETRY",
    "REMINDER",
    "PAYMENT_LINK"
]


candidates = []

for action in actions:

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


# ========================================
# ML PREDICTIONS
# ========================================

candidate_df[
    "ml_probability"
] = pipeline.predict_proba(
    candidate_df
)[:, 1]


# ========================================
# ACTION COST
# ========================================

candidate_df[
    "action_cost"
] = candidate_df[
    "action"
].map(ACTION_COSTS)


# ========================================
# NATURAL RECOVERY
# ========================================

wait_probability = candidate_df.loc[
    candidate_df["action"] == "WAIT",
    "ml_probability"
].iloc[0]


# ========================================
# INCREMENTAL VALUE
# ========================================

candidate_df[
    "incremental_recovery"
] = (
    candidate_df["ml_probability"]
    -
    wait_probability
)


candidate_df[
    "incremental_revenue"
] = (
    candidate_df["amount"]
    *
    candidate_df["incremental_recovery"]
)


candidate_df[
    "net_value"
] = (
    candidate_df["incremental_revenue"]
    -
    candidate_df["action_cost"]
)


# ========================================
# REASON POLICY ACTION
# ========================================

reason = payment["failure_reason"]

policy_action = REASON_POLICY.get(
    reason,
    "WAIT"
)


candidate_df[
    "reason_policy"
] = candidate_df["action"].apply(
    lambda x:
        x == policy_action
)


# ========================================
# POLICY VALIDATION
# ========================================

candidate_df[
    "policy_allowed"
] = False


for index, row in candidate_df.iterrows():

    result = check_policy(

        action=row["action"],

        recovery_probability=
            row["ml_probability"],

        attempt_number=
            payment["attempt_number"],

        payment_recovered=False,

        amount=payment["amount"]
    )

    candidate_df.loc[
        index,
        "policy_allowed"
    ] = result["allowed"]


# ========================================
# HYBRID SCORING
# ========================================

# The reason policy provides the
# business/domain preference.
#
# ML probability remains important.
#
# We give a strong preference to the
# reason-aware action while still using
# ML probability and economics.

candidate_df[
    "hybrid_score"
] = candidate_df[
    "net_value"
]


# Bonus for domain-recommended action.

candidate_df.loc[
    candidate_df["reason_policy"] == True,
    "hybrid_score"
] += 20


# ========================================
# REMOVE DISALLOWED ACTIONS
# ========================================

allowed_df = candidate_df[
    candidate_df["policy_allowed"]
].copy()


# ========================================
# NO ACTION
# ========================================

if len(allowed_df) == 0:

    print()
    print("========================================")
    print("FINAL RECOMMENDATION")
    print("========================================")

    print(
        "Action: NO_ACTION"
    )

    print(
        "Reason: No action passed policy checks."
    )

    raise SystemExit


# ========================================
# SELECT BEST ACTION
# ========================================

allowed_df = allowed_df.sort_values(
    "hybrid_score",
    ascending=False
)


best = allowed_df.iloc[0]


# ========================================
# ACTION ANALYSIS
# ========================================

print()
print("========================================")
print("ACTION ANALYSIS")
print("========================================")

display_columns = [

    "action",

    "ml_probability",

    "reason_policy",

    "action_cost",

    "incremental_revenue",

    "net_value",

    "hybrid_score"
]


print(
    allowed_df[
        display_columns
    ].to_string(index=False)
)


# ========================================
# FINAL RECOMMENDATION
# ========================================

print()
print("========================================")
print("FINAL RECOMMENDATION")
print("========================================")

print(
    "Action:",
    best["action"]
)

print(
    "ML recovery probability:",
    round(
        best["ml_probability"] * 100,
        2
    ),
    "%"
)

print(
    "Natural recovery probability:",
    round(
        wait_probability * 100,
        2
    ),
    "%"
)

print(
    "Reason-policy recommendation:",
    policy_action
)

print(
    "Expected additional revenue: ₹",
    round(
        best["incremental_revenue"],
        2
    )
)

print(
    "Action cost: ₹",
    round(
        best["action_cost"],
        2
    )
)

print(
    "Net incremental value: ₹",
    round(
        best["net_value"],
        2
    )
)

print(
    "Hybrid score:",
    round(
        best["hybrid_score"],
        2
    )
)

print(
    "Reason-policy match:",
    "YES"
    if best["reason_policy"]
    else "NO"
)