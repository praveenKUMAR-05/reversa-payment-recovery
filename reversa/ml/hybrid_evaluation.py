import pandas as pd
import joblib

# ========================================
# REVERSA HYBRID EVALUATION
# ========================================

print("========================================")
print("REVERSA HYBRID POLICY EVALUATION")
print("========================================")


# ========================================
# LOAD DATA
# ========================================

df = pd.read_csv("../data/test_action_outcomes.csv")

model = joblib.load("../data/reversa_model.pkl")


# ========================================
# FEATURES
# ========================================

features = [
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


# ========================================
# ACTIONS
# ========================================

actions = [
    "WAIT",
    "RETRY",
    "REMINDER",
    "PAYMENT_LINK"
]


# ========================================
# COSTS
# ========================================

ACTION_COSTS = {
    "WAIT": 0,
    "RETRY": 5,
    "REMINDER": 1,
    "PAYMENT_LINK": 2
}


# ========================================
# REASON POLICY
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
# UNIQUE PAYMENTS
# ========================================

payments = df.drop_duplicates(
    subset=["payment_id"]
).copy()


print()
print("Test payments:", len(payments))


# ========================================
# HYBRID DECISION
# ========================================

results = []


for _, payment in payments.iterrows():

    candidates = []

    for action in actions:

        row = {
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
        }

        candidates.append(row)


    candidate_df = pd.DataFrame(candidates)


    # ====================================
    # ML PREDICTIONS
    # ====================================

    probabilities = model.predict_proba(
        candidate_df[features]
    )[:, 1]


    candidate_df[
        "ml_probability"
    ] = probabilities


    # ====================================
    # NATURAL RECOVERY
    # ====================================

    wait_probability = candidate_df.loc[
        candidate_df["action"] == "WAIT",
        "ml_probability"
    ].iloc[0]


    # ====================================
    # ECONOMIC VALUE
    # ====================================

    candidate_df["incremental_recovery"] = (
        candidate_df["ml_probability"]
        - wait_probability
    )


    candidate_df["action_cost"] = (
        candidate_df["action"]
        .map(ACTION_COSTS)
    )


    candidate_df["incremental_revenue"] = (
        candidate_df["amount"]
        *
        candidate_df["incremental_recovery"]
    )


    candidate_df["net_value"] = (
        candidate_df["incremental_revenue"]
        -
        candidate_df["action_cost"]
    )


    # ====================================
    # REASON POLICY
    # ====================================

    reason = payment["failure_reason"]

    policy_action = REASON_POLICY.get(
        reason,
        "WAIT"
    )


    candidate_df["reason_policy"] = (
        candidate_df["action"]
        == policy_action
    )


    # ====================================
    # HYBRID SCORE
    # ====================================

    candidate_df["hybrid_score"] = (
        candidate_df["net_value"]
        +
        candidate_df["reason_policy"]
        * 20
    )


    # ====================================
    # SELECT ACTION
    # ====================================

    best = candidate_df.sort_values(
        "hybrid_score",
        ascending=False
    ).iloc[0]


    # ====================================
    # ACTUAL OUTCOME
    # ====================================

    actual = df[
        df["payment_id"]
        ==
        payment["payment_id"]
    ]

    actual_row = actual[
        actual["action"]
        ==
        best["action"]
    ]

    recovered = int(
        actual_row["recovered"].iloc[0]
    )


    results.append({

        "payment_id":
            payment["payment_id"],

        "failure_reason":
            reason,

        "selected_action":
            best["action"],

        "ml_probability":
            best["ml_probability"],

        "net_value":
            best["net_value"],

        "recovered":
            recovered,

        "reason_policy_action":
            policy_action
    })


# ========================================
# RESULTS DATAFRAME
# ========================================

results_df = pd.DataFrame(results)


# ========================================
# OVERALL PERFORMANCE
# ========================================

recovered = results_df["recovered"].sum()

total = len(results_df)

recovery_rate = (
    recovered / total
    * 100
)


print()
print("========================================")
print("HYBRID PERFORMANCE")
print("========================================")

print(
    "Payments:",
    total
)

print(
    "Recovered:",
    recovered
)

print(
    "Recovery rate:",
    round(recovery_rate, 2),
    "%"
)


# ========================================
# ACTION DISTRIBUTION
# ========================================

print()
print("========================================")
print("ACTION DISTRIBUTION")
print("========================================")

print(
    results_df[
        "selected_action"
    ].value_counts()
)


# ========================================
# PERFORMANCE BY FAILURE
# ========================================

print()
print("========================================")
print("PERFORMANCE BY FAILURE REASON")
print("========================================")

reason_results = (
    results_df
    .groupby("failure_reason")
    .agg(
        payments=("payment_id", "count"),
        recovered=("recovered", "sum")
    )
)

reason_results["recovery_rate"] = (
    reason_results["recovered"]
    /
    reason_results["payments"]
    * 100
)

print(reason_results)


# ========================================
# POLICY AGREEMENT
# ========================================

results_df["policy_match"] = (
    results_df["selected_action"]
    ==
    results_df["reason_policy_action"]
)


matches = results_df[
    "policy_match"
].sum()


match_rate = (
    matches / total * 100
)


print()
print("========================================")
print("HYBRID / POLICY AGREEMENT")
print("========================================")

print(
    "Agreement:",
    matches,
    "/",
    total
)

print(
    "Agreement rate:",
    round(match_rate, 2),
    "%"
)


# ========================================
# BASELINES
# ========================================

print()
print("========================================")
print("BASELINE COMPARISON")
print("========================================")


baseline_actions = [
    "WAIT",
    "RETRY",
    "REMINDER",
    "PAYMENT_LINK"
]


for action in baseline_actions:

    recovered_count = 0

    for _, payment in payments.iterrows():

        actual = df[
            (df["payment_id"]
             ==
             payment["payment_id"])
            &
            (df["action"]
             ==
             action)
        ]

        recovered_count += int(
            actual["recovered"].iloc[0]
        )


    rate = (
        recovered_count
        /
        total
        * 100
    )


    print(
        f"ALWAYS {action:<15}: "
        f"{recovered_count} recovered "
        f"({rate:.2f}%)"
    )


# ========================================
# REASON POLICY
# ========================================

reason_policy_recovered = 0


for _, payment in payments.iterrows():

    action = REASON_POLICY.get(
        payment["failure_reason"],
        "WAIT"
    )

    actual = df[
        (df["payment_id"]
         ==
         payment["payment_id"])
        &
        (df["action"]
         ==
         action)
    ]

    reason_policy_recovered += int(
        actual["recovered"].iloc[0]
    )


reason_policy_rate = (
    reason_policy_recovered
    /
    total
    * 100
)


print(
    f"REASON POLICY      : "
    f"{reason_policy_recovered} recovered "
    f"({reason_policy_rate:.2f}%)"
)


# ========================================
# HYBRID
# ========================================

print(
    f"HYBRID             : "
    f"{recovered} recovered "
    f"({recovery_rate:.2f}%)"
)


# ========================================
# IMPROVEMENT
# ========================================

best_baseline_rate = max(

    (
        df[
            df["action"] == action
        ]["recovered"].groupby(
            df[
                df["action"] == action
            ]["payment_id"]
        ).first().sum()
        / total
        * 100
    )
    for action in baseline_actions
)


print()
print("========================================")
print("FINAL IMPACT")
print("========================================")

print(
    "Best baseline:",
    round(best_baseline_rate, 2),
    "%"
)

print(
    "Hybrid:",
    round(recovery_rate, 2),
    "%"
)

print(
    "Improvement:",
    round(
        recovery_rate
        - best_baseline_rate,
        2
    ),
    "percentage points"
)


# ========================================
# SAVE
# ========================================

results_df.to_csv(
    "../data/hybrid_results.csv",
    index=False
)


print()
print("Saved:")
print("../data/hybrid_results.csv")

print()
print("========================================")
print("HYBRID EVALUATION COMPLETE")
print("========================================")