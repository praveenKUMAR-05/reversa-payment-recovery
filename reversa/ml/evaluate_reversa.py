import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from policy_engine import check_policy


# ========================================
# REVERSA FINAL EVALUATION
# ========================================

print("=" * 40)
print("REVERSA FINAL EVALUATION")
print("=" * 40)


# ========================================
# LOAD MODEL
# ========================================

pipeline = joblib.load(
    "../data/reversa_model.pkl"
)


# ========================================
# LOAD HELD-OUT DATA
# ========================================

data = pd.read_csv(
    "../data/test_action_outcomes.csv"
)

payment_ids = pd.read_csv(
    "../data/test_payment_ids.csv"
)


# ========================================
# MODEL FEATURES
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

X = data[features]
y = data["recovered"]


# ========================================
# MODEL PERFORMANCE
# ========================================

probabilities = pipeline.predict_proba(X)[:, 1]

predictions = (
    probabilities >= 0.50
).astype(int)


accuracy = accuracy_score(
    y,
    predictions
)

precision = precision_score(
    y,
    predictions
)

recall = recall_score(
    y,
    predictions
)

f1 = f1_score(
    y,
    predictions
)

roc_auc = roc_auc_score(
    y,
    probabilities
)


print()
print("MODEL PERFORMANCE")
print("----------------------------------------")

print(
    f"Accuracy       : {accuracy * 100:.2f}%"
)

print(
    f"Precision      : {precision * 100:.2f}%"
)

print(
    f"Recall         : {recall * 100:.2f}%"
)

print(
    f"F1 Score       : {f1 * 100:.2f}%"
)

print(
    f"ROC-AUC        : {roc_auc * 100:.2f}%"
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
# GROUP TEST DATA BY PAYMENT
# ========================================

payments = (
    data
    .groupby("payment_id")
    .first()
    .reset_index()
)


# ========================================
# SIMULATE REVERSA
# ========================================

results = []


for _, payment in payments.iterrows():

    candidates = []

    for action in ACTION_COSTS.keys():

        row = {
            "amount": payment["amount"],
            "failure_reason": payment["failure_reason"],
            "attempt_number": payment["attempt_number"],
            "subscription_months": payment["subscription_months"],
            "successful_payments": payment["successful_payments"],
            "failed_payments": payment["failed_payments"],
            "avg_delay_days": payment["avg_delay_days"],
            "reminder_success_rate": payment["reminder_success_rate"],
            "action": action
        }

        candidates.append(row)


    candidate_df = pd.DataFrame(
        candidates
    )


    # Predict recovery probability

    candidate_df[
        "recovery_probability"
    ] = pipeline.predict_proba(
        candidate_df[features]
    )[:, 1]


    # Policy filtering

    allowed = []

    for _, row in candidate_df.iterrows():

        policy = check_policy(
            action=row["action"],
            recovery_probability=row[
                "recovery_probability"
            ],
            attempt_number=payment[
                "attempt_number"
            ],
            payment_recovered=False,
            amount=payment["amount"]
        )

        if policy["allowed"]:
            allowed.append(row)


    # No allowed action

    if not allowed:

        selected_action = "NO_ACTION"

    else:

        allowed_df = pd.DataFrame(
            allowed
        )


        wait_probability = (
            candidate_df.loc[
                candidate_df["action"] == "WAIT",
                "recovery_probability"
            ].iloc[0]
        )


        allowed_df[
            "incremental_recovery"
        ] = (
            allowed_df[
                "recovery_probability"
            ]
            -
            wait_probability
        )


        allowed_df[
            "incremental_revenue"
        ] = (
            allowed_df[
                "incremental_recovery"
            ]
            *
            payment["amount"]
        )


        allowed_df[
            "action_cost"
        ] = allowed_df[
            "action"
        ].map(ACTION_COSTS)


        allowed_df[
            "net_incremental_value"
        ] = (
            allowed_df[
                "incremental_revenue"
            ]
            -
            allowed_df[
                "action_cost"
            ]
        )


        best = allowed_df.sort_values(
            "net_incremental_value",
            ascending=False
        ).iloc[0]


        if best[
            "net_incremental_value"
        ] <= 0:

            selected_action = "NO_ACTION"

        else:

            selected_action = best[
                "action"
            ]


    # ====================================
    # FIND ACTUAL OUTCOME
    # ====================================

    actual_row = data[
        (data["payment_id"] == payment["payment_id"])
        &
        (data["action"] == selected_action)
    ]


    if len(actual_row) > 0:

        recovered = int(
            actual_row[
                "recovered"
            ].iloc[0]
        )

    else:

        recovered = 0


    results.append({

        "payment_id":
            payment["payment_id"],

        "selected_action":
            selected_action,

        "recovered":
            recovered,

        "amount":
            payment["amount"]
    })


# ========================================
# RESULTS DATAFRAME
# ========================================

results_df = pd.DataFrame(
    results
)


# ========================================
# REVERSA PERFORMANCE
# ========================================

reversa_recovered = results_df[
    "recovered"
].sum()

total_payments = len(
    results_df
)

reversa_recovery_rate = (
    reversa_recovered
    /
    total_payments
)


# ========================================
# BUSINESS REVENUE
# ========================================

results_df[
    "revenue"
] = (
    results_df[
        "amount"
    ]
    *
    results_df[
        "recovered"
    ]
)


results_df[
    "cost"
] = results_df[
    "selected_action"
].map(ACTION_COSTS).fillna(0)


reversa_gross = results_df[
    "revenue"
].sum()

reversa_cost = results_df[
    "cost"
].sum()

reversa_net = (
    reversa_gross
    -
    reversa_cost
)


# ========================================
# BASELINES
# ========================================

baseline_actions = [
    "WAIT",
    "RETRY",
    "REMINDER",
    "PAYMENT_LINK"
]


baseline_results = []


for action in baseline_actions:

    action_data = data[
        data["action"] == action
    ]


    recovered = action_data[
        "recovered"
    ].sum()


    # One record per payment/action,
    # therefore divide by number of payments

    recovery_rate = (
        recovered
        /
        total_payments
    )


    gross = (
        action_data[
            "amount"
        ]
        *
        action_data[
            "recovered"
        ]
    ).sum()


    cost = (
        action_data[
            "recovered"
        ].count()
        *
        ACTION_COSTS[action]
    )


    net = (
        gross
        -
        cost
    )


    baseline_results.append({

        "strategy":
            f"ALWAYS {action}",

        "recovered":
            recovered,

        "recovery_rate":
            recovery_rate,

        "cost":
            cost,

        "gross_revenue":
            gross,

        "net_revenue":
            net
    })


baseline_results.append({

    "strategy":
        "REVERSA",

    "recovered":
        reversa_recovered,

    "recovery_rate":
        reversa_recovery_rate,

    "cost":
        reversa_cost,

    "gross_revenue":
        reversa_gross,

    "net_revenue":
        reversa_net
})


comparison = pd.DataFrame(
    baseline_results
)


# ========================================
# FINAL BUSINESS COMPARISON
# ========================================

baseline_only = comparison[
    comparison["strategy"] != "REVERSA"
]


best_baseline = baseline_only.loc[
    baseline_only[
        "net_revenue"
    ].idxmax()
]


improvement = (
    reversa_net
    -
    best_baseline[
        "net_revenue"
    ]
)


recovery_improvement = (
    reversa_recovery_rate
    -
    best_baseline[
        "recovery_rate"
    ]
)


# ========================================
# PRINT BUSINESS RESULTS
# ========================================

print()
print("=" * 40)
print("BUSINESS PERFORMANCE")
print("=" * 40)

print(
    f"Test Payments  : {total_payments}"
)

print()
print(
    f"REVERSA Recovered : {reversa_recovered}"
)

print(
    f"REVERSA Recovery  : "
    f"{reversa_recovery_rate * 100:.2f}%"
)

print(
    f"REVERSA Net Revenue : "
    f"₹ {reversa_net:,.0f}"
)


# ========================================
# COMPARISON TABLE
# ========================================

print()
print("=" * 40)
print("STRATEGY COMPARISON")
print("=" * 40)

display = comparison.copy()

display[
    "recovery_rate"
] = (
    display[
        "recovery_rate"
    ] * 100
).round(2)


print(
    display.to_string(
        index=False
    )
)


# ========================================
# FINAL RESULT
# ========================================

print()
print("=" * 40)
print("REVERSA IMPACT")
print("=" * 40)

print(
    "Best baseline      :",
    best_baseline["strategy"]
)

print(
    "Baseline net revenue: ₹",
    f"{best_baseline['net_revenue']:,.0f}"
)

print(
    "REVERSA net revenue : ₹",
    f"{reversa_net:,.0f}"
)

print(
    "Net revenue improvement: ₹",
    f"{improvement:,.0f}"
)

print(
    "Recovery improvement:",
    f"{recovery_improvement * 100:+.2f}",
    "percentage points"
)

print()
print("=" * 40)
print("EVALUATION COMPLETE")
print("=" * 40)