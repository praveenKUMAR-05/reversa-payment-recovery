import pandas as pd


# ========================================
# REVERSA ACTION ANALYSIS
# ========================================

print("========================================")
print("REVERSA ACTION ANALYSIS")
print("========================================")


# ========================================
# LOAD HELD-OUT DATA
# ========================================

df = pd.read_csv(
    "../data/test_action_outcomes.csv"
)

print()
print("Test action records:", len(df))
print(
    "Unique payments:",
    df["payment_id"].nunique()
)


# ========================================
# 1. ACTUAL RECOVERY BY ACTION
# ========================================

print()
print("========================================")
print("ACTUAL RECOVERY BY ACTION")
print("========================================")

action_results = (
    df.groupby("action")["recovered"]
    .agg(
        records="count",
        recovered="sum",
        recovery_rate="mean"
    )
    .sort_values(
        "recovery_rate",
        ascending=False
    )
)

action_results["recovery_rate"] *= 100

print(
    action_results.round(2)
)


# ========================================
# 2. ACTUAL RECOVERY BY FAILURE REASON
# ========================================

print()
print("========================================")
print("RECOVERY BY FAILURE REASON")
print("========================================")

reason_results = (
    df.groupby(
        ["failure_reason", "action"]
    )["recovered"]
    .mean()
    .unstack()
)

reason_results *= 100

print(
    reason_results.round(2)
)


# ========================================
# 3. FIND BEST ACTUAL ACTION
# ========================================

print()
print("========================================")
print("BEST ACTUAL ACTION BY FAILURE REASON")
print("========================================")

best_action = (
    df.groupby(
        ["failure_reason", "action"]
    )["recovered"]
    .mean()
    .reset_index()
)

best_action = best_action.loc[
    best_action.groupby(
        "failure_reason"
    )["recovered"].idxmax()
]

best_action = best_action.sort_values(
    "failure_reason"
)

best_action["recovered"] *= 100

print(
    best_action.to_string(
        index=False
    )
)


# ========================================
# 4. MODEL PREDICTION ANALYSIS
# ========================================

print()
print("========================================")
print("MODEL PREDICTION ANALYSIS")
print("========================================")


# These columns are created by the
# diagnostic / decision process if available.

prediction_columns = [
    "predicted_WAIT",
    "predicted_RETRY",
    "predicted_REMINDER",
    "predicted_PAYMENT_LINK"
]

available = [
    c for c in prediction_columns
    if c in df.columns
]

if len(available) == 4:

    print()
    print("Average predicted probability:")
    print("--------------------------------")

    prediction_summary = (
        df[available]
        .mean()
        .sort_values(
            ascending=False
        )
    )

    print(
        prediction_summary.round(4)
    )

else:

    print(
        "Prediction probability columns "
        "are not present in test_action_outcomes.csv."
    )


# ========================================
# 5. THEORETICAL ORACLE POLICY
# ========================================

print()
print("========================================")
print("ORACLE POLICY")
print("========================================")

print(
    "The oracle chooses the action with"
)
print(
    "the highest ACTUAL recovery rate"
)
print(
    "for each failure reason."
)


# Map failure reason -> best action

oracle_map = (
    best_action
    .set_index("failure_reason")["action"]
    .to_dict()
)

print()
print("Oracle policy:")

for reason, action in oracle_map.items():

    print(
        f"{reason:25s} -> {action}"
    )


# ========================================
# 6. ESTIMATE ORACLE PERFORMANCE
# ========================================

payment_df = (
    df.groupby(
        "payment_id"
    )
    .first()
    .reset_index()
)

# Each payment has four action records.
# Calculate what would happen if the
# best action for its failure reason
# was always selected.

oracle_recovered = 0

for _, payment in payment_df.iterrows():

    reason = payment[
        "failure_reason"
    ]

    action = oracle_map[
        reason
    ]

    result = df[
        (df["payment_id"] == payment["payment_id"])
        &
        (df["action"] == action)
    ]

    if len(result) > 0:

        oracle_recovered += int(
            result.iloc[0]["recovered"]
        )


oracle_rate = (
    oracle_recovered
    /
    len(payment_df)
    *
    100
)


print()
print("Oracle recovered:", oracle_recovered)
print(
    "Oracle recovery rate:",
    round(oracle_rate, 2),
    "%"
)


# ========================================
# 7. ACTION COUNTS
# ========================================

print()
print("========================================")
print("DATASET ACTION COUNTS")
print("========================================")

print(
    df["action"].value_counts()
)


# ========================================
# FINAL MESSAGE
# ========================================

print()
print("========================================")
print("ANALYSIS COMPLETE")
print("========================================")

print()
print(
    "Next step: compare the Oracle policy"
)
print(
    "against the current REVERSA policy."
)