import pandas as pd


# ========================================
# REVERSA REASON-AWARE POLICY
# ========================================

print("========================================")
print("REVERSA REASON-AWARE POLICY")
print("========================================")


# ========================================
# LOAD HELD-OUT DATA
# ========================================

df = pd.read_csv(
    "../data/test_action_outcomes.csv"
)


# ========================================
# POLICY
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
# SELECT ACTION
# ========================================

df["selected_action"] = (
    df["failure_reason"]
    .map(REASON_POLICY)
)


# ========================================
# KEEP SELECTED ACTION RECORD
# ========================================

selected = df[
    df["action"] ==
    df["selected_action"]
].copy()


# ========================================
# PERFORMANCE
# ========================================

recovered = selected["recovered"].sum()

payments = selected["payment_id"].nunique()

recovery_rate = (
    recovered
    /
    payments
    *
    100
)


print()
print("========================================")
print("POLICY PERFORMANCE")
print("========================================")

print(
    "Payments:",
    payments
)

print(
    "Recovered:",
    recovered
)

print(
    "Recovery rate:",
    round(
        recovery_rate,
        2
    ),
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
    selected["selected_action"]
    .value_counts()
)


# ========================================
# PERFORMANCE BY FAILURE REASON
# ========================================

print()
print("========================================")
print("PERFORMANCE BY FAILURE REASON")
print("========================================")

reason_performance = (
    selected
    .groupby("failure_reason")["recovered"]
    .agg(
        payments="count",
        recovered="sum",
        recovery_rate="mean"
    )
)

reason_performance["recovery_rate"] *= 100

print(
    reason_performance.round(2)
)


# ========================================
# COMPARE WITH BASELINES
# ========================================

print()
print("========================================")
print("BASELINE COMPARISON")
print("========================================")


baselines = {}


for action in [
    "WAIT",
    "RETRY",
    "REMINDER",
    "PAYMENT_LINK"
]:

    action_df = df[
        df["action"] == action
    ]

    recovered_count = (
        action_df
        ["recovered"]
        .sum()
    )

    rate = (
        recovered_count
        /
        payments
        *
        100
    )

    baselines[action] = {
        "recovered":
            recovered_count,

        "recovery_rate":
            rate
    }


for action, result in baselines.items():

    print(
        f"ALWAYS {action:14s}"
        f": {result['recovered']:3d}"
        f" recovered"
        f" ({result['recovery_rate']:.2f}%)"
    )


print(
    f"REVERSA REASON POLICY"
    f": {recovered:3d}"
    f" recovered"
    f" ({recovery_rate:.2f}%)"
)


# ========================================
# IMPROVEMENT
# ========================================

best_baseline_rate = max(
    result["recovery_rate"]
    for result in baselines.values()
)

improvement = (
    recovery_rate
    -
    best_baseline_rate
)


print()
print("========================================")
print("POLICY IMPACT")
print("========================================")

print(
    "Best baseline recovery:",
    round(
        best_baseline_rate,
        2
    ),
    "%"
)

print(
    "REVERSA recovery:",
    round(
        recovery_rate,
        2
    ),
    "%"
)

print(
    "Improvement:",
    round(
        improvement,
        2
    ),
    "percentage points"
)


print()
print("========================================")
print("POLICY ANALYSIS COMPLETE")
print("========================================")