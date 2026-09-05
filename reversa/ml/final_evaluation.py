import pandas as pd


# ========================================
# REVERSA FINAL SYSTEM EVALUATION
# ========================================

print("========================================")
print("REVERSA FINAL SYSTEM EVALUATION")
print("========================================")


# ========================================
# LOAD HYBRID RESULTS
# ========================================

df = pd.read_csv(
    "../data/hybrid_results.csv"
)

total_payments = len(df)

print()
print("Test payments:", total_payments)


# ========================================
# HYBRID PERFORMANCE
# ========================================

hybrid_recovered = int(
    df["recovered"].sum()
)

hybrid_rate = (
    hybrid_recovered /
    total_payments
) * 100


print()
print("========================================")
print("HYBRID PERFORMANCE")
print("========================================")

print(
    "Recovered:",
    hybrid_recovered
)

print(
    "Recovery rate:",
    round(hybrid_rate, 2),
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
    df["selected_action"]
    .value_counts()
)


# ========================================
# RECOVERY BY SELECTED ACTION
# ========================================

print()
print("========================================")
print("RECOVERY BY SELECTED ACTION")
print("========================================")

action_performance = (
    df.groupby("selected_action")["recovered"]
    .agg(
        payments="count",
        recovered="sum",
        recovery_rate="mean"
    )
)

action_performance["recovery_rate"] *= 100

print(
    action_performance.round(2)
)


# ========================================
# RECOVERY BY FAILURE REASON
# ========================================

print()
print("========================================")
print("RECOVERY BY FAILURE REASON")
print("========================================")

reason_performance = (
    df.groupby("failure_reason")["recovered"]
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
# POLICY AGREEMENT
# ========================================

agreement_count = int(
    df["policy_match"].sum()
)

agreement_rate = (
    agreement_count /
    total_payments
) * 100


print()
print("========================================")
print("HYBRID / REASON POLICY AGREEMENT")
print("========================================")

print(
    "Agreement:",
    agreement_count,
    "/",
    total_payments
)

print(
    "Agreement rate:",
    round(agreement_rate, 2),
    "%"
)


# ========================================
# BASELINE RESULTS
# ========================================

baselines = {
    "ALWAYS WAIT": 334,
    "ALWAYS RETRY": 270,
    "ALWAYS REMINDER": 410,
    "ALWAYS PAYMENT_LINK": 430,
    "REASON POLICY": 472,
    "HYBRID": hybrid_recovered
}


print()
print("========================================")
print("FINAL STRATEGY COMPARISON")
print("========================================")

for strategy, recovered in baselines.items():

    rate = (
        recovered /
        total_payments
    ) * 100

    print(
        f"{strategy:<24}: "
        f"{recovered} recovered "
        f"({rate:.2f}%)"
    )


# ========================================
# BEST FIXED BASELINE
# ========================================

fixed_baselines = {
    "ALWAYS WAIT":
        baselines["ALWAYS WAIT"],

    "ALWAYS RETRY":
        baselines["ALWAYS RETRY"],

    "ALWAYS REMINDER":
        baselines["ALWAYS REMINDER"],

    "ALWAYS PAYMENT_LINK":
        baselines["ALWAYS PAYMENT_LINK"]
}

best_baseline_name = max(
    fixed_baselines,
    key=fixed_baselines.get
)

best_baseline_recovered = (
    fixed_baselines[best_baseline_name]
)

best_baseline_rate = (
    best_baseline_recovered /
    total_payments
) * 100


# ========================================
# IMPROVEMENT
# ========================================

improvement = (
    hybrid_rate -
    best_baseline_rate
)


reason_policy_rate = (
    baselines["REASON POLICY"] /
    total_payments
) * 100


hybrid_vs_reason_policy = (
    hybrid_rate -
    reason_policy_rate
)


# ========================================
# FINAL IMPACT
# ========================================

print()
print("========================================")
print("FINAL IMPACT")
print("========================================")

print(
    "Best fixed baseline:",
    best_baseline_name
)

print(
    "Best baseline recovery:",
    round(
        best_baseline_rate,
        2
    ),
    "%"
)

print(
    "Reason-policy recovery:",
    round(
        reason_policy_rate,
        2
    ),
    "%"
)

print(
    "Hybrid recovery:",
    round(
        hybrid_rate,
        2
    ),
    "%"
)

print(
    "Improvement over best baseline:",
    round(
        improvement,
        2
    ),
    "percentage points"
)

print(
    "Hybrid improvement over reason policy:",
    round(
        hybrid_vs_reason_policy,
        2
    ),
    "percentage points"
)


# ========================================
# FINAL VERDICT
# ========================================

print()
print("========================================")
print("REVERSA FINAL VERDICT")
print("========================================")

if hybrid_rate > best_baseline_rate:

    print(
        "SUCCESS: REVERSA outperforms "
        "all fixed-action baselines."
    )

else:

    print(
        "WARNING: REVERSA does not outperform "
        "the best fixed-action baseline."
    )


print()
print("========================================")
print("EVALUATION COMPLETE")
print("========================================")