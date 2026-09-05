import pandas as pd

from sklearn.model_selection import GroupShuffleSplit

from train_model import pipeline


# --------------------------------
# LOAD DATA
# --------------------------------

data = pd.read_csv(
    "../data/action_outcomes.csv"
)


# --------------------------------
# CREATE SAME TEST SET
# --------------------------------

X = data[
    [
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
]

y = data["recovered"]


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


test_data = data.iloc[test_idx].copy()


print("Test action records:", len(test_data))

print(
    "Test payment cases:",
    test_data["payment_id"].nunique()
)

# --------------------------------
# REVERSA CONFIGURATION
# --------------------------------

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

MIN_RECOVERY_PROBABILITY = 0.30
MAX_RETRIES = 2


# --------------------------------
# STORE RESULTS
# --------------------------------

results = []


# --------------------------------
# PROCESS EACH PAYMENT
# --------------------------------

for payment_id, payment_group in test_data.groupby(
    "payment_id"
):

    # Take one row because customer/payment
    # information is the same for all actions
    payment = payment_group.iloc[0]


    # --------------------------------
    # CREATE FOUR ACTION CANDIDATES
    # --------------------------------

    candidates = []

    for action in ACTIONS:

        candidates.append({

            "amount": payment["amount"],

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

            "action": action
        })


    candidate_df = pd.DataFrame(candidates)


    # --------------------------------
    # PREDICT RECOVERY PROBABILITY
    # --------------------------------

    probabilities = pipeline.predict_proba(
        candidate_df
    )[:, 1]


    candidate_df["recovery_probability"] = (
        probabilities
    )

    # --------------------------------
    # CALCULATE INCREMENTAL VALUE
    # --------------------------------

    candidate_df["action_cost"] = (
        candidate_df["action"].map(
            ACTION_COSTS
        )
    )


    # WAIT represents natural recovery.
    # We compare every intervention against WAIT.
    wait_probability = candidate_df.loc[
        candidate_df["action"] == "WAIT",
        "recovery_probability"
    ].iloc[0]


    # Additional probability created
    # by each action over natural recovery.
    candidate_df["incremental_recovery"] = (
        candidate_df["recovery_probability"]
        -
        wait_probability
    )


    # Additional money expected from
    # choosing this action instead of WAIT.
    candidate_df["incremental_revenue"] = (
        candidate_df["amount"]
        *
        candidate_df["incremental_recovery"]
    )


    # Subtract the intervention cost.
    candidate_df["net_incremental_value"] = (
        candidate_df["incremental_revenue"]
        -
        candidate_df["action_cost"]
    )
    


    # --------------------------------
    # APPLY POLICY
    # --------------------------------

    allowed_actions = []

    for _, row in candidate_df.iterrows():

        action = row["action"]

        probability = row[
            "recovery_probability"
        ]

        attempt_number = payment[
            "attempt_number"
        ]


        allowed = True


        # Recovery probability rule
        if probability < MIN_RECOVERY_PROBABILITY:

            allowed = False


        # Retry limit rule
        if (
            action == "RETRY"
            and attempt_number >= MAX_RETRIES
        ):

            allowed = False


        if allowed:

            allowed_actions.append(row)


    # --------------------------------
    # HANDLE NO SAFE ACTION
    # --------------------------------

    if len(allowed_actions) == 0:

        results.append({

            "payment_id": payment_id,

            "amount": payment["amount"],

            "selected_action": "NO_ACTION",

            "predicted_probability": 0,

            "expected_revenue": 0,

            "actual_recovered": 0
        })

        continue


    # --------------------------------
    # SELECT BEST ACTION
    # --------------------------------

    allowed_df = pd.DataFrame(
        allowed_actions
    )

    # --------------------------------
    # BOUNDED ECONOMIC DECISION
    # --------------------------------

    best_candidate = allowed_df.loc[
        allowed_df["net_incremental_value"].idxmax()
    ]


    # If no intervention creates positive
    # incremental value, do nothing.
    if best_candidate["net_incremental_value"] <= 0:

        results.append({

            "payment_id": payment_id,

            "amount": payment["amount"],

            "selected_action": "NO_ACTION",

            "predicted_probability":
                wait_probability,

            "expected_revenue": 0,

            "actual_recovered":
                int(
                    payment_group[
                        payment_group["action"] == "WAIT"
                    ].iloc[0]["recovered"]
                )
        })

        continue


    best = allowed_df.loc[
        allowed_df["net_incremental_value"].idxmax()
    ]


    # --------------------------------
    # FIND ACTUAL OUTCOME
    # --------------------------------

    actual_row = payment_group[
        payment_group["action"]
        ==
        best["action"]
    ]


    if len(actual_row) > 0:

        actual_recovered = int(
            actual_row.iloc[0]["recovered"]
        )

    else:

        actual_recovered = 0


    # --------------------------------
    # STORE RESULT
    # --------------------------------

    results.append({

        "payment_id": payment_id,

        "amount": payment["amount"],

        "selected_action":
            best["action"],

        "predicted_probability":
            best["recovery_probability"],

        "expected_revenue":
            best["net_incremental_value"],

        "actual_recovered":
            actual_recovered
    })


# --------------------------------
# CREATE RESULTS DATAFRAME
# --------------------------------

results_df = pd.DataFrame(results)


# --------------------------------
# DISPLAY SUMMARY
# --------------------------------

print()
print("================================")
print("REVERSA BATCH RESULTS")
print("================================")

print(
    "Payments processed:",
    len(results_df)
)

print()
print("Action distribution:")
print(
    results_df["selected_action"]
    .value_counts()
)

print()
print("Actual recovered payments:")

print(
    results_df["actual_recovered"]
    .sum()
)

# --------------------------------
# BASELINE COMPARISON
# --------------------------------

TOTAL_PAYMENTS = results_df["payment_id"].nunique()


def evaluate_baseline(action):

    action_rows = test_data[
        test_data["action"] == action
    ]

    recovered = int(
        action_rows["recovered"].sum()
    )

    recovery_rate = (
        recovered /
        TOTAL_PAYMENTS
        * 100
    )

    cost_per_action = ACTION_COSTS[action]

    total_cost = (
        TOTAL_PAYMENTS *
        cost_per_action
    )

    gross_revenue = (
        recovered *
        test_data["amount"].iloc[0]
    )

    net_revenue = (
        gross_revenue -
        total_cost
    )

    return {
        "strategy": f"ALWAYS {action}",
        "recovered": recovered,
        "recovery_rate": recovery_rate,
        "cost": total_cost,
        "gross_revenue": gross_revenue,
        "net_revenue": net_revenue
    }


# --------------------------------
# CALCULATE SIMPLE BASELINES
# --------------------------------

baseline_results = []

for action in ACTIONS:

    baseline_results.append(
        evaluate_baseline(action)
    )


# --------------------------------
# CALCULATE REVERSA ECONOMICS
# --------------------------------

reversa_recovered = int(
    results_df["actual_recovered"].sum()
)


# Cost of the action actually selected
reversa_cost = 0

for _, row in results_df.iterrows():

    action = row["selected_action"]

    if action in ACTION_COSTS:

        reversa_cost += ACTION_COSTS[action]


gross_revenue = (
    reversa_recovered *
    results_df["amount"].iloc[0]
)


reversa_net_revenue = (
    gross_revenue -
    reversa_cost
)


reversa_rate = (
    reversa_recovered /
    TOTAL_PAYMENTS
    * 100
)


baseline_results.append({

    "strategy": "REVERSA",

    "recovered":
        reversa_recovered,

    "recovery_rate":
        reversa_rate,

    "cost":
        reversa_cost,

    "gross_revenue":
        gross_revenue,

    "net_revenue":
        reversa_net_revenue
})


# --------------------------------
# DISPLAY COMPARISON
# --------------------------------

comparison_df = pd.DataFrame(
    baseline_results
)


print()
print("========================================")
print("REVERSA ECONOMIC COMPARISON")
print("========================================")

print(
    comparison_df[
        [
            "strategy",
            "recovered",
            "recovery_rate",
            "cost",
            "gross_revenue",
            "net_revenue"
        ]
    ].to_string(
        index=False
    )
)


# --------------------------------
# FIND BEST BASELINE
# --------------------------------

baseline_only = comparison_df[
    comparison_df["strategy"] != "REVERSA"
]


best_baseline = baseline_only.loc[
    baseline_only["net_revenue"].idxmax()
]


print()
print("========================================")
print("FINAL COMPARISON")
print("========================================")

print(
    "Best baseline:",
    best_baseline["strategy"]
)

print(
    "Baseline net revenue: ₹",
    round(
        best_baseline["net_revenue"],
        2
    )
)

print(
    "REVERSA net revenue: ₹",
    round(
        reversa_net_revenue,
        2
    )
)


improvement = (
    reversa_net_revenue -
    best_baseline["net_revenue"]
)


print(
    "REVERSA improvement: ₹",
    round(
        improvement,
        2
    )
)