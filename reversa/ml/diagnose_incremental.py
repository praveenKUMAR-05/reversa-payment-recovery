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
# FEATURES
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


# --------------------------------
# SAME TEST SPLIT
# --------------------------------

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


# --------------------------------
# ACTIONS
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


# --------------------------------
# STORE DIAGNOSTICS
# --------------------------------

diagnostics = []


# --------------------------------
# PROCESS PAYMENTS
# --------------------------------

for payment_id, group in test_data.groupby(
    "payment_id"
):

    payment = group.iloc[0]

    candidates = []

    for action in ACTIONS:

        candidates.append({

            "amount": payment["amount"],
            "failure_reason": payment["failure_reason"],
            "attempt_number": payment["attempt_number"],
            "subscription_months": payment["subscription_months"],
            "successful_payments": payment["successful_payments"],
            "failed_payments": payment["failed_payments"],
            "avg_delay_days": payment["avg_delay_days"],
            "reminder_success_rate":
                payment["reminder_success_rate"],
            "action": action
        })


    candidate_df = pd.DataFrame(
        candidates
    )


    # --------------------------------
    # MODEL PREDICTIONS
    # --------------------------------

    probabilities = pipeline.predict_proba(
        candidate_df
    )[:, 1]

    candidate_df["predicted_probability"] = (
        probabilities
    )


    # --------------------------------
    # WAIT BASELINE
    # --------------------------------

    wait_probability = candidate_df.loc[
        candidate_df["action"] == "WAIT",
        "predicted_probability"
    ].iloc[0]


    candidate_df["incremental_recovery"] = (
        candidate_df["predicted_probability"]
        -
        wait_probability
    )


    candidate_df["action_cost"] = (
        candidate_df["action"].map(
            ACTION_COSTS
        )
    )


    candidate_df["net_incremental_value"] = (
        candidate_df["amount"]
        *
        candidate_df["incremental_recovery"]
        -
        candidate_df["action_cost"]
    )


    # --------------------------------
    # REVERSA DECISION
    # --------------------------------

    best = candidate_df.loc[
        candidate_df["net_incremental_value"].idxmax()
    ]

    if best["net_incremental_value"] <= 0:

        selected_action = "NO_ACTION"

    else:

        selected_action = best["action"]


    # --------------------------------
    # ACTUAL OUTCOMES
    # --------------------------------

    actuals = {}

    for action in ACTIONS:

        row = group[
            group["action"] == action
        ]

        actuals[action] = int(
            row.iloc[0]["recovered"]
        )


    # --------------------------------
    # RECORD
    # --------------------------------

    diagnostics.append({

        "payment_id": payment_id,

        "failure_reason":
            payment["failure_reason"],

        "selected_action":
            selected_action,

        "predicted_WAIT":
            candidate_df.loc[
                candidate_df["action"] == "WAIT",
                "predicted_probability"
            ].iloc[0],

        "predicted_RETRY":
            candidate_df.loc[
                candidate_df["action"] == "RETRY",
                "predicted_probability"
            ].iloc[0],

        "predicted_REMINDER":
            candidate_df.loc[
                candidate_df["action"] == "REMINDER",
                "predicted_probability"
            ].iloc[0],

        "predicted_PAYMENT_LINK":
            candidate_df.loc[
                candidate_df["action"] == "PAYMENT_LINK",
                "predicted_probability"
            ].iloc[0],

        "actual_WAIT":
            actuals["WAIT"],

        "actual_RETRY":
            actuals["RETRY"],

        "actual_REMINDER":
            actuals["REMINDER"],

        "actual_PAYMENT_LINK":
            actuals["PAYMENT_LINK"]
    })


diagnostics_df = pd.DataFrame(
    diagnostics
)


# --------------------------------
# OVERALL RESULTS
# --------------------------------

print()
print("========================================")
print("INCREMENTAL DECISION DIAGNOSTICS")
print("========================================")

print(
    "Payments:",
    len(diagnostics_df)
)


# --------------------------------
# SELECTED ACTION PERFORMANCE
# --------------------------------

def selected_actual(row):

    action = row["selected_action"]

    if action == "WAIT":
        return row["actual_WAIT"]

    if action == "RETRY":
        return row["actual_RETRY"]

    if action == "REMINDER":
        return row["actual_REMINDER"]

    if action == "PAYMENT_LINK":
        return row["actual_PAYMENT_LINK"]

    return row["actual_WAIT"]


diagnostics_df["selected_actual"] = (
    diagnostics_df.apply(
        selected_actual,
        axis=1
    )
)


print()
print("Action distribution:")

print(
    diagnostics_df["selected_action"]
    .value_counts()
)


print()
print("Recovery of selected action:")

print(
    diagnostics_df.groupby(
        "selected_action"
    )["selected_actual"]
    .mean()
)


# --------------------------------
# COMPARE AGAINST PAYMENT LINK
# --------------------------------

diagnostics_df["payment_link_better"] = (
    diagnostics_df["actual_PAYMENT_LINK"]
    >
    diagnostics_df["selected_actual"]
)


print()
print(
    "Cases where PAYMENT_LINK would "
    "have recovered but REVERSA did not:"
)

print(
    diagnostics_df[
        "payment_link_better"
    ].sum()
)


# --------------------------------
# SHOW MISTAKES
# --------------------------------

mistakes = diagnostics_df[
    (
        diagnostics_df["selected_action"]
        !=
        "PAYMENT_LINK"
    )
    &
    (
        diagnostics_df["actual_PAYMENT_LINK"]
        ==
        1
    )
    &
    (
        diagnostics_df["selected_actual"]
        ==
        0
    )
]


print()
print("========================================")
print("IMPORTANT MISSED OPPORTUNITIES")
print("========================================")

print(
    mistakes[
        [
            "payment_id",
            "failure_reason",
            "selected_action",
            "predicted_WAIT",
            "predicted_RETRY",
            "predicted_REMINDER",
            "predicted_PAYMENT_LINK",
            "actual_WAIT",
            "actual_RETRY",
            "actual_REMINDER",
            "actual_PAYMENT_LINK"
        ]
    ].head(30).to_string(
        index=False
    )
)