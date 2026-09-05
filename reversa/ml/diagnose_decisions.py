import pandas as pd

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

from sklearn.model_selection import GroupShuffleSplit

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


# --------------------------------
# CONFIGURATION
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

MIN_PROBABILITY = 0.60
MAX_RETRIES = 2


results = []


# --------------------------------
# PROCESS EACH PAYMENT
# --------------------------------

for payment_id, group in test_data.groupby(
    "payment_id"
):

    payment = group.iloc[0]

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


    candidate_df = pd.DataFrame(
        candidates
    )


    # --------------------------------
    # MODEL PREDICTIONS
    # --------------------------------

    probabilities = pipeline.predict_proba(
        candidate_df
    )[:, 1]


    candidate_df["probability"] = probabilities


    # --------------------------------
    # EXPECTED REVENUE
    # --------------------------------

    candidate_df["cost"] = (
        candidate_df["action"]
        .map(ACTION_COSTS)
    )


    candidate_df["expected_revenue"] = (
        candidate_df["amount"]
        *
        candidate_df["probability"]
        -
        candidate_df["cost"]
    )


    # --------------------------------
    # APPLY POLICY
    # --------------------------------

    allowed = []

    for _, row in candidate_df.iterrows():

        action = row["action"]

        probability = row["probability"]


        if probability < MIN_PROBABILITY:
            continue


        if (
            action == "RETRY"
            and payment["attempt_number"] >= MAX_RETRIES
        ):
            continue


        allowed.append(row)


    # --------------------------------
    # REVERSA ACTION
    # --------------------------------

    if len(allowed) == 0:

        selected = "NO_ACTION"

    else:

        allowed_df = pd.DataFrame(
            allowed
        )

        selected = allowed_df.loc[
            allowed_df["expected_revenue"].idxmax()
        ]["action"]


    # --------------------------------
    # ACTUAL OUTCOMES
    # --------------------------------

    actual_outcomes = (
        group
        .set_index("action")["recovered"]
        .to_dict()
    )


    # --------------------------------
    # ACTUAL BEST ACTION
    # --------------------------------

    best_actual = max(
        actual_outcomes,
        key=actual_outcomes.get
    )


    results.append({

        "payment_id": payment_id,

        "failure_reason":
            payment["failure_reason"],

        "attempt_number":
            payment["attempt_number"],

        "selected_action":
            selected,

        "actual_best_action":
            best_actual,

        "selected_recovered":
            actual_outcomes.get(
                selected,
                0
            ),

        "retry_actual":
            actual_outcomes.get(
                "RETRY",
                0
            ),

        "reminder_actual":
            actual_outcomes.get(
                "REMINDER",
                0
            ),

        "payment_link_actual":
            actual_outcomes.get(
                "PAYMENT_LINK",
                0
            ),

        "wait_actual":
            actual_outcomes.get(
                "WAIT",
                0
            )
    })


# --------------------------------
# RESULTS
# --------------------------------

results_df = pd.DataFrame(results)


# --------------------------------
# SUMMARY
# --------------------------------

print()
print("================================")
print("DECISION DIAGNOSTICS")
print("================================")

print(
    "Total payments:",
    len(results_df)
)

print()

print("REVERSA selected:")
print(
    results_df[
        "selected_action"
    ].value_counts()
)

print()

print("Actual outcome of selected action:")

print(
    results_df[
        "selected_recovered"
    ].value_counts()
)


# --------------------------------
# POLICY SELECTION ACCURACY
# --------------------------------

valid = results_df[
    results_df["selected_action"]
    != "NO_ACTION"
]


if len(valid) > 0:

    correct = (
        valid["selected_action"]
        ==
        valid["actual_best_action"]
    ).sum()

    accuracy = correct / len(valid)

    print()

    print(
        "Action-selection accuracy:",
        round(accuracy * 100, 2),
        "%"
    )


# --------------------------------
# FAILURE REASON ANALYSIS
# --------------------------------

print()
print("Recovery by failure reason:")

print(
    results_df.groupby(
        "failure_reason"
    )[
        "selected_recovered"
    ].mean()
)


# --------------------------------
# WORST REVERSA DECISIONS
# --------------------------------

wrong = results_df[
    (
        results_df["selected_recovered"] == 0
    )
    &
    (
        results_df["actual_best_action"]
        != "NO_ACTION"
    )
]


print()
print("Wrong decisions:", len(wrong))

print()

print(
    wrong[
        [
            "payment_id",
            "failure_reason",
            "attempt_number",
            "selected_action",
            "actual_best_action",
            "retry_actual",
            "reminder_actual",
            "payment_link_actual",
            "wait_actual"
        ]
    ].head(20).to_string(index=False)
)