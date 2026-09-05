import pandas as pd

from train_model import pipeline


# --------------------------------
# LOAD FAILED PAYMENTS
# --------------------------------

payments = pd.read_csv(
    "../data/failed_payments.csv"
)


# --------------------------------
# SELECT ONE PAYMENT
# --------------------------------

payment = payments.iloc[0]

print("Payment ID:", payment["payment_id"])
print("Customer ID:", payment["customer_id"])
print("Amount:", payment["amount"])
print("Failure reason:", payment["failure_reason"])


# --------------------------------
# CREATE FOUR POSSIBLE ACTIONS
# --------------------------------

actions = [
    "WAIT",
    "RETRY",
    "REMINDER",
    "PAYMENT_LINK"
]


candidates = []

for action in actions:

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


candidate_df["predicted_recovery_probability"] = probabilities


# --------------------------------
# RANK ACTIONS
# --------------------------------

candidate_df = candidate_df.sort_values(
    "predicted_recovery_probability",
    ascending=False
)


print()
print("ACTION RECOMMENDATIONS")
print("----------------------")

print(
    candidate_df[
        [
            "action",
            "predicted_recovery_probability"
        ]
    ].to_string(index=False)
)


# --------------------------------
# BEST ACTION
# --------------------------------

best_action = candidate_df.iloc[0]["action"]

best_probability = candidate_df.iloc[0][
    "predicted_recovery_probability"
]


print()
print("RECOMMENDED ACTION:", best_action)

print(
    "PREDICTED RECOVERY:",
    round(best_probability * 100, 2),
    "%"
)