import numpy as np
import pandas as pd

np.random.seed(44)

# Load failed payment cases
payments = pd.read_csv("failed_payments.csv")

actions = [
    "WAIT",
    "RETRY",
    "REMINDER",
    "PAYMENT_LINK"
]

records = []


def calculate_recovery_probability(row, action):

    # Start with a baseline probability
    probability = 0.30

    # --------------------------------
    # CUSTOMER BEHAVIOR
    # --------------------------------

    failure_rate = (
        row["failed_payments"] /
        row["subscription_months"]
    )

    # Customers with better payment history
    # are generally easier to recover
    probability += (1 - failure_rate) * 0.25

    # --------------------------------
    # ACTION EFFECT
    # --------------------------------

    if action == "WAIT":

        probability += 0.05

        # Waiting works better for customers
        # who usually pay with a small delay
        if row["avg_delay_days"] <= 2:
            probability += 0.15

    elif action == "RETRY":

        probability += 0.15

        # Retry works particularly well
        # for bank/network problems
        if row["failure_reason"] == "bank_error":
            probability += 0.20

        # Immediate retry is less useful
        # for insufficient funds
        if row["failure_reason"] == "insufficient_funds":
            probability -= 0.10

    elif action == "REMINDER":

        # Previous reminder behavior matters
        probability += (
            row["reminder_success_rate"] * 0.35
        )

        # Reminder is useful for customers
        # who tend to pay late
        if row["avg_delay_days"] >= 1:
            probability += 0.10

    elif action == "PAYMENT_LINK":

        probability += 0.10

        # Useful when the payment method itself
        # may be the problem
        if row["failure_reason"] == "payment_method_issue":
            probability += 0.20

        # Also useful for authentication problems
        if row["failure_reason"] == "authentication_failed":
            probability += 0.15

    # --------------------------------
    # HIGH FAILURE HISTORY
    # --------------------------------

    if failure_rate > 0.40:
        probability -= 0.15

    # --------------------------------
    # ADD SMALL RANDOM VARIATION
    # --------------------------------

    probability += np.random.normal(0, 0.03)

    # Keep probability between 5% and 95%
    probability = np.clip(probability, 0.05, 0.95)

    return probability


# --------------------------------
# GENERATE OUTCOMES
# --------------------------------

for _, row in payments.iterrows():

    for action in actions:

        probability = calculate_recovery_probability(
            row,
            action
        )

        # Generate actual outcome
        recovered = int(
            np.random.random() < probability
        )

        # Time taken to recover
        if recovered:

            if action == "WAIT":
                recovery_time = np.random.uniform(1, 3)

            elif action == "RETRY":
                recovery_time = np.random.uniform(0.1, 1)

            elif action == "REMINDER":
                recovery_time = np.random.uniform(0.2, 2)

            else:
                recovery_time = np.random.uniform(0.1, 1.5)

        else:
            recovery_time = np.nan

        records.append({

            "payment_id": row["payment_id"],

            "customer_id": row["customer_id"],

            "amount": row["amount"],

            "failure_reason": row["failure_reason"],

            "attempt_number": row["attempt_number"],

            "subscription_months": row["subscription_months"],

            "successful_payments": row["successful_payments"],

            "failed_payments": row["failed_payments"],

            "avg_delay_days": row["avg_delay_days"],

            "reminder_success_rate": row["reminder_success_rate"],

            "action": action,

            "recovery_probability_ground_truth": round(
                probability, 4
            ),

            "recovered": recovered,

            "time_to_recovery_days": (
                round(recovery_time, 2)
                if recovered
                else np.nan
            )
        })


# --------------------------------
# CREATE DATAFRAME
# --------------------------------

df = pd.DataFrame(records)

# Save dataset
df.to_csv(
    "action_outcomes.csv",
    index=False
)


# --------------------------------
# BASIC VALIDATION
# --------------------------------

print("Failed payment cases:", len(payments))

print(
    "Action-outcome records:",
    len(df)
)

print()

print("Expected records:")
print(len(payments) * 4)

print()

print("Action distribution:")
print(df["action"].value_counts())

print()

print("Recovery rate by action:")

print(
    df.groupby("action")["recovered"]
    .mean()
    .sort_values(ascending=False)
)

print()

print("First 10 records:")
print(df.head(10))