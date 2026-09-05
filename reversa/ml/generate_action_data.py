import numpy as np
import pandas as pd

np.random.seed(42)

NUM_CUSTOMERS = 10000
MONTHLY_AMOUNT = 499

customers = []

# ---------------------------------------------------------
# 1. GENERATE CUSTOMERS
# ---------------------------------------------------------

for i in range(NUM_CUSTOMERS):

    customer_id = f"C{i+1:05d}"

    profile = np.random.choice(
        [
            "reliable",
            "occasionally_late",
            "frequent_failure",
            "churn_risk"
        ],
        p=[0.50, 0.25, 0.15, 0.10]
    )

    if profile == "reliable":

        subscription_months = np.random.randint(6, 37)
        successful_payments = np.random.randint(6, 37)
        failed_payments = np.random.randint(0, 3)
        avg_delay = np.random.uniform(0, 1)
        reminder_success = np.random.uniform(0.70, 1.00)

    elif profile == "occasionally_late":

        subscription_months = np.random.randint(4, 25)
        successful_payments = np.random.randint(3, 20)
        failed_payments = np.random.randint(1, 5)
        avg_delay = np.random.uniform(1, 4)
        reminder_success = np.random.uniform(0.45, 0.80)

    elif profile == "frequent_failure":

        subscription_months = np.random.randint(3, 20)
        successful_payments = np.random.randint(1, 12)
        failed_payments = np.random.randint(3, 8)
        avg_delay = np.random.uniform(3, 8)
        reminder_success = np.random.uniform(0.15, 0.50)

    else:

        subscription_months = np.random.randint(1, 12)
        successful_payments = np.random.randint(0, 7)
        failed_payments = np.random.randint(4, 10)
        avg_delay = np.random.uniform(5, 15)
        reminder_success = np.random.uniform(0.05, 0.30)

    customers.append({
        "customer_id": customer_id,
        "profile": profile,
        "subscription_months": subscription_months,
        "successful_payments": successful_payments,
        "failed_payments": failed_payments,
        "avg_delay_days": round(avg_delay, 2),
        "reminder_success_rate": round(reminder_success, 2),
        "monthly_amount": MONTHLY_AMOUNT
    })


customers_df = pd.DataFrame(customers)


# ---------------------------------------------------------
# 2. SELECT CUSTOMERS WITH FAILED PAYMENTS
# ---------------------------------------------------------

failed_customers = customers_df.sample(
    n=3000,
    random_state=42
).copy()

failed_customers["payment_id"] = [
    f"P{i+1:06d}" for i in range(len(failed_customers))
]

# ---------------------------------------------------------
# 3. FAILURE REASONS
# ---------------------------------------------------------

failed_customers["failure_reason"] = np.random.choice(
    [
        "bank_error",
        "payment_method_issue",
        "insufficient_funds",
        "authentication_failed"
    ],
    size=len(failed_customers),
    p=[0.33, 0.27, 0.22, 0.18]
)

failed_customers["attempt_number"] = np.random.randint(
    1,
    4,
    size=len(failed_customers)
)


# ---------------------------------------------------------
# 4. ACTION-SPECIFIC RECOVERY MODEL
#
# Important:
# Different situations should have different best actions.
# This creates a meaningful decision problem.
# ---------------------------------------------------------

def base_recovery(row, action):

    reason = row["failure_reason"]
    profile = row["profile"]
    attempt = row["attempt_number"]
    reminder_rate = row["reminder_success_rate"]

    # Start from a neutral probability.
    probability = 0.50

    # -----------------------------------------------------
    # FAILURE REASON
    # -----------------------------------------------------

    if reason == "bank_error":

        if action == "RETRY":
            probability = 0.78

        elif action == "WAIT":
            probability = 0.64

        elif action == "REMINDER":
            probability = 0.56

        elif action == "PAYMENT_LINK":
            probability = 0.60


    elif reason == "insufficient_funds":

        if action == "WAIT":
            probability = 0.66

        elif action == "REMINDER":
            probability = 0.61

        elif action == "PAYMENT_LINK":
            probability = 0.58

        elif action == "RETRY":
            probability = 0.30


    elif reason == "payment_method_issue":

        if action == "PAYMENT_LINK":
            probability = 0.80

        elif action == "REMINDER":
            probability = 0.63

        elif action == "WAIT":
            probability = 0.48

        elif action == "RETRY":
            probability = 0.38


    elif reason == "authentication_failed":

        if action == "PAYMENT_LINK":
            probability = 0.84

        elif action == "REMINDER":
            probability = 0.69

        elif action == "WAIT":
            probability = 0.43

        elif action == "RETRY":
            probability = 0.25


    # -----------------------------------------------------
    # CUSTOMER BEHAVIOUR
    # -----------------------------------------------------

    if profile == "reliable":

        probability += 0.08

    elif profile == "occasionally_late":

        probability += 0.03

    elif profile == "frequent_failure":

        probability -= 0.06

    elif profile == "churn_risk":

        probability -= 0.12


    # -----------------------------------------------------
    # REMINDER RESPONSE
    # -----------------------------------------------------

    if action == "REMINDER":

        probability += (reminder_rate - 0.50) * 0.30


    # -----------------------------------------------------
    # ATTEMPT NUMBER
    # -----------------------------------------------------

    if attempt >= 3:

        if action == "RETRY":

            probability -= 0.15

        if action == "WAIT":

            probability -= 0.03

        if action == "REMINDER":

            probability -= 0.02


    # -----------------------------------------------------
    # RANDOM REAL-WORLD NOISE
    # -----------------------------------------------------

    probability += np.random.normal(0, 0.045)

    return float(np.clip(probability, 0.05, 0.95))


# ---------------------------------------------------------
# 5. GENERATE ACTION OUTCOMES
# ---------------------------------------------------------

actions = [
    "WAIT",
    "RETRY",
    "REMINDER",
    "PAYMENT_LINK"
]

records = []

for _, customer in failed_customers.iterrows():

    for action in actions:

        probability = base_recovery(
            customer,
            action
        )

        recovered = np.random.random() < probability

        records.append({

            "payment_id": customer["payment_id"],

            "customer_id": customer["customer_id"],

            "amount": customer["monthly_amount"],

            "profile": customer["profile"],

            "subscription_months": customer["subscription_months"],

            "successful_payments": customer["successful_payments"],

            "failed_payments": customer["failed_payments"],

            "avg_delay_days": customer["avg_delay_days"],

            "reminder_success_rate": customer["reminder_success_rate"],

            "failure_reason": customer["failure_reason"],

            "attempt_number": customer["attempt_number"],

            "action": action,

            "recovery_probability_true": round(
                probability,
                4
            ),

            "recovered": int(recovered)
        })


action_df = pd.DataFrame(records)


# ---------------------------------------------------------
# 6. SAVE DATASET
# ---------------------------------------------------------

action_df.to_csv(
    "../data/action_outcomes.csv",
    index=False
)


# ---------------------------------------------------------
# 7. DISPLAY DATA QUALITY
# ---------------------------------------------------------

print()
print("======================================")
print("NEW REVERSA DATASET GENERATED")
print("======================================")

print(f"Customers: {len(customers_df)}")

print(
    f"Failed payments: "
    f"{len(failed_customers)}"
)

print(
    f"Action records: "
    f"{len(action_df)}"
)

print()
print("Failure distribution:")

print(
    failed_customers["failure_reason"]
    .value_counts()
)

print()
print("Recovery by action:")

print(
    action_df.groupby("action")["recovered"]
    .mean()
    .sort_values(ascending=False)
)

print()
print("Recovery by failure reason:")

print(
    action_df
    .groupby(
        ["failure_reason", "action"]
    )["recovered"]
    .mean()
)

print()
print(
    "Saved: ../data/action_outcomes.csv"
)