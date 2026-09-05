import numpy as np
import pandas as pd

np.random.seed(42)

NUM_CUSTOMERS = 10000

customers = []

for i in range(NUM_CUSTOMERS):

    customer_id = f"C{i+1:05d}"

    # Assign a behavioral profile
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
        failure_rate = np.random.uniform(0.01, 0.08)
        avg_delay = np.random.uniform(0, 1.0)
        reminder_success = np.random.uniform(0.70, 1.00)

    elif profile == "occasionally_late":

        subscription_months = np.random.randint(4, 25)
        failure_rate = np.random.uniform(0.08, 0.20)
        avg_delay = np.random.uniform(1, 4)
        reminder_success = np.random.uniform(0.45, 0.80)

    elif profile == "frequent_failure":

        subscription_months = np.random.randint(3, 20)
        failure_rate = np.random.uniform(0.20, 0.40)
        avg_delay = np.random.uniform(3, 8)
        reminder_success = np.random.uniform(0.15, 0.50)

    else:

        subscription_months = np.random.randint(1, 12)
        failure_rate = np.random.uniform(0.35, 0.60)
        avg_delay = np.random.uniform(5, 15)
        reminder_success = np.random.uniform(0.05, 0.30)

    # Total monthly payment attempts
    total_payments = subscription_months

    # Generate number of failed payments
    failed_payments = np.random.binomial(
        total_payments,
        failure_rate
    )

    # Remaining payments were successful
    successful_payments = total_payments - failed_payments

    customers.append({
        "customer_id": customer_id,
        "profile": profile,
        "subscription_months": subscription_months,
        "successful_payments": successful_payments,
        "failed_payments": failed_payments,
        "avg_delay_days": round(avg_delay, 2),
        "reminder_success_rate": round(reminder_success, 2),
        "monthly_amount": 499
    })

df = pd.DataFrame(customers)

df.to_csv("customers.csv", index=False)

print("Generated customers:", len(df))
print()
print(df.head())
print()
print("Profile distribution:")
print(df["profile"].value_counts())