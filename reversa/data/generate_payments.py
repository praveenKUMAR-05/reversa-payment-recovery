import numpy as np
import pandas as pd

np.random.seed(43)

# Load our customer dataset
customers = pd.read_csv("customers.csv")

payments = []

for _, customer in customers.iterrows():

    # Simulate the customer's current monthly payment
    payment_id = f"P{len(payments) + 1:06d}"

    customer_id = customer["customer_id"]
    amount = customer["monthly_amount"]

    # Use the customer's historical failure behavior
    historical_failure_rate = (
        customer["failed_payments"] /
        customer["subscription_months"]
    )

    # Slight variation for the current billing cycle
    current_failure_probability = np.clip(
        historical_failure_rate * np.random.uniform(0.8, 1.2),
        0.01,
        0.90
    )

    # Decide whether the current payment fails
    payment_failed = np.random.random() < current_failure_probability

    if payment_failed:

        # Select a reason for the failure
        failure_reason = np.random.choice(
            [
                "insufficient_funds",
                "bank_error",
                "authentication_failed",
                "payment_method_issue"
            ],
            p=[0.45, 0.25, 0.15, 0.15]
        )

        payments.append({
            "payment_id": payment_id,
            "customer_id": customer_id,
            "amount": amount,
            "failure_reason": failure_reason,
            "attempt_number": 1,
            "subscription_months": customer["subscription_months"],
            "successful_payments": customer["successful_payments"],
            "failed_payments": customer["failed_payments"],
            "avg_delay_days": customer["avg_delay_days"],
            "reminder_success_rate": customer["reminder_success_rate"]
        })


# Convert to DataFrame
df = pd.DataFrame(payments)

# Save failed payments
df.to_csv("failed_payments.csv", index=False)

print("Total customers:", len(customers))
print("Failed payments generated:", len(df))
print()

print("First 5 failed payments:")
print(df.head())
print()

print("Failure reasons:")
print(df["failure_reason"].value_counts())