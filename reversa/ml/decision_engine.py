import pandas as pd
import joblib

from policy_engine import check_policy


# ========================================
# LOAD TRAINED MODEL
# ========================================

pipeline = joblib.load(
    "../data/reversa_model.pkl"
)


# ========================================
# ACTION COSTS
# ========================================

ACTION_COSTS = {
    "WAIT": 0,
    "RETRY": 5,
    "REMINDER": 1,
    "PAYMENT_LINK": 2
}


# ========================================
# POSSIBLE ACTIONS
# ========================================

ACTIONS = [
    "WAIT",
    "RETRY",
    "REMINDER",
    "PAYMENT_LINK"
]


# ========================================
# DECISION FUNCTION
# ========================================

def decide_payment(payment):

    candidates = []

    # ------------------------------------
    # CREATE ACTION CANDIDATES
    # ------------------------------------

    for action in ACTIONS:

        candidates.append({

            "amount":
                payment["amount"],

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

            "action":
                action
        })


    candidate_df = pd.DataFrame(
        candidates
    )


    # ------------------------------------
    # PREDICT RECOVERY PROBABILITY
    # ------------------------------------

    probabilities = pipeline.predict_proba(
        candidate_df
    )[:, 1]


    candidate_df[
        "recovery_probability"
    ] = probabilities


    # ------------------------------------
    # ACTION COST
    # ------------------------------------

    candidate_df[
        "action_cost"
    ] = candidate_df[
        "action"
    ].map(ACTION_COSTS)


    # ------------------------------------
    # NATURAL RECOVERY BASELINE
    # ------------------------------------

    wait_probability = candidate_df.loc[
        candidate_df["action"] == "WAIT",
        "recovery_probability"
    ].iloc[0]


    # ------------------------------------
    # INCREMENTAL RECOVERY
    # ------------------------------------

    candidate_df[
        "incremental_recovery"
    ] = (
        candidate_df["recovery_probability"]
        -
        wait_probability
    )


    # ------------------------------------
    # EXPECTED ADDITIONAL REVENUE
    # ------------------------------------

    candidate_df[
        "incremental_revenue"
    ] = (
        candidate_df["amount"]
        *
        candidate_df["incremental_recovery"]
    )


    # ------------------------------------
    # NET INCREMENTAL VALUE
    # ------------------------------------

    candidate_df[
        "net_incremental_value"
    ] = (
        candidate_df["incremental_revenue"]
        -
        candidate_df["action_cost"]
    )


    # ------------------------------------
    # POLICY FILTER
    # ------------------------------------

    allowed_candidates = []

    policy_results = {}

    for _, row in candidate_df.iterrows():

        result = check_policy(

            action=row["action"],

            recovery_probability=
                row["recovery_probability"],

            attempt_number=
                payment["attempt_number"],

            payment_recovered=False,

            amount=payment["amount"]
        )


        policy_results[row["action"]] = result


        if result["allowed"]:

            allowed_candidates.append(row)


    # ------------------------------------
    # NO ACTION
    # ------------------------------------

    if len(allowed_candidates) == 0:

        return {

            "payment_id":
                payment["payment_id"],

            "customer_id":
                payment["customer_id"],

            "amount":
                float(payment["amount"]),

            "failure_reason":
                payment["failure_reason"],

            "recommended_action":
                "NO_ACTION",

            "recovery_probability":
                None,

            "natural_recovery_probability":
                float(wait_probability),

            "incremental_recovery":
                0.0,

            "action_cost":
                0.0,

            "incremental_revenue":
                0.0,

            "net_incremental_value":
                0.0,

            "reason":
                "All interventions failed policy checks.",

            "policy_results":
                policy_results
        }


    # ------------------------------------
    # ALLOWED DATAFRAME
    # ------------------------------------

    allowed_df = pd.DataFrame(
        allowed_candidates
    )


    # ------------------------------------
    # SORT BY ECONOMIC VALUE
    # ------------------------------------

    allowed_df = allowed_df.sort_values(
        "net_incremental_value",
        ascending=False
    )


    best = allowed_df.iloc[0]


    # ------------------------------------
    # NO POSITIVE VALUE
    # ------------------------------------

    if best["net_incremental_value"] <= 0:

        return {

            "payment_id":
                payment["payment_id"],

            "customer_id":
                payment["customer_id"],

            "amount":
                float(payment["amount"]),

            "failure_reason":
                payment["failure_reason"],

            "recommended_action":
                "NO_ACTION",

            "recovery_probability":
                float(best["recovery_probability"]),

            "natural_recovery_probability":
                float(wait_probability),

            "incremental_recovery":
                float(best["incremental_recovery"]),

            "action_cost":
                float(best["action_cost"]),

            "incremental_revenue":
                float(best["incremental_revenue"]),

            "net_incremental_value":
                float(best["net_incremental_value"]),

            "reason":
                "No allowed intervention has positive incremental value.",

            "policy_results":
                policy_results
        }


    # ------------------------------------
    # FINAL RESULT
    # ------------------------------------

    return {

        "payment_id":
            payment["payment_id"],

        "customer_id":
            payment["customer_id"],

        "amount":
            float(payment["amount"]),

        "failure_reason":
            payment["failure_reason"],

        "recommended_action":
            best["action"],

        "recovery_probability":
            float(best["recovery_probability"]),

        "natural_recovery_probability":
            float(wait_probability),

        "incremental_recovery":
            float(best["incremental_recovery"]),

        "action_cost":
            float(best["action_cost"]),

        "incremental_revenue":
            float(best["incremental_revenue"]),

        "net_incremental_value":
            float(best["net_incremental_value"]),

        "reason":
            "Highest net incremental value among policy-allowed actions.",

        "policy_results":
            policy_results
    }


# ========================================
# LOCAL TEST
# ========================================

if __name__ == "__main__":

    payments = pd.read_csv(
        "../data/failed_payments.csv"
    )

    payment = payments.iloc[0]

    result = decide_payment(payment)

    print()
    print("================================")
    print("REVERSA DECISION ENGINE")
    print("================================")

    print(
        "Payment ID:",
        result["payment_id"]
    )

    print(
        "Customer ID:",
        result["customer_id"]
    )

    print(
        "Amount: ₹",
        result["amount"]
    )

    print(
        "Failure:",
        result["failure_reason"]
    )

    print()
    print("================================")
    print("FINAL RECOMMENDATION")
    print("================================")

    print(
        "Action:",
        result["recommended_action"]
    )

    if result["recovery_probability"] is not None:

        print(
            "Recovery probability:",
            round(
                result["recovery_probability"] * 100,
                2
            ),
            "%"
        )

    print(
        "Natural recovery probability:",
        round(
            result["natural_recovery_probability"] * 100,
            2
        ),
        "%"
    )

    print(
        "Expected additional revenue: ₹",
        round(
            result["incremental_revenue"],
            2
        )
    )

    print(
        "Action cost: ₹",
        round(
            result["action_cost"],
            2
        )
    )

    print(
        "Net incremental value: ₹",
        round(
            result["net_incremental_value"],
            2
        )
    )

    print(
        "Reason:",
        result["reason"]
    )