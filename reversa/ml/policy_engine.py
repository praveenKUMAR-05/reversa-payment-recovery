# --------------------------------
# REVERSA POLICY ENGINE
# --------------------------------


MAX_RETRIES = 2

MIN_RECOVERY_PROBABILITY = 0.60


def check_policy(
    action,
    recovery_probability,
    attempt_number,
    payment_recovered,
    amount
):

    reasons = []


    # --------------------------------
    # RULE 1
    # PAYMENT ALREADY RECOVERED
    # --------------------------------

    if payment_recovered:

        return {
            "allowed": False,
            "reason": "Payment already recovered."
        }


    # --------------------------------
    # RULE 2
    # RETRY LIMIT
    # --------------------------------

    if action == "RETRY":

        if attempt_number >= MAX_RETRIES:

            return {
                "allowed": False,
                "reason":
                    "Maximum retry limit reached."
            }


    # --------------------------------
    # RULE 3
    # RECOVERY PROBABILITY
    # --------------------------------

    if recovery_probability < MIN_RECOVERY_PROBABILITY:

        return {
            "allowed": False,
            "reason":
                "Recovery probability below safety threshold."
        }


    # --------------------------------
    # RULE 4
    # VALID AMOUNT
    # --------------------------------

    if amount <= 0:

        return {
            "allowed": False,
            "reason":
                "Invalid payment amount."
        }


    # --------------------------------
    # ALL RULES PASSED
    # --------------------------------

    return {
        "allowed": True,
        "reason":
            "Action passed all policy checks."
    }

