# REVERSA — Intelligent Failed Payment Recovery System

> An ML-powered payment recovery decision engine that determines the most effective recovery action for failed payments using machine learning, policy constraints, and economic optimization.

---

## Overview

Failed payments are a major source of revenue leakage in subscription and recurring-payment systems.

A simple recovery strategy such as **always retrying**, **always sending a reminder**, or **always generating a payment link** treats every failed payment the same way.

REVERSA takes a different approach.

It analyzes each failed payment using:

- Customer payment behavior
- Failure reason
- Payment history
- Recovery probability
- Natural recovery probability
- Action cost
- Expected incremental revenue
- Business and policy constraints

The system then selects the recovery action with the highest expected economic value while respecting predefined policies.

### Supported Recovery Actions

| Action | Description |
|---|---|
| `WAIT` | Allow the customer to recover naturally |
| `RETRY` | Attempt the payment again |
| `REMINDER` | Send a payment reminder |
| `PAYMENT_LINK` | Provide a new payment link |

---

# Key Idea

REVERSA does not simply ask:

> **"Which action has the highest recovery probability?"**

Instead, it asks:

> **"Which allowed action creates the highest expected incremental business value?"**

The decision process is:

```text
Failed Payment
      │
      ▼
Customer & Payment Features
      │
      ▼
ML Recovery Prediction
      │
      ▼
Reason-Aware Policy
      │
      ▼
Economic Evaluation
      │
      ├── Recovery Probability
      ├── Natural Recovery
      ├── Action Cost
      └── Expected Revenue
      │
      ▼
Hybrid Decision Engine
      │
      ▼
Recommended Recovery Action

```
```text

                         REVERSA
                            |
             +--------------+--------------+
             |                             |
             v                             v
       React Frontend                 FastAPI Backend
             |                             |
             |                       REST API Request
             |                             |
             +-------------+---------------+
                           |
                           v
                Hybrid Decision Engine
                           |
             +-------------+-------------+
             |                           |
             v                           v
      ML Prediction              Reason-Aware Policy
             |                           |
             |                           |
             +-------------+-------------+
                           |
                           v
                 Economic Optimization
                           |
                           v
                 Best Recovery Action

```

Architecture Components

1. React Frontend

Provides the user interface for submitting failed payment information and displaying the recommended recovery action.

2. FastAPI Backend

Provides REST API endpoints that receive payment information and communicate with the decision engine.

3. ML Prediction

Predicts the probability that each recovery action will successfully recover the payment.

4. Reason-Aware Policy

Considers the specific failure reason and applies business rules before allowing an action.

5. Economic Optimization

Calculates the expected additional revenue and subtracts the action cost.

6. Hybrid Decision Engine

Combines ML predictions, reason-aware policy, and economic value to select the final action.

Machine Learning

The ML model is trained using historical action-outcome data containing multiple recovery actions for failed payments.

Input Features

The model uses features including:

Payment amount
Failure reason
Attempt number
Subscription duration
Successful payment count
Failed payment count
Average payment delay
Historical reminder success rate
Recovery action
Target
recovered

The target indicates whether the selected recovery action successfully recovered the payment.

Model Performance
Metric	Score
Accuracy	67.13%
Precision	69.58%
Recall	80.61%
F1 Score	74.69%
ROC-AUC	72.01%

The ML model is used as one component of the complete decision system rather than as the sole decision-maker.

Reason-Aware Policy

Different payment failure reasons respond differently to recovery actions.

Evaluation showed the following best-performing actions:

Failure Reason	Best Action	Recovery Rate
Authentication Failed	PAYMENT_LINK	88.60%
Bank Error	RETRY	76.00%
Insufficient Funds	WAIT	65.44%
Payment Method Issue	PAYMENT_LINK	85.14%

REVERSA uses this information through its reason-aware policy layer.

Economic Decision Model

REVERSA evaluates the economic value of an intervention.

Incremental Recovery
Incremental Recovery
=
Action Recovery Probability
-
Natural Recovery Probability
Expected Additional Revenue
Expected Additional Revenue
=
Payment Amount
×
Incremental Recovery
Net Incremental Value
Net Incremental Value
=
Expected Additional Revenue
-
Action Cost

The system selects the action with the highest valid net value while respecting policy constraints.

Hybrid Decision Engine

The final REVERSA system combines three components:

Machine Learning
       +
Reason-Aware Policy
       +
Economic Optimization
       =
REVERSA Hybrid Decision

This allows the system to balance:

Prediction accuracy
Failure-specific behavior
Business rules
Recovery probability
Revenue
Action cost
Final Evaluation

The final evaluation was performed on:

600 held-out payment cases
2400 action records
Strategy Comparison
Strategy	Recovered	Recovery Rate
ALWAYS WAIT	334	55.67%
ALWAYS RETRY	270	45.00%
ALWAYS REMINDER	410	68.33%
ALWAYS PAYMENT_LINK	430	71.67%
REASON POLICY	472	78.67%
REVERSA HYBRID	476	79.33%
Key Result

REVERSA achieved a:

79.33% Recovery Rate

The best fixed-action strategy achieved:

71.67% Recovery Rate

Therefore:

79.33% - 71.67%
=
+7.67 percentage points

The hybrid system also improved over the reason-aware policy:

79.33% - 78.67%
=
+0.67 percentage points
Final Verdict

REVERSA outperformed all fixed-action recovery strategies on the held-out evaluation dataset.

Example Decision

Example failed payment:

Payment ID: P000001
Customer ID: C04701
Amount: ₹499
Failure: authentication_failed

REVERSA can produce:

Recommended Action:
PAYMENT_LINK

Recovery Probability:
90.65%

Natural Recovery Probability:
58.22%

Expected Additional Revenue:
₹161.80

Action Cost:
₹2.00

Net Incremental Value:
₹159.80

The final recommendation is based on expected economic value while respecting the system's policies.

REST API

REVERSA provides a REST API using FastAPI.

Health Check
GET /

Example response:

{
  "system": "REVERSA",
  "status": "running",
  "message": "Failed payment recovery decision engine"
}
Analyze Payment
POST /api/analyze

Example request:

{
  "payment_id": "P000001",
  "customer_id": "C04701",
  "amount": 499,
  "failure_reason": "authentication_failed",
  "attempt_number": 1,
  "subscription_months": 12,
  "successful_payments": 10,
  "failed_payments": 2,
  "avg_delay_days": 1.5,
  "reminder_success_rate": 0.70
}

Example response:

{
  "payment_id": "P000001",
  "customer_id": "C04701",
  "amount": 499.0,
  "failure_reason": "authentication_failed",
  "selected_action": "PAYMENT_LINK",
  "recovery_probability": 0.876,
  "natural_recovery_probability": 0.4933,
  "expected_additional_revenue": 190.98,
  "action_cost": 2.0,
  "net_incremental_value": 188.98,
  "reason": "Highest net incremental value among policy-allowed actions."
}
Tech Stack
Frontend
React
Vite
JavaScript
CSS
Backend
Python
FastAPI
Uvicorn
Machine Learning
Scikit-learn
Pandas
Joblib
Development
Git
GitHub
Project Structure
reversa/
|
+-- backend/
|   +-- main.py
|   +-- ...
|
+-- frontend/
|   +-- src/
|   |   +-- App.jsx
|   |   +-- App.css
|   |   +-- ...
|   +-- package.json
|   +-- vite.config.js
|
+-- ml/
|   +-- generate_action_data.py
|   +-- train_model.py
|   +-- policy_engine.py
|   +-- reason_policy.py
|   +-- decision_engine.py
|   +-- hybrid_decision.py
|   +-- hybrid_evaluation.py
|   +-- final_evaluation.py
|   +-- ...
|
+-- data/
|   +-- action_outcomes.csv
|   +-- failed_payments.csv
|   +-- reversa_model.pkl
|   +-- test_action_outcomes.csv
|   +-- test_payment_ids.csv
|   +-- hybrid_results.csv
|
+-- README.md
Installation
1. Clone Repository
git clone https://github.com/praveenKUMAR-05/reversa-payment-recovery.git
cd reversa-payment-recovery
2. Backend Setup

Create a virtual environment:

python -m venv venv

Windows:

venv\Scripts\activate

Install dependencies:

pip install pandas scikit-learn joblib fastapi uvicorn

Start the backend:

cd backend
python -m uvicorn main:app --reload --port 8000

Backend:

http://127.0.0.1:8000
3. Frontend Setup

Open another terminal:

cd frontend
npm install
npm run dev

Frontend:

http://localhost:5173
Running the ML Pipeline

Generate the action dataset:

cd ml
python generate_action_data.py

Train the model:

python train_model.py

Run the decision engine:

python decision_engine.py

Run the hybrid decision engine:

python hybrid_decision.py

Evaluate the hybrid system:

python hybrid_evaluation.py

Run final evaluation:

python final_evaluation.py
Evaluation Methodology

The system uses a held-out test set containing:

600 unique payment cases
2400 action records

Each payment has outcome information for multiple possible recovery actions.

This enables comparison between:

Fixed recovery strategies
Reason-aware policy
Hybrid decision strategy

The evaluation measures actual payment recovery and compares REVERSA against fixed-action baselines.

Why REVERSA?

Traditional recovery systems may follow a static strategy:

Payment Failed
      |
      v
Same Action For Everyone

REVERSA instead follows:

Payment Failed
      |
      v
Understand Failure
      |
      v
Predict Recovery
      |
      v
Apply Business Policy
      |
      v
Calculate Economic Value
      |
      v
Select Best Action

This makes the system more adaptive and focused on business value.

Limitations

This project is a prototype decision-support system.

The current evaluation dataset is generated/simulated and should not be interpreted as production payment performance.

Before production deployment, the model and policy should be validated using real historical payment outcomes.

Future Improvements
Real payment gateway integration
Production database integration
Online learning from new payment outcomes
Model calibration
A/B testing of recovery strategies
Time-to-recovery prediction
Customer lifetime value integration
Model drift monitoring
Authentication and authorization
Cloud deployment
Production monitoring and logging
Results At a Glance
Metric	Result
Test Payments	600
Best Fixed Baseline	71.67%
Reason Policy	78.67%
REVERSA Hybrid	79.33%
Improvement over Best Baseline	+7.67 pp
Author

Praveen Kumar

GitHub:
https://github.com/praveenKUMAR-05

