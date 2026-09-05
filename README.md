# REVERSA — Intelligent Failed Payment Recovery

REVERSA is an ML-powered payment recovery decision engine that analyzes failed payments and recommends the most effective recovery action based on machine learning predictions, failure reasons, business policies, and expected economic value.

---

## Problem

When a payment fails, using the same recovery action for every customer can lead to unnecessary costs and missed recovery opportunities.

REVERSA dynamically evaluates possible recovery actions:

- **WAIT**
- **RETRY**
- **REMINDER**
- **PAYMENT_LINK**

The system selects the most suitable action for each failed payment.

---

## How It Works

```text
Failed Payment
      ↓
Payment & Customer Data
      ↓
ML Recovery Prediction
      ↓
Reason-Aware Policy
      ↓
Economic Value Calculation
      ↓
Hybrid Decision Engine
      ↓
Recommended Recovery Action

```


REVERSA considers:

Failure reason
Payment amount
Attempt number
Customer payment history
Subscription history
Historical reminder success
Recovery probability
Natural recovery probability
Action cost
Expected incremental revenue
Policy constraints
System Architecture

```text
                         REVERSA
                            |
             +--------------+--------------+
             |                             |
             ↓                             ↓
      React Frontend                 FastAPI Backend
             |                             |
             |                             ↓
             |                     Decision Engine
             |                             |
             |              +--------------+--------------+
             |              |                             |
             |              ↓                             ↓
             |       ML Prediction              Reason-Aware Policy
             |              |                             |
             |              +--------------+--------------+
             |                             |
             |                             ↓
             |                    Economic Optimization
             |                             |
             +-----------------------------+
                                           ↓
                                Final Recovery Action

```
Machine Learning

The model is trained using payment and customer features combined with historical recovery actions.

Features
Payment amount
Failure reason
Attempt number
Subscription months
Successful payments
Failed payments
Average payment delay
Reminder success rate
Recovery action
Model Performance
Metric	Result
Accuracy	67.13%
Precision	69.58%
Recall	80.61%
F1 Score	74.69%
ROC-AUC	72.01%
Reason-Aware Recovery Policy

Different payment failure reasons respond differently to recovery actions.

Failure Reason	Recommended Action	Recovery Rate
Authentication Failed	PAYMENT_LINK	88.60%
Bank Error	RETRY	76.00%
Insufficient Funds	WAIT	65.44%
Payment Method Issue	PAYMENT_LINK	85.14%

This policy layer allows REVERSA to make failure-specific decisions instead of treating every failed payment identically.

Economic Decisioning

REVERSA does not select an action only based on recovery probability.

Instead, it calculates the expected incremental economic value of each valid recovery action.

Incremental Recovery
Incremental Recovery
=
Action Recovery Probability
-
Natural Recovery Probability
Expected Additional Revenue
Expected Additional Revenue
=
Payment Amount × Incremental Recovery
Net Incremental Value
Net Incremental Value
=
Expected Additional Revenue - Action Cost

The action with the highest valid economic value is selected while respecting policy constraints.

Final Evaluation

The final evaluation was performed on:

600 held-out payment cases

Results
Strategy	Recovered	Recovery Rate
Always Wait	334 / 600	55.67%
Always Retry	270 / 600	45.00%
Always Reminder	410 / 600	68.33%
Always Payment Link	430 / 600	71.67%
Reason Policy	472 / 600	78.67%
REVERSA Hybrid	476 / 600	79.33%
Key Result

REVERSA achieved a:

79.33% recovery rate

compared with:

71.67% for the best fixed-action baseline

Improvement

REVERSA achieved:

+7.67 percentage points

over the best fixed-action strategy.

The hybrid system also improved over the reason-aware policy by:

+0.67 percentage points

Note: The dataset used for evaluation is simulated/constructed for this project and should not be interpreted as production payment performance.

Example Decision

For a failed payment:

Payment ID: P000001
Customer ID: C04701
Amount: ₹499
Failure: authentication_failed

REVERSA can recommend:

Action: PAYMENT_LINK

Recovery Probability: 90.65%
Natural Recovery Probability: 58.22%

Expected Additional Revenue: ₹161.80
Action Cost: ₹2.00

Net Incremental Value: ₹159.80

This demonstrates how REVERSA combines ML prediction, business policy, and economic optimization to produce an actionable recommendation.

Web Dashboard

The project includes a React-based dashboard for analyzing failed payments and displaying the recommended recovery action.

Dashboard Capabilities
Failed payment input
Customer information
Failure reason selection
Recovery recommendation
Recovery probability
Natural recovery probability
Expected additional revenue
Action cost
Net incremental value
Strategy comparison
REST API

REVERSA exposes a FastAPI backend for payment analysis.

Health Check

Endpoint

GET /
Example Response
{
  "system": "REVERSA",
  "status": "running",
  "message": "Failed payment recovery decision engine"
}
Analyze Payment

Endpoint

POST /api/analyze
Example Request
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
Python
Pandas
Scikit-learn
Joblib
Development Tools
Git
GitHub
Project Structure
reversa/
│
├── backend/
│   └── main.py
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── App.css
│   │   └── ...
│   ├── package.json
│   └── vite.config.js
│
├── ml/
│   ├── generate_action_data.py
│   ├── train_model.py
│   ├── policy_engine.py
│   ├── reason_policy.py
│   ├── decision_engine.py
│   ├── hybrid_decision.py
│   ├── hybrid_evaluation.py
│   ├── final_evaluation.py
│   └── ...
│
├── data/
│   ├── action_outcomes.csv
│   ├── failed_payments.csv
│   ├── reversa_model.pkl
│   └── ...
│
└── README.md
Running the Project
1. Start the Backend

Open a terminal and run:

cd backend
python -m uvicorn main:app --reload --port 8000

Backend will be available at:

http://127.0.0.1:8000
2. Start the Frontend

Open another terminal:

cd frontend
npm install
npm run dev

Frontend will be available at:

http://localhost:5173
ML Pipeline
Generate the Dataset
cd ml
python generate_action_data.py
Train the Model
python train_model.py
Run the Decision Engine
python decision_engine.py
Run Hybrid Evaluation
python hybrid_evaluation.py
Run Final Evaluation
python final_evaluation.py
Future Improvements
Real payment gateway integration
Real historical payment data
Online model learning
A/B testing of recovery strategies
Customer lifetime value integration
Model monitoring and drift detection
Cloud deployment
Production database integration
Author

Praveen Kumar

GitHub: praveenKUMAR-05
