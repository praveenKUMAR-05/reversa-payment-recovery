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
