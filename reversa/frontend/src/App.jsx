import { useState } from "react";
import "./App.css";

function App() {
  const [payment, setPayment] = useState({
    payment_id: "P000001",
    customer_id: "C04701",
    amount: 499,
    failure_reason: "authentication_failed",
    attempt_number: 2,
    subscription_months: 8,
    successful_payments: 6,
    failed_payments: 2,
    avg_delay_days: 1.8,
    reminder_success_rate: 0.72,
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const failureReasons = [
    "authentication_failed",
    "bank_error",
    "insufficient_funds",
    "payment_method_issue",
  ];

  const handleChange = (e) => {
    const { name, value } = e.target;

    setPayment({
      ...payment,
      [name]:
        name === "amount" ||
        name === "attempt_number" ||
        name === "subscription_months" ||
        name === "successful_payments" ||
        name === "failed_payments" ||
        name === "avg_delay_days"
          ? Number(value)
          : value,
    });
  };

  const analyzePayment = async () => {
  setLoading(true);
  setError("");

  try {
    const response = await fetch(
      "http://127.0.0.1:8000/api/analyze",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payment),
      }
    );

    if (!response.ok) {
      throw new Error(
        `Backend returned ${response.status}`
      );
    }

    const data = await response.json();

    setResult({
      action: data.selected_action,
      probability:
        Number(data.recovery_probability * 100).toFixed(2),
      natural:
        Number(
          data.natural_recovery_probability * 100
        ).toFixed(2),
      revenue:
        Number(
          data.expected_additional_revenue
        ).toFixed(2),
      cost:
        Number(data.action_cost).toFixed(2),
      net:
        Number(data.net_incremental_value).toFixed(2),
      reason: data.reason,
    });

  } catch (err) {

    console.error(err);

    setError(
      "Unable to connect to REVERSA backend. Make sure the FastAPI server is running."
    );

    setResult(null);

  } finally {

    setLoading(false);

  }
};

   
  return (
    <div className="app">
      {/* SIDEBAR */}
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">R</div>
          <div>
            <h1>REVERSA</h1>
            <span>Payment Recovery AI</span>
          </div>
        </div>

        <nav>
          <div className="nav-item active">
            <span>⌂</span>
            Dashboard
          </div>

          <div className="nav-item">
            <span>◈</span>
            Decision Engine
          </div>

          <div className="nav-item">
            <span>▣</span>
            Analytics
          </div>

          <div className="nav-item">
            <span>⚙</span>
            Policy
          </div>
        </nav>

        <div className="sidebar-bottom">
          <div className="system-status">
            <span className="status-dot"></span>
            System Online
          </div>

          <small>REVERSA v1.0</small>
        </div>
      </aside>

      {/* MAIN */}
      <main className="main">
        <header className="topbar">
          <div>
            <p className="eyebrow">INTELLIGENT PAYMENT RECOVERY</p>
            <h2>Recovery Decision Dashboard</h2>
            <p className="subtitle">
              Analyze failed payments and select the most economically valuable
              recovery action.
            </p>
          </div>

          <div className="header-status">
            <span className="status-dot"></span>
            Engine Running
          </div>
        </header>

        {/* STATS */}
        <section className="stats">
          <div className="stat-card">
            <span>MODEL RECOVERY</span>
            <strong>79.33%</strong>
            <small>Hybrid evaluation</small>
          </div>

          <div className="stat-card">
            <span>BEST BASELINE</span>
            <strong>71.67%</strong>
            <small>Always Payment Link</small>
          </div>

          <div className="stat-card">
            <span>IMPROVEMENT</span>
            <strong>+7.67%</strong>
            <small>Percentage points</small>
          </div>

          <div className="stat-card">
            <span>TEST PAYMENTS</span>
            <strong>600</strong>
            <small>Held-out evaluation</small>
          </div>
        </section>

        <div className="dashboard-grid">
          {/* PAYMENT INPUT */}
          <section className="card payment-card">
            <div className="card-header">
              <div>
                <p className="card-label">PAYMENT ANALYSIS</p>
                <h3>Failed Payment</h3>
              </div>

              <span className="badge danger">FAILED</span>
            </div>

            <div className="form-grid">
              <div className="field">
                <label>Payment ID</label>
                <input
                  name="payment_id"
                  value={payment.payment_id}
                  onChange={handleChange}
                />
              </div>

              <div className="field">
                <label>Customer ID</label>
                <input
                  name="customer_id"
                  value={payment.customer_id}
                  onChange={handleChange}
                />
              </div>

              <div className="field">
                <label>Amount (₹)</label>
                <input
                  type="number"
                  name="amount"
                  value={payment.amount}
                  onChange={handleChange}
                />
              </div>

              <div className="field">
                <label>Attempt Number</label>
                <input
                  type="number"
                  name="attempt_number"
                  value={payment.attempt_number}
                  onChange={handleChange}
                />
              </div>

              <div className="field full">
                <label>Failure Reason</label>
                <select
                  name="failure_reason"
                  value={payment.failure_reason}
                  onChange={handleChange}
                >
                  {failureReasons.map((reason) => (
                    <option key={reason} value={reason}>
                      {reason}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <div className="customer-details">
              <div>
                <span>Successful Payments</span>
                <strong>{payment.successful_payments}</strong>
              </div>

              <div>
                <span>Failed Payments</span>
                <strong>{payment.failed_payments}</strong>
              </div>

              <div>
                <span>Subscription</span>
                <strong>{payment.subscription_months} mo</strong>
              </div>

              <div>
                <span>Avg. Delay</span>
                <strong>{payment.avg_delay_days} days</strong>
              </div>
            </div>

            <button className="analyze-btn" onClick={analyzePayment} disabled={loading}>
              {loading ? "Analyzing..." : "Analyze Payment"}
              <span>→</span>
            </button>
          </section>
          {error && (
            <div className="error-message">
              {error}
            </div>
          )}

          {/* RESULT */}
          <section className="card recommendation-card">
            {!result ? (
              <div className="empty-state">
                <div className="empty-icon">✦</div>
                <h3>Ready to Analyze</h3>
                <p>
                  Enter payment information and run the decision engine to
                  generate an optimal recovery action.
                </p>
              </div>
            ) : (
              <>
                <div className="card-header">
                  <div>
                    <p className="card-label">REVERSA RECOMMENDATION</p>
                    <h3>Optimal Recovery Action</h3>
                  </div>

                  <span
                    className={`badge ${
                      result.action === "NO_ACTION"
                        ? "danger"
                        : "success"
                    }`}
                  >
                    {result.action === "NO_ACTION"
                      ? "NO ACTION"
                      : "POLICY ALLOWED"}
                  </span>     
                  </div>

                <div className="recommendation">
                  <div>
                    <span className="recommendation-label">
                      RECOMMENDED ACTION
                    </span>

                    <h4>{result.action}</h4>

                    <p>{result.reason}</p>
                  </div>

                  <div className="probability">
                    <strong>{result.probability}%</strong>
                    <span>Recovery probability</span>
                  </div>
                </div>

                <div className="metrics">
                  <div className="metric">
                    <span>Natural Recovery</span>
                    <strong>{result.natural}%</strong>
                  </div>

                  <div className="metric">
                    <span>Additional Revenue</span>
                    <strong>₹ {result.revenue}</strong>
                  </div>

                  <div className="metric">
                    <span>Action Cost</span>
                    <strong>₹ {result.cost}</strong>
                  </div>

                  <div className="metric highlight">
                    <span>Net Incremental Value</span>
                    <strong>₹ {result.net}</strong>
                  </div>
                </div>

                <div className="decision-explanation">
                  <span>WHY REVERSA CHOSE THIS</span>

                  <p>
                    REVERSA compares the predicted recovery probability,
                    natural recovery probability, action cost and policy
                    constraints before selecting the action with the highest
                    expected economic value.
                  </p>
                </div>
              </>
            )}
          </section>
        </div>

        {/* BOTTOM */}
        <section className="card strategy-card">
          <div className="card-header">
            <div>
              <p className="card-label">SYSTEM PERFORMANCE</p>
              <h3>Strategy Comparison</h3>
            </div>
          </div>

          <div className="strategy-table">
            <div className="table-row table-head">
              <span>Strategy</span>
              <span>Recovery</span>
              <span>Recovery Rate</span>
              <span>Result</span>
            </div>

            <div className="table-row">
              <span>Always Wait</span>
              <span>334 / 600</span>
              <span>55.67%</span>
              <span>Baseline</span>
            </div>

            <div className="table-row">
              <span>Always Reminder</span>
              <span>410 / 600</span>
              <span>68.33%</span>
              <span>Baseline</span>
            </div>

            <div className="table-row">
              <span>Always Payment Link</span>
              <span>430 / 600</span>
              <span>71.67%</span>
              <span>Best Fixed Baseline</span>
            </div>

            <div className="table-row reason-row">
              <span>Reason Policy</span>
              <span>472 / 600</span>
              <span>78.67%</span>
              <span>+7.00 pp</span>
            </div>

            <div className="table-row hybrid-row">
              <span>REVERSA Hybrid</span>
              <span>476 / 600</span>
              <span>79.33%</span>
              <span>+7.67 pp</span>
            </div>
          </div>
        </section>

        <footer>
          REVERSA • Intelligent Failed Payment Recovery • ML + Policy + Economic Decisioning
        </footer>
      </main>
    </div>
  );
}

export default App;