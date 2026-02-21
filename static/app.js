/* ─────────────────────────────────────────────
   ChurnGuard — Frontend Logic v3
   ───────────────────────────────────────────── */

const API_BASE = window.location.origin;

// ── Human-readable labels ────────────────────
const FEATURE_LABELS = {
  gender: "Gender",
  age: "Age",
  partner: "Partner",
  dependents: "Dependents",
  tenure_months: "Tenure",
  monthly_charges: "Monthly Charges",
  contract_type: "Contract",
  payment_method: "Payment Method",
  paperless_billing: "Paperless Billing",
  internet_service: "Internet Service",
  online_security: "Online Security",
  tech_support: "Tech Support",
  num_tickets: "Support Tickets",
};

const VALUE_LABELS = {
  gender: { 0: "Male", 1: "Female" },
  partner: { 0: "No", 1: "Yes" },
  dependents: { 0: "No", 1: "Yes" },
  contract_type: { 0: "Monthly", 1: "Annual", 2: "Biennial" },
  payment_method: { 0: "Bank Transfer", 1: "Credit Card", 2: "E-Check", 3: "Mailed Check" },
  paperless_billing: { 0: "No", 1: "Yes" },
  internet_service: { 0: "None", 1: "DSL", 2: "Fiber Optic" },
  online_security: { 0: "No", 1: "Yes" },
  tech_support: { 0: "No", 1: "Yes" },
};

// ── DOM references ───────────────────────────
const form = document.getElementById("predict-form");
const predictBtn = document.getElementById("predict-btn");
const resultPlaceholder = document.getElementById("result-placeholder");
const resultContent = document.getElementById("result-content");
const statusDot = document.querySelector(".status-dot");
const statusText = document.querySelector(".status-text");

// ── Sync range sliders with inputs ───────────
const rangeFields = ["age", "tenure_months", "monthly_charges", "num_tickets"];
rangeFields.forEach((name) => {
  const input = document.getElementById(name);
  const range = document.getElementById(`${name}-range`);
  if (!input || !range) return;
  input.addEventListener("input", () => { range.value = input.value; });
  range.addEventListener("input", () => { input.value = range.value; });
});

// ── Toggle switches: update label text ───────
const toggleFields = ["partner", "dependents", "online_security", "tech_support", "paperless_billing"];
toggleFields.forEach((name) => {
  const checkbox = document.getElementById(name);
  const label = document.getElementById(`${name}-label`);
  if (!checkbox || !label) return;
  const update = () => { label.textContent = checkbox.checked ? "Yes" : "No"; };
  checkbox.addEventListener("change", update);
  update(); // init
});

// ── Navigation ───────────────────────────────
document.querySelectorAll(".nav-link").forEach((link) => {
  link.addEventListener("click", (e) => {
    e.preventDefault();
    const target = link.dataset.section;
    document.querySelectorAll(".section").forEach((s) => s.classList.remove("active"));
    document.querySelectorAll(".nav-link").forEach((l) => l.classList.remove("active"));
    document.getElementById(target).classList.add("active");
    link.classList.add("active");
  });
});

// ── API health check ─────────────────────────
async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    const data = await res.json();
    statusDot.classList.add("online");
    statusDot.classList.remove("offline");
    statusText.textContent = data.model_loaded ? "Online" : "Model missing";
  } catch {
    statusDot.classList.add("offline");
    statusDot.classList.remove("online");
    statusText.textContent = "Offline";
  }
}
checkHealth();
setInterval(checkHealth, 15000);

// ── Load model info ──────────────────────────
async function loadModelInfo() {
  try {
    const res = await fetch(`${API_BASE}/model-info`);
    if (!res.ok) throw new Error("No model info");
    const data = await res.json();

    // Model info card
    const infoHtml = `
      <div class="info-row"><span class="info-label">Type</span><span class="info-value">${data.model_type || "N/A"}</span></div>
      <div class="info-row"><span class="info-label">Version</span><span class="info-value">${(data.model_version || "").slice(0, 12)}…</span></div>
      <div class="info-row"><span class="info-label">MLflow Run</span><span class="info-value">${(data.mlflow_run_id || "").slice(0, 12)}…</span></div>
      <div class="info-row"><span class="info-label">Churn Rate</span><span class="info-value">${data.churn_rate ? (data.churn_rate * 100).toFixed(1) + "%" : "N/A"}</span></div>
      <div class="info-row"><span class="info-label">Features</span><span class="info-value">${(data.feature_cols || []).length} (${(data.raw_feature_cols || []).length} raw + ${(data.feature_cols || []).length - (data.raw_feature_cols || []).length} engineered)</span></div>
      ${data.best_params ? `<div class="info-row"><span class="info-label">Best Params</span><span class="info-value" style="font-size:.72rem;max-width:200px;text-align:right;">${formatParams(data.best_params)}</span></div>` : ""}
    `;
    document.getElementById("model-info-content").innerHTML = infoHtml;

    // Metrics card
    const metrics = [
      { name: "F1 Score (CV)", value: data.cv_f1, color: "#38bdf8" },
      { name: "F1 Score (Val)", value: data.val_f1, color: "#a78bfa" },
      { name: "Accuracy", value: data.val_accuracy, color: "#34d399" },
      { name: "Precision", value: data.val_precision, color: "#fbbf24" },
      { name: "Recall", value: data.val_recall, color: "#fb923c" },
      { name: "ROC AUC", value: data.val_roc_auc, color: "#f472b6" },
    ];
    const metricsHtml = metrics
      .filter((m) => m.value != null)
      .map(
        (m) => `
        <div class="metric-bar">
          <div class="metric-bar-header">
            <span class="metric-name">${m.name}</span>
            <span class="metric-value">${(m.value * 100).toFixed(1)}%</span>
          </div>
          <div class="metric-bar-track">
            <div class="metric-bar-fill" style="width:${m.value * 100}%;background:${m.color}"></div>
          </div>
        </div>`
      )
      .join("");
    document.getElementById("model-metrics-content").innerHTML = metricsHtml;

    // Features card (distinguish raw vs engineered)
    const rawCols = data.raw_feature_cols || [];
    const allCols = data.feature_cols || [];
    if (allCols.length) {
      const chipsHtml = allCols
        .map((f) => {
          const isEngineered = !rawCols.includes(f);
          return `<span class="feature-chip${isEngineered ? ' engineered' : ''}">${f}${isEngineered ? ' ⚙️' : ''}</span>`;
        })
        .join("");
      document.getElementById("model-features-content").innerHTML =
        `<div class="feature-chips">${chipsHtml}</div>
         <p style="margin-top:.75rem;font-size:.75rem;color:var(--text-muted)">
           <span class="feature-chip" style="font-size:.7rem;padding:.2rem .5rem">raw</span> = user input &nbsp;
           <span class="feature-chip engineered" style="font-size:.7rem;padding:.2rem .5rem">engineered ⚙️</span> = auto-computed
         </p>`;
    }
  } catch {
    document.getElementById("model-info-content").innerHTML = '<p class="loading-text">Unable to load model information.</p>';
  }
}

function formatParams(params) {
  if (!params) return "";
  return Object.entries(params)
    .map(([k, v]) => `${k.replace("clf__", "")}: ${v}`)
    .join(", ");
}

loadModelInfo();

// ── Gauge drawing ────────────────────────────
function drawGauge(probability) {
  const fill = document.getElementById("gauge-fill");
  const total = Math.PI * 80;
  const offset = total * (1 - probability);
  fill.style.strokeDasharray = `${total}`;
  fill.style.strokeDashoffset = `${offset}`;

  if (probability >= 0.6) {
    fill.style.stroke = "#f87171";
  } else if (probability >= 0.35) {
    fill.style.stroke = "#fbbf24";
  } else {
    fill.style.stroke = "#34d399";
  }

  document.getElementById("gauge-value").textContent = `${(probability * 100).toFixed(1)}%`;
}

// ── Prediction ───────────────────────────────
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  await predict();
});

async function predict() {
  const payload = {
    gender: parseInt(document.getElementById("gender").value),
    age: parseFloat(document.getElementById("age").value),
    partner: document.getElementById("partner").checked ? 1 : 0,
    dependents: document.getElementById("dependents").checked ? 1 : 0,
    tenure_months: parseFloat(document.getElementById("tenure_months").value),
    monthly_charges: parseFloat(document.getElementById("monthly_charges").value),
    contract_type: parseInt(document.getElementById("contract_type").value),
    payment_method: parseInt(document.getElementById("payment_method").value),
    paperless_billing: document.getElementById("paperless_billing").checked ? 1 : 0,
    internet_service: parseInt(document.getElementById("internet_service").value),
    online_security: document.getElementById("online_security").checked ? 1 : 0,
    tech_support: document.getElementById("tech_support").checked ? 1 : 0,
    num_tickets: parseFloat(document.getElementById("num_tickets").value),
  };

  predictBtn.disabled = true;
  predictBtn.querySelector(".btn-text").textContent = "Analyzing…";
  predictBtn.querySelector(".btn-loader").hidden = false;

  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Server error");
    }

    const data = await res.json();
    showResult(data);
  } catch (err) {
    alert(`Error: ${err.message}`);
  } finally {
    predictBtn.disabled = false;
    predictBtn.querySelector(".btn-text").textContent = "Predict";
    predictBtn.querySelector(".btn-loader").hidden = true;
  }
}

// ── Format feature values for display ────────
function formatFeatureValue(feature, value) {
  if (VALUE_LABELS[feature] && VALUE_LABELS[feature][value] !== undefined) {
    return VALUE_LABELS[feature][value];
  }
  if (feature === "monthly_charges") return `$${Number(value).toFixed(0)}`;
  if (feature === "tenure_months") return `${value} mo`;
  if (feature === "age") return `${value} yrs`;
  if (feature === "num_tickets") return `${value}`;
  return String(value);
}

// ── Show Result ──────────────────────────────
function showResult(data) {
  resultPlaceholder.hidden = true;
  resultContent.hidden = false;
  resultContent.classList.add("pop-in");

  const proba = data.churn_probability ?? (data.churn_prediction === 1 ? 0.9 : 0.1);
  const isChurn = data.churn_prediction === 1;

  // Gauge
  drawGauge(proba);

  // Verdict
  const verdict = document.getElementById("verdict");
  verdict.className = `verdict ${isChurn ? "churn" : "safe"}`;
  document.getElementById("verdict-icon").textContent = isChurn ? "⚠️" : "✅";
  document.getElementById("verdict-text").textContent = isChurn
    ? "Churn risk detected"
    : "Loyal customer";

  // Details
  document.getElementById("detail-prediction").textContent = isChurn ? "Churn" : "No Churn";
  document.getElementById("detail-proba").textContent = `${(proba * 100).toFixed(1)}%`;

  let riskLevel, riskColor;
  if (proba >= 0.6) { riskLevel = "High"; riskColor = "var(--danger)"; }
  else if (proba >= 0.35) { riskLevel = "Medium"; riskColor = "var(--warning)"; }
  else { riskLevel = "Low"; riskColor = "var(--success)"; }

  const riskEl = document.getElementById("detail-risk");
  riskEl.textContent = riskLevel;
  riskEl.style.color = riskColor;

  // Feature contributions
  renderContributions(data.feature_contributions || []);

  // Recommendation
  const recEl = document.getElementById("recommendation");
  if (proba >= 0.6) {
    recEl.innerHTML = "💡 <strong>Urgent action recommended:</strong> Offer a discount or upgrade, proactively reach out to the customer, and review recent support tickets.";
    recEl.style.borderLeftColor = "var(--danger)";
    recEl.style.background = "var(--danger-bg)";
  } else if (proba >= 0.35) {
    recEl.innerHTML = "💡 <strong>Monitoring recommended:</strong> Schedule a satisfaction check-in, review open tickets, and propose a longer-term contract.";
    recEl.style.borderLeftColor = "var(--warning)";
    recEl.style.background = "var(--warning-bg)";
  } else {
    recEl.innerHTML = "💡 <strong>Satisfied customer:</strong> Maintain service quality. Opportunity for cross-sell or upsell.";
    recEl.style.borderLeftColor = "var(--success)";
    recEl.style.background = "var(--success-bg)";
  }

  setTimeout(() => resultContent.classList.remove("pop-in"), 500);
}

// ── Render Feature Contributions ─────────────
function renderContributions(contributions) {
  const container = document.getElementById("contributions");
  const list = document.getElementById("contributions-list");

  if (!contributions || contributions.length === 0) {
    container.hidden = true;
    return;
  }
  container.hidden = false;

  const maxContrib = Math.max(...contributions.map((c) => Math.abs(c.contribution)));

  list.innerHTML = contributions
    .slice(0, 6)
    .map((c) => {
      const isPositive = c.contribution > 0; // positive = pushes toward churn
      const width = Math.min((Math.abs(c.contribution) / maxContrib) * 100, 100);
      const color = isPositive ? "var(--danger)" : "var(--success)";
      const label = FEATURE_LABELS[c.feature] || c.feature;
      const valueStr = formatFeatureValue(c.feature, c.value);
      const arrow = isPositive ? "↑" : "↓";
      const direction = isPositive ? "Churn" : "Retain";
      const pct = (Math.abs(c.contribution) * 100).toFixed(1);

      return `
        <div class="contrib-row">
          <div class="contrib-header">
            <span class="contrib-feature">${label}</span>
            <span class="contrib-value">${valueStr}</span>
          </div>
          <div class="contrib-bar-track">
            <div class="contrib-bar" style="width:${width}%;background:${color}"></div>
          </div>
          <span class="contrib-direction" style="color:${color}">${arrow} ${direction} ${pct}%</span>
        </div>
      `;
    })
    .join("");
}

// ── Presets ───────────────────────────────────
document.querySelectorAll(".preset-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    const preset = JSON.parse(btn.dataset.preset);
    Object.entries(preset).forEach(([key, value]) => {
      // Handle toggles (checkboxes)
      const checkbox = document.getElementById(key);
      if (checkbox && checkbox.type === "checkbox") {
        checkbox.checked = value === 1;
        // Trigger change event for label update
        checkbox.dispatchEvent(new Event("change"));
        return;
      }
      // Handle inputs and selects
      const input = document.getElementById(key);
      if (input) {
        input.value = value;
        const range = document.getElementById(`${key}-range`);
        if (range) range.value = value;
      }
    });
    // Auto-predict
    predict();
  });
});
