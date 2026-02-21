/* ─────────────────────────────────────────────
   ChurnGuard — Frontend Logic
   ───────────────────────────────────────────── */

const API_BASE = window.location.origin;

// ── DOM references ───────────────────────────
const form = document.getElementById("predict-form");
const predictBtn = document.getElementById("predict-btn");
const resultCard = document.getElementById("result-card");
const resultPlaceholder = document.getElementById("result-placeholder");
const resultContent = document.getElementById("result-content");
const statusDot = document.querySelector(".status-dot");
const statusText = document.querySelector(".status-text");

// Sync range sliders with inputs
const rangeFields = ["age", "tenure_months", "monthly_charges", "num_tickets"];
rangeFields.forEach((name) => {
  const input = document.getElementById(name);
  const range = document.getElementById(`${name}-range`);
  if (!input || !range) return;
  input.addEventListener("input", () => { range.value = input.value; });
  range.addEventListener("input", () => { input.value = range.value; });
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
    statusText.textContent = data.model_loaded ? "En ligne" : "Modèle absent";
  } catch {
    statusDot.classList.add("offline");
    statusDot.classList.remove("online");
    statusText.textContent = "Hors ligne";
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
      <div class="info-row"><span class="info-label">Random State</span><span class="info-value">${data.random_state ?? "N/A"}</span></div>
      ${data.best_params ? `<div class="info-row"><span class="info-label">Best Params</span><span class="info-value" style="font-size:.75rem;max-width:200px;text-align:right;">${formatParams(data.best_params)}</span></div>` : ""}
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

    // Features card
    if (data.feature_cols) {
      const chipsHtml = data.feature_cols.map((f) => `<span class="feature-chip">${f}</span>`).join("");
      document.getElementById("model-features-content").innerHTML = `<div class="feature-chips">${chipsHtml}</div>`;
    }
  } catch {
    document.getElementById("model-info-content").innerHTML = '<p class="loading-text">Impossible de charger les informations du modèle.</p>';
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
  const total = Math.PI * 80; // arc length
  const offset = total * (1 - probability);
  fill.style.strokeDasharray = `${total}`;
  fill.style.strokeDashoffset = `${offset}`;

  // Color based on risk
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
    age: parseFloat(document.getElementById("age").value),
    tenure_months: parseFloat(document.getElementById("tenure_months").value),
    monthly_charges: parseFloat(document.getElementById("monthly_charges").value),
    contract_type: parseInt(document.getElementById("contract_type").value),
    num_tickets: parseFloat(document.getElementById("num_tickets").value),
  };

  // UI loading state
  predictBtn.disabled = true;
  predictBtn.querySelector(".btn-text").textContent = "Analyse en cours…";
  predictBtn.querySelector(".btn-loader").hidden = false;

  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Erreur serveur");
    }

    const data = await res.json();
    showResult(data);
  } catch (err) {
    alert(`Erreur: ${err.message}`);
  } finally {
    predictBtn.disabled = false;
    predictBtn.querySelector(".btn-text").textContent = "Prédire";
    predictBtn.querySelector(".btn-loader").hidden = true;
  }
}

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
    ? "Risque de churn détecté"
    : "Client fidèle";

  // Details
  document.getElementById("detail-prediction").textContent = isChurn ? "Churn" : "Pas de churn";
  document.getElementById("detail-proba").textContent = `${(proba * 100).toFixed(1)}%`;

  let riskLevel, riskColor;
  if (proba >= 0.6) { riskLevel = "Élevé"; riskColor = "var(--danger)"; }
  else if (proba >= 0.35) { riskLevel = "Moyen"; riskColor = "var(--warning)"; }
  else { riskLevel = "Faible"; riskColor = "var(--success)"; }

  const riskEl = document.getElementById("detail-risk");
  riskEl.textContent = riskLevel;
  riskEl.style.color = riskColor;

  // Recommendation
  const recEl = document.getElementById("recommendation");
  if (proba >= 0.6) {
    recEl.innerHTML = "💡 <strong>Action urgente recommandée :</strong> Proposer un geste commercial (remise, upgrade), contacter le client proactivement, et analyser les tickets support récents.";
    recEl.style.borderLeftColor = "var(--danger)";
    recEl.style.background = "var(--danger-bg)";
  } else if (proba >= 0.35) {
    recEl.innerHTML = "💡 <strong>Surveillance recommandée :</strong> Planifier un contact de satisfaction, vérifier les tickets ouverts, et proposer un contrat à durée plus longue.";
    recEl.style.borderLeftColor = "var(--warning)";
    recEl.style.background = "var(--warning-bg)";
  } else {
    recEl.innerHTML = "💡 <strong>Client satisfait :</strong> Maintenir la qualité de service. Opportunité de cross-sell ou upsell.";
    recEl.style.borderLeftColor = "var(--success)";
    recEl.style.background = "var(--success-bg)";
  }

  // Remove animation class after it completes
  setTimeout(() => resultContent.classList.remove("pop-in"), 500);
}

// ── Presets ───────────────────────────────────
document.querySelectorAll(".preset-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    const preset = JSON.parse(btn.dataset.preset);
    Object.entries(preset).forEach(([key, value]) => {
      const input = document.getElementById(key);
      if (input) {
        input.value = value;
        // Sync range slider
        const range = document.getElementById(`${key}-range`);
        if (range) range.value = value;
      }
    });
    // Auto-predict
    predict();
  });
});
