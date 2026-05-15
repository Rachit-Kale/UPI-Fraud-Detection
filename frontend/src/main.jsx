import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  AlertTriangle,
  BadgeIndianRupee,
  BarChart3,
  Blocks,
  Clock3,
  Cpu,
  Database,
  GitBranch,
  Gauge,
  Layers3,
  MapPin,
  Network,
  Radar,
  RefreshCw,
  ShieldCheck,
  Sparkles,
  WalletCards,
} from "lucide-react";
import "./styles.css";

const API_BASE = "/api";

const defaultForm = {
  amount: 650,
  transaction_type: "TRANSFER",
  device_type: "Android",
  merchant_category: "Personal",
  timestamp: new Date().toISOString().slice(0, 16),
  sender_id: "user_1024",
  receiver_id: "user_2048",
  location: "Mumbai",
};

function App() {
  const [page, setPage] = useState("simulation");
  const [form, setForm] = useState(defaultForm);
  const [status, setStatus] = useState(null);
  const [presets, setPresets] = useState([]);
  const [analytics, setAnalytics] = useState(null);
  const [result, setResult] = useState(null);
  const [logs, setLogs] = useState(() => {
    const saved = localStorage.getItem("upi_prediction_logs");
    return saved ? JSON.parse(saved) : [];
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    async function bootstrap() {
      try {
        const [statusResponse, presetResponse, analyticsResponse] = await Promise.all([
          fetch(`${API_BASE}/models/status`),
          fetch(`${API_BASE}/presets`),
          fetch(`${API_BASE}/analytics/data`),
        ]);
        setStatus(await statusResponse.json());
        setPresets(await presetResponse.json());
        setAnalytics(await analyticsResponse.json());
      } catch {
        setError("Local API is not reachable. Start it with: uvicorn api.main:app --reload");
      }
    }
    bootstrap();
  }, []);

  useEffect(() => {
    localStorage.setItem("upi_prediction_logs", JSON.stringify(logs.slice(0, 10)));
  }, [logs]);

  function updateField(field, value) {
    setForm((current) => ({ ...current, [field]: value }));
  }

  function applyPreset(preset) {
    const timestamp = new Date();
    timestamp.setHours(preset.hour, 0, 0, 0);
    setForm({
      amount: preset.amount,
      transaction_type: preset.transaction_type,
      device_type: preset.device_type,
      merchant_category: preset.merchant_category,
      timestamp: timestamp.toISOString().slice(0, 16),
      sender_id: preset.sender_id,
      receiver_id: preset.receiver_id,
      location: preset.location,
    });
    setResult(null);
  }

  async function runPrediction(event) {
    event.preventDefault();
    setLoading(true);
    setError("");

    try {
      const response = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...form,
          amount: Number(form.amount),
          timestamp: new Date(form.timestamp).toISOString(),
        }),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Prediction failed.");
      }
      setResult(data);
      setLogs((current) => [
        {
          time: new Date().toLocaleTimeString(),
          amount: Number(form.amount),
          type: form.transaction_type,
          fraud: data.supervised.fraud_probability,
          anomaly: data.anomaly.anomaly_score,
          confidence: data.anomaly.anomaly_confidence,
          label: data.anomaly.anomaly_label,
        },
        ...current,
      ]);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="shell">
      <section className="hero">
        <div>
          <div className="eyebrow">
            <Sparkles size={16} />
            Offline Research Dashboard
          </div>
          <h1>UPI Fraud Lab</h1>
          <p>
            Explore imported transaction data, trace the offline ML workflow, and test
            supervised and anomaly signals without a final fusion engine.
          </p>
        </div>
        <StatusPanel status={status} />
      </section>

      <nav className="page-tabs">
        <TabButton active={page === "simulation"} icon={<Radar size={18} />} label="Simulation" onClick={() => setPage("simulation")} />
        <TabButton active={page === "workflow"} icon={<GitBranch size={18} />} label="Workflow" onClick={() => setPage("workflow")} />
        <TabButton active={page === "analytics"} icon={<BarChart3 size={18} />} label="Analytics" onClick={() => setPage("analytics")} />
      </nav>

      {error && (
        <div className="notice">
          <AlertTriangle size={18} />
          {error}
        </div>
      )}

      {page === "simulation" && (
        <SimulationPage
          form={form}
          result={result}
          logs={logs}
          loading={loading}
          presets={presets}
          updateField={updateField}
          applyPreset={applyPreset}
          runPrediction={runPrediction}
        />
      )}
      {page === "workflow" && <WorkflowPage />}
      {page === "analytics" && <AnalyticsPage analytics={analytics} />}
    </main>
  );
}

function SimulationPage({ form, result, logs, loading, presets, updateField, applyPreset, runPrediction }) {
  const fraudProbability = result?.supervised?.fraud_probability ?? 0;
  const anomalyConfidence = result?.anomaly?.anomaly_confidence ?? 0;

  return (
    <section className="workspace">
      <form className="control-panel" onSubmit={runPrediction}>
        <div className="panel-head">
          <div>
            <span>Manual Transaction</span>
            <h2>Simulation Input</h2>
          </div>
          <button type="submit" className="run-button" disabled={loading}>
            {loading ? <RefreshCw className="spin" size={18} /> : <Radar size={18} />}
            {loading ? "Testing" : "Run Test"}
          </button>
        </div>

        <PresetStrip presets={presets} onApply={applyPreset} />

        <div className="field-grid">
          <NumberField icon={<BadgeIndianRupee size={18} />} label="Amount" value={form.amount} onChange={(value) => updateField("amount", value)} />
          <SelectField icon={<WalletCards size={18} />} label="Transaction Type" value={form.transaction_type} options={["PAYMENT", "TRANSFER", "CASH_IN", "CASH_OUT", "DEBIT", "UPI", "TOP_UP"]} onChange={(value) => updateField("transaction_type", value)} />
          <SelectField icon={<Cpu size={18} />} label="Device Type" value={form.device_type} options={["Android", "iOS", "Web", "POS", "Unknown"]} onChange={(value) => updateField("device_type", value)} />
          <TextField label="Merchant Category" value={form.merchant_category} onChange={(value) => updateField("merchant_category", value)} />
          <TextField label="Sender ID" value={form.sender_id} onChange={(value) => updateField("sender_id", value)} />
          <TextField label="Receiver ID" value={form.receiver_id} onChange={(value) => updateField("receiver_id", value)} />
          <TextField icon={<MapPin size={18} />} label="Location" value={form.location} onChange={(value) => updateField("location", value)} />
          <DateField icon={<Clock3 size={18} />} label="Transaction Time" value={form.timestamp} onChange={(value) => updateField("timestamp", value)} />
        </div>
      </form>

      <section className="result-panel">
        <div className="panel-head">
          <div>
            <span>Model Outputs</span>
            <h2>Separate Signals</h2>
          </div>
          <Gauge size={28} />
        </div>

        <div className="signal-grid">
          <SignalCard title="Supervised Model" value={formatPercent(fraudProbability)} label={result?.supervised?.fraud_prediction ? "Fraud" : "Legitimate"} accent="red" progress={fraudProbability} />
          <SignalCard title="Anomaly Model" value={result ? result.anomaly.anomaly_score.toFixed(4) : "0.0000"} label={result?.anomaly?.anomaly_label ?? "Waiting"} accent="teal" progress={anomalyConfidence} />
        </div>

        <ComparisonChart amount={Number(form.amount)} fraudProbability={fraudProbability} anomalyConfidence={anomalyConfidence} />
        <EvidencePanel diagnostics={result?.diagnostics} />
        <LogTable logs={logs} />
      </section>
    </section>
  );
}

function WorkflowPage() {
  const steps = [
    ["Data Loading", "Kaggle and Zenodo files are loaded from data/raw.", Database],
    ["Schema Mapping", "Dataset-specific columns are normalized into one transaction schema.", Blocks],
    ["Preprocessing", "Missing values, scaling, encoding, and outlier handling prepare the model matrix.", Layers3],
    ["Feature Engineering", "Behavioral, velocity, risk, and temporal signals are generated offline.", Activity],
    ["Supervised Models", "XGBoost and Random Forest produce fraud probability scores.", ShieldCheck],
    ["Anomaly Models", "Isolation Forest and LOF produce separate anomaly scores.", Network],
    ["Dashboard Testing", "Manual transactions compare both model families without risk fusion.", Gauge],
  ];

  return (
    <section className="workflow-page">
      <div className="section-head">
        <span>Project Pipeline</span>
        <h2>Offline Batch Workflow</h2>
      </div>
      <div className="workflow-canvas">
        <div className="flow-line" />
        {steps.map(([title, body, Icon], index) => (
          <article className="flow-node" style={{ "--delay": `${index * 120}ms` }} key={title}>
            <div className="node-index">{index + 1}</div>
            <Icon size={24} />
            <h3>{title}</h3>
            <p>{body}</p>
          </article>
        ))}
      </div>
      <div className="workflow-band">
        <MetricTile label="Current scope" value="2 model families" />
        <MetricTile label="Output style" value="Separate signals" />
        <MetricTile label="Fusion engine" value="Not implemented" />
      </div>
    </section>
  );
}

function AnalyticsPage({ analytics }) {
  if (!analytics?.ready) {
    return (
      <section className="result-panel solo-panel">
        <div className="panel-head">
          <div>
            <span>Imported Data</span>
            <h2>Analytics Waiting</h2>
          </div>
          <Database size={28} />
        </div>
        <p className="muted">{analytics?.message || "Analytics are loading from the local API."}</p>
      </section>
    );
  }

  const summary = analytics.summary;
  return (
    <section className="analytics-page">
      <div className="section-head">
        <span>Imported Data</span>
        <h2>Dataset Analytics</h2>
      </div>
      <div className="metric-grid">
        <MetricTile label="Mapped Transactions" value={compactNumber(summary.total_rows)} />
        <MetricTile label="Fraud Rows" value={compactNumber(summary.fraud_rows)} />
        <MetricTile label="Fraud Rate" value={formatPercent(summary.fraud_rate)} />
        <MetricTile label="Average Amount" value={currency(summary.average_amount)} />
      </div>

      <div className="analytics-grid">
        <ChartPanel title="Fraud vs Legitimate">
          <DonutChart fraud={summary.fraud_rows} legitimate={summary.legitimate_rows} />
        </ChartPanel>
        <ChartPanel title="Amount Distribution">
          <BarList data={analytics.amount_bins} />
        </ChartPanel>
        <ChartPanel title="Transaction Types">
          <BarList data={analytics.transaction_types} />
        </ChartPanel>
        <ChartPanel title="Hourly Volume">
          <SparkBars data={analytics.hourly_volume} />
        </ChartPanel>
        <ChartPanel title="Device Types">
          <BarList data={analytics.device_types} />
        </ChartPanel>
        <ChartPanel title="Top Locations">
          <BarList data={analytics.locations} />
        </ChartPanel>
      </div>
    </section>
  );
}

function StatusPanel({ status }) {
  const ready = status?.ready;
  return (
    <aside className={`status-panel ${ready ? "ready" : "waiting"}`}>
      <ShieldCheck size={24} />
      <div>
        <span>Model Status</span>
        <strong>{ready ? "Ready for testing" : "Training artifacts missing"}</strong>
      </div>
      <div className="artifact-dots">
        {Object.entries(status?.artifacts || {}).map(([key, value]) => (
          <span key={key} title={key} className={value ? "on" : ""} />
        ))}
      </div>
    </aside>
  );
}

function TabButton({ active, icon, label, onClick }) {
  return (
    <button className={active ? "active" : ""} type="button" onClick={onClick}>
      {icon}
      {label}
    </button>
  );
}

function PresetStrip({ presets, onApply }) {
  return (
    <div className="presets">
      {presets.map((preset) => (
        <button type="button" key={preset.name} onClick={() => onApply(preset)}>
          {preset.name}
        </button>
      ))}
    </div>
  );
}

function FieldWrap({ icon, label, children }) {
  return (
    <label className="field">
      <span>
        {icon}
        {label}
      </span>
      {children}
    </label>
  );
}

function NumberField({ icon, label, value, onChange }) {
  return (
    <FieldWrap icon={icon} label={label}>
      <input type="number" min="0" value={value} onChange={(event) => onChange(event.target.value)} />
    </FieldWrap>
  );
}

function TextField({ icon, label, value, onChange }) {
  return (
    <FieldWrap icon={icon} label={label}>
      <input value={value} onChange={(event) => onChange(event.target.value)} />
    </FieldWrap>
  );
}

function DateField({ icon, label, value, onChange }) {
  return (
    <FieldWrap icon={icon} label={label}>
      <input type="datetime-local" value={value} onChange={(event) => onChange(event.target.value)} />
    </FieldWrap>
  );
}

function SelectField({ icon, label, value, options, onChange }) {
  return (
    <FieldWrap icon={icon} label={label}>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {options.map((option) => (
          <option key={option} value={option}>{option}</option>
        ))}
      </select>
    </FieldWrap>
  );
}

function SignalCard({ title, value, label, accent, progress }) {
  return (
    <article className={`signal-card ${accent}`}>
      <span>{title}</span>
      <strong>{value}</strong>
      <em>{label}</em>
      <div className="meter">
        <i style={{ width: `${clampPercent(progress * 100)}%` }} />
      </div>
    </article>
  );
}

function ComparisonChart({ amount, fraudProbability, anomalyConfidence }) {
  const scaledAmount = Math.min(100, amount / 1000);
  const rows = [
    ["Amount intensity", scaledAmount],
    ["Fraud probability", fraudProbability * 100],
    ["Anomaly percentile", anomalyConfidence * 100],
  ];
  return (
    <div className="comparison">
      <div className="mini-title">
        <Activity size={17} />
        Transaction Comparison
      </div>
      {rows.map(([label, value]) => (
        <div className="bar-row" key={label}>
          <span>{label}</span>
          <div><i style={{ width: `${clampPercent(value)}%` }} /></div>
          <b>{value.toFixed(1)}</b>
        </div>
      ))}
    </div>
  );
}

function EvidencePanel({ diagnostics }) {
  if (!diagnostics) {
    return (
      <div className="evidence-panel">
        <div className="mini-title">Model Evidence</div>
        <p>No evidence run yet. Submit a transaction to see repeatability and input sensitivity checks.</p>
      </div>
    );
  }

  const repeatFraud = diagnostics.deterministic.repeat_fraud_delta;
  const repeatAnomaly = diagnostics.deterministic.repeat_anomaly_delta;
  const sensitivity = diagnostics.sensitivity || [];

  return (
    <div className="evidence-panel">
      <div className="mini-title">Model Evidence</div>
      <div className="evidence-summary">
        <span>Same input repeat fraud delta: <strong>{formatDelta(repeatFraud)}</strong></span>
        <span>Same input repeat anomaly delta: <strong>{formatDelta(repeatAnomaly)}</strong></span>
      </div>
      <div className="sensitivity-list">
        {sensitivity.map((item) => (
          <div className="sensitivity-row" key={item.name}>
            <span>{item.name}</span>
            <em>Fraud {formatSignedPercent(item.fraud_delta)}</em>
            <small>Anomaly {formatSignedPercent(item.anomaly_delta)}</small>
          </div>
        ))}
      </div>
    </div>
  );
}

function LogTable({ logs }) {
  return (
    <div className="logs">
      <div className="mini-title">Prediction Logs</div>
      {logs.length === 0 ? (
        <p>No prediction runs yet.</p>
      ) : (
        logs.map((log, index) => (
          <div className="log-row" key={`${log.time}-${index}`}>
            <span>{log.time}</span>
            <strong>{log.type}</strong>
            <em>{formatPercent(log.fraud)}</em>
            <small>{log.label}</small>
          </div>
        ))
      )}
    </div>
  );
}

function MetricTile({ label, value }) {
  return (
    <article className="metric-tile">
      <span>{label}</span>
      <strong>{value}</strong>
    </article>
  );
}

function ChartPanel({ title, children }) {
  return (
    <article className="chart-panel">
      <h3>{title}</h3>
      {children}
    </article>
  );
}

function BarList({ data }) {
  const maxValue = Math.max(...data.map((item) => item.value), 1);
  return (
    <div className="bar-list">
      {data.map((item) => (
        <div className="bar-row wide" key={item.label}>
          <span>{item.label}</span>
          <div><i style={{ width: `${(item.value / maxValue) * 100}%` }} /></div>
          <b>{compactNumber(item.value)}</b>
        </div>
      ))}
    </div>
  );
}

function SparkBars({ data }) {
  const maxValue = Math.max(...data.map((item) => item.value), 1);
  return (
    <div className="spark-bars">
      {data.map((item) => (
        <span key={item.label} title={`${item.label}:00 - ${compactNumber(item.value)}`}>
          <i style={{ height: `${Math.max(6, (item.value / maxValue) * 100)}%` }} />
        </span>
      ))}
    </div>
  );
}

function DonutChart({ fraud, legitimate }) {
  const total = Math.max(fraud + legitimate, 1);
  const fraudPercent = (fraud / total) * 100;
  return (
    <div className="donut-wrap">
      <div className="donut" style={{ "--fraud": `${fraudPercent}%` }}>
        <strong>{fraudPercent.toFixed(2)}%</strong>
        <span>Fraud</span>
      </div>
      <div className="legend">
        <span><i className="fraud-dot" /> Fraud {compactNumber(fraud)}</span>
        <span><i className="ok-dot" /> Legitimate {compactNumber(legitimate)}</span>
      </div>
    </div>
  );
}

function formatPercent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function formatSignedPercent(value) {
  const percent = (value * 100).toFixed(1);
  return `${value >= 0 ? "+" : ""}${percent}%`;
}

function formatDelta(value) {
  return value < 0.000001 ? "0.000000" : value.toFixed(6);
}

function compactNumber(value) {
  return Intl.NumberFormat("en-IN", { notation: "compact", maximumFractionDigits: 1 }).format(value || 0);
}

function currency(value) {
  return Intl.NumberFormat("en-IN", { maximumFractionDigits: 0 }).format(value || 0);
}

function clampPercent(value) {
  return Math.min(100, Math.max(0, value || 0));
}

createRoot(document.getElementById("root")).render(<App />);
