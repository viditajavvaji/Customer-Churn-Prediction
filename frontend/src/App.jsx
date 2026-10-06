import { useEffect, useState } from 'react';

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const fields = [
  { key: 'gender', label: 'Gender', type: 'select', options: ['Female', 'Male'] },
  { key: 'SeniorCitizen', label: 'Senior citizen', type: 'select', options: ['0', '1'], display: ['No', 'Yes'] },
  { key: 'Partner', label: 'Partner', type: 'select', options: ['Yes', 'No'] },
  { key: 'Dependents', label: 'Dependents', type: 'select', options: ['Yes', 'No'] },
  { key: 'tenure', label: 'Tenure (months)', type: 'number', min: 0, max: 72 },
  { key: 'PhoneService', label: 'Phone service', type: 'select', options: ['Yes', 'No'] },
  { key: 'MultipleLines', label: 'Multiple lines', type: 'select', options: ['No', 'Yes', 'No phone service'] },
  { key: 'InternetService', label: 'Internet service', type: 'select', options: ['DSL', 'Fiber optic', 'No'] },
  { key: 'OnlineSecurity', label: 'Online security', type: 'select', options: ['No', 'Yes', 'No internet service'] },
  { key: 'OnlineBackup', label: 'Online backup', type: 'select', options: ['No', 'Yes', 'No internet service'] },
  { key: 'DeviceProtection', label: 'Device protection', type: 'select', options: ['No', 'Yes', 'No internet service'] },
  { key: 'TechSupport', label: 'Tech support', type: 'select', options: ['No', 'Yes', 'No internet service'] },
  { key: 'StreamingTV', label: 'Streaming TV', type: 'select', options: ['No', 'Yes', 'No internet service'] },
  { key: 'StreamingMovies', label: 'Streaming movies', type: 'select', options: ['No', 'Yes', 'No internet service'] },
  { key: 'Contract', label: 'Contract', type: 'select', options: ['Month-to-month', 'One year', 'Two year'] },
  { key: 'PaperlessBilling', label: 'Paperless billing', type: 'select', options: ['Yes', 'No'] },
  { key: 'PaymentMethod', label: 'Payment method', type: 'select', options: ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'] },
  { key: 'MonthlyCharges', label: 'Monthly charges ($)', type: 'number', min: 18.25, max: 118.75, step: '0.01' },
  { key: 'TotalCharges', label: 'Total charges ($)', type: 'number', min: 0, max: 10000, step: '0.01' },
];

const initialForm = Object.fromEntries(fields.map(field => [field.key, field.type === 'select' ? field.options[0] : '']));

function InputField({ field, value, onChange }) {
  return (
    <label className="input-field">
      <span>{field.label}</span>
      {field.type === 'select' ? (
        <select value={value} onChange={event => onChange(field.key, event.target.value)}>
          {field.options.map((option, index) => <option value={option} key={option}>{field.display?.[index] ?? option}</option>)}
        </select>
      ) : (
        <input type="number" min={field.min} max={field.max} step={field.step || '1'} value={value} required onChange={event => onChange(field.key, event.target.value)} />
      )}
    </label>
  );
}

function App() {
  const [page, setPage] = useState('choice');
  const [darkMode, setDarkMode] = useState(() => window.localStorage.getItem('churnwise-theme') === 'dark');
  const [form, setForm] = useState(initialForm);
  const [result, setResult] = useState(null);
  const [csvFile, setCsvFile] = useState(null);
  const [batchResults, setBatchResults] = useState([]);
  const [selectedBatchResult, setSelectedBatchResult] = useState(null);
  const [modelReady, setModelReady] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    document.documentElement.dataset.theme = darkMode ? 'dark' : 'light';
    window.localStorage.setItem('churnwise-theme', darkMode ? 'dark' : 'light');
  }, [darkMode]);

  useEffect(() => {
    fetch(`${API}/health`)
      .then(response => response.json())
      .then(data => setModelReady(Boolean(data.model_loaded)))
      .catch(() => setModelReady(false));
  }, []);

  function updateField(key, value) {
    setForm(current => ({ ...current, [key]: value }));
  }

  async function submitSingle(event) {
    event.preventDefault();
    setError('');
    setLoading(true);
    const customer = {
      ...form,
      SeniorCitizen: Number(form.SeniorCitizen),
      tenure: Number(form.tenure),
      MonthlyCharges: Number(form.MonthlyCharges),
      TotalCharges: Number(form.TotalCharges),
    };
    try {
      const response = await fetch(`${API}/predict`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(customer),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Prediction failed.');
      setResult(data);
      setPage('single-result');
    } catch (err) {
      setError(err.message || 'Could not contact the prediction API. Make sure the backend is running.');
    } finally {
      setLoading(false);
    }
  }

  async function submitCsv(event) {
    event.preventDefault();
    if (!csvFile) return;
    setError('');
    setLoading(true);
    try {
      const response = await fetch(`${API}/predict-csv`, {
        method: 'POST',
        headers: { 'Content-Type': 'text/csv', 'X-Filename': csvFile.name },
        body: csvFile,
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'CSV prediction failed.');
      setBatchResults(data.results || []);
      setPage('csv-result');
    } catch (err) {
      setError(err.message || 'Could not upload this CSV.');
    } finally {
      setLoading(false);
    }
  }

  function returnHome() {
    setError('');
    setPage('choice');
  }

  return (
    <main className="app">
      <header className="site-header">
        <h1>Customer Churn Prediction</h1>
        <button className="theme-toggle" type="button" aria-pressed={darkMode} onClick={() => setDarkMode(value => !value)}>
          {darkMode ? '☀ Light mode' : '☾ Dark mode'}
        </button>
      </header>

      {page === 'choice' && (
        <section className="page-card">
          <div className="page-title"><h2>What would you like to check?</h2></div>
          {modelReady === false && <div className="notice error">The prediction API or model is unavailable. Start the backend and check <code>http://localhost:8000/health</code>.</div>}
          <div className="mode-options">
            <button type="button" className="mode-option" onClick={() => { setError(''); setPage('single-input'); }}>
              <strong>One customer</strong><span>Enter one customer’s details in a form.</span>
            </button>
            <button type="button" className="mode-option" onClick={() => { setError(''); setCsvFile(null); setPage('csv-input'); }}>
              <strong>Multiple customers</strong><span>Upload a CSV file and get a prediction table.</span>
            </button>
          </div>
        </section>
      )}

      {page === 'single-input' && (
        <section className="page-card">
          <div className="page-title"><button className="back-link" type="button" onClick={returnHome}>← Back</button><h2>Customer details</h2><p>Fill in the customer’s account, services, and billing information.</p></div>
          {error && <div className="notice error" role="alert">{error}</div>}
          <form onSubmit={submitSingle}>
            <div className="input-grid">{fields.map(field => <InputField key={field.key} field={field} value={form[field.key]} onChange={updateField} />)}</div>
            <div className="form-footer"><span>All fields are required.</span><button className="primary-button" type="submit" disabled={loading}>{loading ? 'Predicting…' : 'Predict churn'}</button></div>
          </form>
        </section>
      )}

      {(page === 'single-result' || page === 'batch-detail') && (page === 'batch-detail' ? selectedBatchResult : result) && (() => {
        const shown = page === 'batch-detail' ? selectedBatchResult : result;
        return (
        <section className="page-card result-page">
          <div className="page-title"><button className="back-link" type="button" onClick={() => page === 'batch-detail' ? setPage('csv-result') : setPage('single-input')}>{page === 'batch-detail' ? '← Back to CSV results' : '← Edit customer'}</button><h2>Prediction result{shown.name ? `: ${shown.name}` : ''}</h2></div>
          <div className={`prediction-summary ${shown.churn ? 'churn' : 'stay'}`}>
            <div><span className="result-label">PREDICTION</span><h3>{shown.prediction}</h3></div>
            <div><span className="result-label">RISK LEVEL</span><h3>{shown.risk_level}</h3></div>
            <div><span className="result-label">CHURN PROBABILITY</span><h3>{Number(shown.churn_probability).toFixed(2)}%</h3></div>
          </div>
          <div className="result-grid">
            <section className="result-section"><h3>Factors increasing risk</h3>{shown.risk_factors?.length ? <ul>{shown.risk_factors.map(item => <li key={item.name}>{item.name}</li>)}</ul> : <p>No factor details available.</p>}</section>
            <section className="result-section"><h3>Factors reducing risk</h3>{shown.protective_factors?.length ? <ul>{shown.protective_factors.map(item => <li key={item.name}>{item.name}</li>)}</ul> : <p>No factor details available.</p>}</section>
          </div>
          <section className="result-section recommendations"><h3>Suggested retention actions</h3><ul>{shown.recommendations?.map(item => <li key={item}>{item}</li>)}</ul></section>
          <div className="form-footer result-footer"><button className="secondary-button" type="button" onClick={page === 'batch-detail' ? () => setPage('csv-result') : returnHome}>{page === 'batch-detail' ? 'Back to CSV results' : 'Start over'}</button></div>
        </section>
        );
      })()}

      {page === 'csv-input' && (
        <section className="page-card">
          <div className="page-title"><button className="back-link" type="button" onClick={returnHome}>← Back</button><h2>Upload customer CSV</h2><p>Upload one row per customer. The model will predict every row.</p></div>
          <div className="csv-help">Use the 19 input columns from the training dataset. Add an optional <code>Name</code> column to identify each customer. Blank <code>TotalCharges</code> values are treated as 0. <a href="/customer-upload-template.csv" download>Download the CSV header template</a></div>
          {error && <div className="notice error" role="alert">{error}</div>}
          <form onSubmit={submitCsv}>
            <label className="file-field"><span>CSV file</span><input type="file" accept=".csv,text/csv" required onChange={event => setCsvFile(event.target.files?.[0] || null)} /></label>
            {csvFile && <p className="selected-file">Selected: {csvFile.name}</p>}
            <div className="form-footer"><span>Maximum 10,000 customer rows per upload.</span><button className="primary-button" type="submit" disabled={loading || !csvFile}>{loading ? 'Analyzing CSV…' : 'Predict customers'}</button></div>
          </form>
        </section>
      )}

      {page === 'csv-result' && (
        <section className="page-card batch-page">
          <div className="page-title"><button className="back-link" type="button" onClick={() => setPage('csv-input')}>← Upload another file</button><h2>CSV prediction results</h2></div>
          <div className="table-scroll"><table className="results-table"><thead><tr><th>Name</th><th>Churn</th><th>Probability</th><th>Risk</th></tr></thead>
            <tbody>{batchResults.map((row, index) => <tr key={`${row.name}-${index}`} onClick={() => { setSelectedBatchResult(row); setPage('batch-detail'); }} tabIndex="0" onKeyDown={event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); setSelectedBatchResult(row); setPage('batch-detail'); } }}>
              <td>{row.name}</td><td className={row.churn ? 'churn-cell' : 'stay-cell'}>{row.churn ? 'Yes' : 'No'}</td><td>{Number(row.churn_probability).toFixed(2)}%</td><td>{row.risk_level}</td>
            </tr>)}</tbody>
          </table></div>
          <p className="table-hint">Select a customer to view their factors and suggested retention actions.</p>
        </section>
      )}

    </main>
  );
}

export default App;
