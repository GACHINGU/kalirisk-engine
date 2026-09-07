# KaliRisk

**KaliRisk** is a credit risk and portfolio decisioning engine. It answers two connected questions that every lender has to get right:

1. **Will this specific applicant default?** — a probability-of-default (PD) model trained on real historical lending data.
2. **Where should the approval cutoff be set across an entire portfolio of applicants**, to maximize profit without breaching a maximum acceptable Expected Loss?

Most credit risk projects stop at the first question. KaliRisk answers both, using the same trained model.

It's trained and validated on real historical lending data — Lending Club USA, 2007–2018 — with a strict chronological train/test split (trained on 2007–2017, tested on a completely unseen 2018) to avoid leaking the future into the past. The PD model (LightGBM) reaches a ROC-AUC of **0.689**.

---

## Live

- **Dashboard (Streamlit UI):** [kalirisk-engine-v1.streamlit.app](https://kalirisk-engine-v1.streamlit.app/)
- **API (FastAPI, Render):** [kalirisk-api.onrender.com](https://kalirisk-api.onrender.com/) — interactive docs at `/docs`
- **Docker image:** [hub.docker.com/r/legendium/kalirisk-api](https://hub.docker.com/r/legendium/kalirisk-api)
- **Source:** this repository

> Note: the free Render tier spins down after inactivity — the first request after a while may take 30–60 seconds to respond.

---

## Architecture

KaliRisk follows a strict 4-layer separation:

| Layer | Contains | Folder |
|---|---|---|
| **Presentation** | FastAPI endpoints, Streamlit dashboard | `app/`, `dashboard/` |
| **Application** | Orchestration — calls domain logic in the right order | `services/` |
| **Domain** | Risk (model), financial (EL, profit), portfolio (optimizer) logic | `domain/` |
| **Data** | Ingestion, validation, feature engineering | `data_pipeline/` |

**The API is a fully standalone product.** It doesn't know or care what's calling it. Any frontend, in any language, can call `/decide`, `/evaluate`, `/explain`, or `/model-insights` directly, with zero KaliRisk-specific logic required on the client side. The only requirement is that the request body matches the expected shape (enforced automatically by Pydantic) — mismatched data returns a clear `422` error, not a silent failure or a wrong answer.

The Streamlit dashboard is a pure display client. It contains no business logic of its own — it only calls the API and renders what comes back, the same as any other client would.

---

## API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/decide` | POST | Given a batch of applicants, finds the optimal approval cutoff (maximizes profit, constrained by max Expected Loss ratio) |
| `/evaluate` | POST | Given a batch of applicants and one specific cutoff, honestly reports the outcome — including if it breaches the safety limit |
| `/explain` | POST | Given one applicant, returns a SHAP-based explanation of their specific prediction |
| `/model-insights` | GET | Returns the model's global feature importance |
| `/train` | POST | Retrains the model on the fixed, known dataset (no caller-supplied path, to prevent path traversal) |

Full request/response schemas are available at `/docs` on the live API.

---

## Tech Stack

- **Model:** LightGBM (gradient boosting), chronological train/test split
- **Explainability:** SHAP (per-applicant), LightGBM feature importances (global)
- **Backend:** FastAPI, Pydantic, Uvicorn
- **Frontend:** Streamlit
- **Data:** pandas, NumPy
- **Testing:** pytest, `TestClient`, `unittest.mock`
- **Deployment:** Docker, Render (API), Streamlit Community Cloud (dashboard)
- **CI/CD:** GitHub Actions — runs the full test suite on every push, then (only if tests pass) builds and pushes a fresh Docker image to Docker Hub. Render auto-deploys the API only after CI succeeds.

---

## Running Locally

```bash
git clone https://github.com/GACHINGU/kalirisk-engine.git
cd kalirisk-engine
pip install -r requirements.txt

# Terminal 1 — API
uvicorn app.main:app --reload

# Terminal 2 — Dashboard
streamlit run dashboard/app.py
```

The dashboard defaults to `http://127.0.0.1:8000` for the API. To point it at a different backend (e.g. the live Render deployment), set the `API_URL` environment variable before launching.

### Running via Docker

```bash
docker pull legendium/kalirisk-api
docker run -p 8000:8000 legendium/kalirisk-api
```

---

## Testing & CI/CD

The project has 35 tests covering the full pipeline — data validation, feature engineering, model training, the financial engine, the optimizer, and every API endpoint. Two tests that require the full 2.26M-row historical dataset are intentionally skipped in CI (they run locally only).

```bash
python -m pytest -v
```

On every push to `main`, GitHub Actions:
1. Runs the full test suite
2. If tests pass, logs into Docker Hub, builds a fresh image, and pushes it
3. Render, watching for a successful CI check, automatically deploys the API

A failing test blocks both the Docker push and the deployment — nothing broken reaches production.

---

## Key Engineering Decisions

- **Chronological split, not random.** Loans are trained on 2007–2017 and tested on 2018 — a genuinely unseen future — rather than a random shuffle, which would let future information leak into training and produce an artificially inflated, untrustworthy accuracy score.
- **The optimizer is constrained, not greedy.** Maximizing profit alone pushes toward approving nearly everyone, since interest income compounds while Expected Loss is a one-time deduction. The optimizer only considers cutoffs that keep the portfolio's total Expected-Loss-to-volume ratio under a defined safety limit — profit is maximized *within* that boundary, never outside it.
- **Explainability is two different questions.** Global feature importance describes what the model relies on *in general*. SHAP values answer *why this specific applicant* got their specific score — verified mathematically to sum (through a sigmoid) to the model's real predicted probability, not just plausible-looking numbers.
- **The model never sees leakage-prone fields.** `loan_status`, `issue_d`, and `id` are excluded from training by contract, not convention — enforced in code, not just in intent.

---

## License

MIT