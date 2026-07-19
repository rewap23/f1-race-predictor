# F1 Podium Predictor

A full-stack machine learning application that predicts Formula 1 race podium probabilities from pre-race qualifying data. Built end-to-end as a learning project covering data engineering, feature engineering, model selection, and production deployment.

**Live demo:** https://rewap23.github.io/f1-race-predictor/ 

**Stack:** Python · FastF1 · pandas · scikit-learn · XGBoost · FastAPI · HTML/CSS/JS

---

## What it does

After Saturday qualifying, a user enters each driver's grid position and current-season statistics. The model outputs a ranked podium probability for all 22 drivers, visualized as a gold/silver/bronze podium with a full probability bar chart.

The backend is a REST API built with FastAPI. The frontend is a single-page HTML/CSS/JS dashboard — no framework — that calls the API and renders results in the browser. The model is a Random Forest classifier trained on 7 seasons of Formula 1 race data.

---

## ML pipeline

### Phase 1 — Data engineering
- Pulled 7 seasons of race results (2018–2025) via the **FastF1** Python library
- Built a caching layer so re-runs don't hit the API again
- Assembled a flat CSV: one row per driver per race, ~2,900 rows across 149 races

### Phase 2 — Feature engineering
All features are computed exclusively from pre-race information to prevent data leakage:

| Feature | Description | How it's built |
|---------|-------------|---------------|
| `GridPosition` | Qualifying position | Raw from FastF1 |
| `QualiGapToPole` | Gap to pole in seconds | Fastest Q3/Q2/Q1 time minus pole time |
| `DriverFormLast5` | Avg finish over last 5 races | Rolling mean with `shift(1)` to prevent leakage |
| `TeamFormLast5` | Team avg finish over last 5 races | Same rolling approach, grouped by constructor |
| `TrackHistory` | Historical avg finish at this circuit | Rolling mean grouped by driver + circuit |
| `PointsFinishesLast5` | Points finishes in last 5 races | Rolling sum of binary top-10 indicator |
| `CumulativeDriverPoints` | Championship points before this race | Cumulative sum with `shift(1)` per season |
| `CumulativeTeamPoints` | Constructor points before this race | Same, grouped by team |
| `HomeRace` | Racing in home country | Driver nationality matched to circuit country |

Key engineering decision: every rolling feature uses `.shift(1)` before `.rolling().mean()` to ensure no race's own result leaks into its own feature value.

### Phase 3 — Model training and selection
Trained five models on a **time-based split** (train: 2018–2024, test: 2025) — random shuffling was deliberately avoided since it would leak future races into training:

| Model | AUC | Precision | Recall | Train/Test gap |
|-------|-----|-----------|--------|----------------|
| Baseline (grid pos rule) | — | — | — | — |
| Logistic Regression | 0.942 | 0.758 | 0.653 | — |
| Decision Tree (depth 3) | 0.943 | 0.674 | 0.806 | small |
| Decision Tree (unlimited) | — | — | — | **0.136** (overfit) |
| Random Forest | **0.951** | 0.716 | 0.736 | **0.015** |
| XGBoost | 0.944 | 0.712 | 0.722 | 0.041 |

**Winner: Random Forest** — best AUC and smallest generalization gap. The unlimited decision tree was intentionally included to demonstrate overfitting (train accuracy 1.000, test 0.864), contrasted with the random forest's tight 1.5% gap achieved through bagging.

XGBoost underperformed the random forest here — a deliberate finding worth noting. With ~2,900 training rows, the dataset is small enough that XGBoost's higher model capacity led to more overfitting than the forest's variance-reducing bagging approach.

### Phase 4 — Evaluation on real 2026 data
Validated the model against the 2026 Monaco GP using real qualifying data. The model correctly predicted Antonelli (P1 start) and Hamilton (P3 start) on the podium. It could not predict Gasly's P3 finish (started P9) — he benefited from post-race penalty decisions handed to other drivers, which no pre-race model can foresee. This limitation is documented in the app's About page.

---

## System architecture

```
┌─────────────────────────────┐
│   index.html                │  Vanilla HTML/CSS/JS
│   Single-page dashboard     │  No framework — fetch() API calls
│   Podium visual + rankings  │  Tab-based UI: Past Race / Upcoming / About
└────────────┬────────────────┘
             │ POST /predict
             │ JSON payload: [{Driver, GridPosition, QualiGapToPole, ...}]
             ▼
┌─────────────────────────────┐
│   app.py — FastAPI server   │  REST API with Pydantic validation
│   GET  /                    │  Health check
│   POST /predict             │  Returns ranked podium probabilities
└────────────┬────────────────┘
             │ joblib.load()
             ▼
┌─────────────────────────────┐
│   random_forest.pkl         │  Trained sklearn RandomForestClassifier
│   feature_cols.pkl          │  Ordered feature list for consistent input
└─────────────────────────────┘
```

---

## Key decisions and learnings

**Why time-based splitting matters**: using a random train/test split on sequential race data would let the model "see" future races during training, producing optimistic test metrics that collapse in real deployment. Every split in this project respects chronological order.

**Why leakage prevention is non-trivial**: rolling features computed naively include the current row's own value. Using `.shift(1)` before `.rolling()` ensures each race's feature only reflects information from strictly prior races — a mistake that's easy to miss and hard to detect without careful numerical verification.

**Why more complex ≠ better**: XGBoost is generally considered the strongest algorithm for tabular data, but the random forest outperformed it here. With a dataset of ~2,900 rows, XGBoost's capacity to fit complex patterns became a liability rather than an asset. Model selection requires empirical comparison, not assumptions.

**What a model can't predict**: in-race events (crashes, mechanical failures, safety cars, steward penalties) are structurally invisible to any pre-race model. Documenting and surfacing these limitations in the UI is part of building an honest ML system.

---

## Skills demonstrated

- **Data engineering**: API integration with FastF1, caching strategy, multi-season data assembly
- **Feature engineering**: rolling window features, leakage prevention, time-series aware transformations
- **Machine learning**: binary classification, model comparison, overfitting diagnosis, AUC/precision/recall evaluation
- **ML systems thinking**: train/test split design for time-series data, model serialization with joblib
- **Backend development**: REST API design with FastAPI, Pydantic data validation, CORS configuration
- **Frontend development**: single-page application in vanilla JS, fetch API, dynamic DOM rendering
- **Software engineering**: modular project structure, separation of data/feature/model layers, .gitignore hygiene, requirements management

---

## Project structure

```
f1-race-predictor/
├── build_race_dataset.py     # Phase 1: FastF1 data pipeline
├── phase2_features.ipynb     # Phase 2: feature engineering notebook
├── phase3_models.ipynb       # Phase 3: model training and comparison
├── predict_2026.py           # Phase 4: validation on 2026 Monaco GP
├── app.py                    # Phase 5: FastAPI backend
├── index.html                # Phase 5: frontend dashboard
├── model/
│   ├── random_forest.pkl     # Trained model (joblib)
│   └── feature_cols.pkl      # Feature column order
├── requirements.txt
└── README.md
```

---

## Running locally

```bash
# 1. Clone and set up environment
git clone https://github.com/rewap23/f1-race-predictor.git
cd f1-race-predictor
python -m venv f1env
source f1env/bin/activate      # Windows: f1env\Scripts\activate
pip install -r requirements.txt

# 2. Pull race data (first run takes 20–30 mins, cached after)
python build_race_dataset.py

# 3. Run feature engineering and model training
# Open phase2_features.ipynb and phase3_models.ipynb in Jupyter
# and run all cells in order

# 4. Start the API server
uvicorn app:app --reload

# 5. Open index.html in your browser
# Select a past race or enter upcoming qualifying data and hit Predict
```

---

## Model limitations

This predictor works from pre-race information only. It cannot account for:
- Safety car periods, red flags, or race restarts
- Mechanical failures and retirements
- Crashes that eliminate front-running drivers
- Post-race penalty decisions by stewards
- Rapidly changing weather conditions mid-race

These limitations are documented in the app's About tab with a concrete example from the 2026 Monaco GP.