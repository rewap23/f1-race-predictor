# F1 Podium Predictor

A machine learning system that predicts Formula 1 race podium probabilities from pre-race qualifying data.

## What it does
- Trained on 7 seasons of F1 race data (2018–2025) using FastF1
- Random Forest classifier achieving AUC 0.951 on the 2025 test season
- Features include qualifying gap to pole, driver/team rolling form, track history, and championship standings
- FastAPI backend serving predictions via REST endpoint
- HTML/CSS/JS frontend with podium visualization and full probability rankings

## Project structure
f1-race-predictor/
├── build_race_dataset.py   # Phase 1: pulls raw data via FastF1
├── phase2_features.ipynb   # Phase 2: feature engineering
├── phase3_models.ipynb     # Phase 3: model training and comparison
├── app.py                  # Phase 5: FastAPI prediction server
├── index.html              # Phase 5: frontend dashboard
├── model/                  # Saved random forest + feature list
│   ├── random_forest.pkl
│   └── feature_cols.pkl
└── requirements.txt

# Setup
```bash
# Create and activate virtual environment
python -m venv f1env
source f1env/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Pull race data (takes 20-30 mins first run)
python build_race_dataset.py

# Start the API server
uvicorn app:app --reload

# Open index.html in your browser
```

## Tech stack
Python · FastF1 · pandas · scikit-learn · XGBoost · FastAPI · HTML/CSS/JS

