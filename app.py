
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware # allows the backend to communicate with the frontend
# the above imports are necessary for the API to function correctly
# Pydantic is a Python library that enforces data validation 
# and type constraints at runtime using standard Python type hints


rf = joblib.load("model/random_forest.pkl")
feature_cols = joblib.load("model/feature_cols.pkl")

app = FastAPI(title="F1 Podium Predictor")

class DriverFeatures(BaseModel):
    Driver: str
    GridPosition: int
    QualiGapToPole: float
    DriverFormLast5: float
    TeamFormLast5: float
    TrackHistory: float
    PointsFinishesLast5: float
    HomeRace: int
    CumulativeDriverPoints: float
    CumulativeTeamPoints: float

# A prediction request is a list of drivers — all 20-22 on the grid
class PredictRequest(BaseModel):
    drivers: List[DriverFeatures]

@app.get("/")
def health_check():
    return {"status": "ok", "model": "random_forest"}

@app.post("/predict")
def predict(request: PredictRequest):
    # Convert the incoming list of drivers into a DataFrame
    df = pd.DataFrame([d.model_dump() for d in request.drivers])

    # Run every driver through the model at once
    probabilities = rf.predict_proba(df[feature_cols])[:, 1]

    # Attach probabilities and sort by most likely to podium
    df["PodiumProbability"] = probabilities
    df = df.sort_values("PodiumProbability", ascending=False).reset_index(drop=True)

    # Build the response
    results = []
    for rank, row in df.iterrows():
        results.append({
            "PredictedRank": rank + 1,
            "Driver": row["Driver"],
            "GridPosition": int(row["GridPosition"]),
            "PodiumProbability": round(float(row["PodiumProbability"]), 4),
        })

    return {
        "predictions": results,
        "predicted_podium": [r["Driver"] for r in results[:3]]
    }

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)

