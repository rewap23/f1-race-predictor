"""
predict_2026.py

Tests the trained random forest model on the 2026 Monaco Grand Prix.
Uses real qualifying data from June 6 2026 as model input, then
compares predictions against what actually happened on race day.

Run with:
    python predict_2026.py

ACTUAL RACE RESULT (June 7 2026):
    P1: Kimi Antonelli  (started P1)
    P2: Lewis Hamilton  (started P3)
    P3: Pierre Gasly    (started P9 — benefited from penalties to others)
"""

import joblib
import pandas as pd

# ── Load saved model ───────────────────────────────────────────────────────
rf = joblib.load("model/random_forest.pkl")
feature_cols = joblib.load("model/feature_cols.pkl")

# ── 2026 Monaco GP — Round 6 ───────────────────────────────────────────────
#
# HOW EACH COLUMN WAS FILLED:
#
# GridPosition      — real qualifying result, June 6 2026
# QualiGapToPole    — seconds behind Antonelli's 1:12.051 pole lap
#                     P1-P3 confirmed from reports; P4+ estimated from
#                     typical Monaco inter-driver gaps (~0.08-0.12s per pos)
# DriverFormLast5   — estimated average finish pos over last 5 races
#                     (Rounds 1-5 of 2026: Australia, China, Japan, Bahrain,
#                     Saudi Arabia, Miami). Antonelli won all 5 so avg ~1.0.
# TeamFormLast5     — estimated average team finish pos over last 5 races
# TrackHistory      — estimated historical avg finish at Monaco (2018-2025)
#                     from training data patterns. New drivers get ~10 (neutral)
# PointsFinishesLast5 — how many of last 5 races finished in points (top 10)
# HomeRace          — 1 for Leclerc only (Monaco is his home circuit)
# CumulativeDriverPoints — championship points earned before Monaco
# CumulativeTeamPoints   — constructor points earned before Monaco
#
# Sources: GPFans, RacingNews365, Wikipedia, RaceFans (June 2026)

monaco_2026 = pd.DataFrame([
    # ── TOP 10 (Q3) ─────────────────────────────────────────────────────────
    {
        "Driver": "ANT",  # Kimi Antonelli — Mercedes — POLE
        "GridPosition": 1, "QualiGapToPole": 0.000,
        "DriverFormLast5": 1.2,   # won all 5 races so far
        "TeamFormLast5": 2.1,     # Mercedes dominant
        "TrackHistory": 10.0,     # rookie, no Monaco history → neutral fill
        "PointsFinishesLast5": 5,
        "HomeRace": 0,
        "CumulativeDriverPoints": 131,
        "CumulativeTeamPoints": 219,
    },
    {
        "Driver": "VER",  # Max Verstappen — Red Bull — P2
        "GridPosition": 2, "QualiGapToPole": 0.043,
        "DriverFormLast5": 5.4,   # some difficult races in 2026
        "TeamFormLast5": 6.8,
        "TrackHistory": 2.8,      # multiple Monaco wins in dataset
        "PointsFinishesLast5": 4,
        "HomeRace": 0,
        "CumulativeDriverPoints": 35,
        "CumulativeTeamPoints": 50,
    },
    {
        "Driver": "HAM",  # Lewis Hamilton — Ferrari — P3
        "GridPosition": 3, "QualiGapToPole": 0.214,
        "DriverFormLast5": 3.4,   # strong Ferrari form in 2026
        "TeamFormLast5": 3.8,
        "TrackHistory": 2.5,      # multiple Monaco wins historically
        "PointsFinishesLast5": 5,
        "HomeRace": 0,
        "CumulativeDriverPoints": 72,
        "CumulativeTeamPoints": 132,
    },
    {
        "Driver": "LEC",  # Charles Leclerc — Ferrari — P4 (crashed in Q3)
        "GridPosition": 4, "QualiGapToPole": 0.310,
        "DriverFormLast5": 4.2,
        "TeamFormLast5": 3.8,
        "TrackHistory": 3.8,      # good Monaco history but some crashes
        "PointsFinishesLast5": 4,
        "HomeRace": 1,            # Monaco is his home race
        "CumulativeDriverPoints": 60,
        "CumulativeTeamPoints": 132,
    },
    {
        "Driver": "HAD",  # Isack Hadjar — Red Bull — P5
        "GridPosition": 5, "QualiGapToPole": 0.390,
        "DriverFormLast5": 7.8,   # rookie, improving
        "TeamFormLast5": 6.8,
        "TrackHistory": 10.0,     # no Monaco history → neutral fill
        "PointsFinishesLast5": 3,
        "HomeRace": 0,
        "CumulativeDriverPoints": 15,
        "CumulativeTeamPoints": 50,
    },
    {
        "Driver": "RUS",  # George Russell — Mercedes — P6
        "GridPosition": 6, "QualiGapToPole": 0.452,
        "DriverFormLast5": 3.8,
        "TeamFormLast5": 2.1,
        "TrackHistory": 5.2,
        "PointsFinishesLast5": 5,
        "HomeRace": 0,
        "CumulativeDriverPoints": 88,
        "CumulativeTeamPoints": 219,
    },
    {
        "Driver": "PIA",  # Oscar Piastri — McLaren — P7
        "GridPosition": 7, "QualiGapToPole": 0.530,
        "DriverFormLast5": 5.0,
        "TeamFormLast5": 5.2,
        "TrackHistory": 7.5,      # limited Monaco history
        "PointsFinishesLast5": 4,
        "HomeRace": 0,
        "CumulativeDriverPoints": 33,
        "CumulativeTeamPoints": 73,
    },
    {
        "Driver": "NOR",  # Lando Norris — McLaren — P8
        "GridPosition": 8, "QualiGapToPole": 0.610,
        "DriverFormLast5": 5.8,
        "TeamFormLast5": 5.2,
        "TrackHistory": 4.8,      # won 2024 Monaco
        "PointsFinishesLast5": 4,
        "HomeRace": 0,
        "CumulativeDriverPoints": 40,
        "CumulativeTeamPoints": 73,
    },
    {
        "Driver": "GAS",  # Pierre Gasly — Alpine — P9
        "GridPosition": 9, "QualiGapToPole": 0.695,
        "DriverFormLast5": 9.2,
        "TeamFormLast5": 10.4,
        "TrackHistory": 7.8,
        "PointsFinishesLast5": 2,
        "HomeRace": 0,
        "CumulativeDriverPoints": 11,
        "CumulativeTeamPoints": 16,
    },
    {
        "Driver": "LAW",  # Liam Lawson — Racing Bulls — P10
        "GridPosition": 10, "QualiGapToPole": 0.780,
        "DriverFormLast5": 8.8,
        "TeamFormLast5": 9.6,
        "TrackHistory": 10.0,     # limited history
        "PointsFinishesLast5": 3,
        "HomeRace": 0,
        "CumulativeDriverPoints": 20,
        "CumulativeTeamPoints": 28,
    },
    # ── P11-16 (Q2 eliminees) ─────────────────────────────────────────────
    {
        "Driver": "COL",  # Franco Colapinto — Williams — P11 (est.)
        "GridPosition": 11, "QualiGapToPole": 0.870,
        "DriverFormLast5": 11.4, "TeamFormLast5": 11.8,
        "TrackHistory": 10.0, "PointsFinishesLast5": 2,
        "HomeRace": 0, "CumulativeDriverPoints": 8, "CumulativeTeamPoints": 15,
    },
    {
        "Driver": "SAI",  # Carlos Sainz — Williams — P12 (est.)
        "GridPosition": 12, "QualiGapToPole": 0.940,
        "DriverFormLast5": 10.8, "TeamFormLast5": 11.8,
        "TrackHistory": 5.5, "PointsFinishesLast5": 2,
        "HomeRace": 0, "CumulativeDriverPoints": 7, "CumulativeTeamPoints": 15,
    },
    {
        "Driver": "HUL",  # Nico Hulkenberg — Audi — P13 (est.)
        "GridPosition": 13, "QualiGapToPole": 1.010,
        "DriverFormLast5": 12.0, "TeamFormLast5": 13.0,
        "TrackHistory": 9.2, "PointsFinishesLast5": 1,
        "HomeRace": 0, "CumulativeDriverPoints": 4, "CumulativeTeamPoints": 7,
    },
    {
        "Driver": "LIN",  # Arvid Lindblad — Racing Bulls — P14 (est.)
        "GridPosition": 14, "QualiGapToPole": 1.080,
        "DriverFormLast5": 12.5, "TeamFormLast5": 9.6,
        "TrackHistory": 10.0, "PointsFinishesLast5": 1,
        "HomeRace": 0, "CumulativeDriverPoints": 8, "CumulativeTeamPoints": 28,
    },
    {
        "Driver": "DOO",  # Jack Doohan — Alpine — P15 (est.)
        "GridPosition": 15, "QualiGapToPole": 1.150,
        "DriverFormLast5": 13.2, "TeamFormLast5": 10.4,
        "TrackHistory": 10.0, "PointsFinishesLast5": 1,
        "HomeRace": 0, "CumulativeDriverPoints": 5, "CumulativeTeamPoints": 16,
    },
    {
        "Driver": "BOR",  # Gabriel Bortoleto — Audi — P16 (est., started pit lane)
        "GridPosition": 16, "QualiGapToPole": 1.220,
        "DriverFormLast5": 14.0, "TeamFormLast5": 13.0,
        "TrackHistory": 10.0, "PointsFinishesLast5": 0,
        "HomeRace": 0, "CumulativeDriverPoints": 3, "CumulativeTeamPoints": 7,
    },
    # ── P17-22 (Q1 eliminees) ─────────────────────────────────────────────
    {
        "Driver": "OCO",  # Esteban Ocon — Haas — P17
        "GridPosition": 17, "QualiGapToPole": 1.350,
        "DriverFormLast5": 14.5, "TeamFormLast5": 14.8,
        "TrackHistory": 8.8, "PointsFinishesLast5": 1,
        "HomeRace": 0, "CumulativeDriverPoints": 3, "CumulativeTeamPoints": 5,
    },
    {
        "Driver": "PER",  # Sergio Perez — Cadillac — P18
        "GridPosition": 18, "QualiGapToPole": 1.420,
        "DriverFormLast5": 15.0, "TeamFormLast5": 15.5,
        "TrackHistory": 7.2, "PointsFinishesLast5": 1,
        "HomeRace": 0, "CumulativeDriverPoints": 2, "CumulativeTeamPoints": 2,
    },
    {
        "Driver": "BEA",  # Ollie Bearman — Haas — P19
        "GridPosition": 19, "QualiGapToPole": 1.490,
        "DriverFormLast5": 15.5, "TeamFormLast5": 14.8,
        "TrackHistory": 10.0, "PointsFinishesLast5": 0,
        "HomeRace": 0, "CumulativeDriverPoints": 2, "CumulativeTeamPoints": 5,
    },
    {
        "Driver": "BOT",  # Valtteri Bottas — Cadillac — P20
        "GridPosition": 20, "QualiGapToPole": 1.560,
        "DriverFormLast5": 16.0, "TeamFormLast5": 15.5,
        "TrackHistory": 9.5, "PointsFinishesLast5": 0,
        "HomeRace": 0, "CumulativeDriverPoints": 0, "CumulativeTeamPoints": 2,
    },
    {
        "Driver": "ALO",  # Fernando Alonso — Aston Martin — P21
        "GridPosition": 21, "QualiGapToPole": 1.640,
        "DriverFormLast5": 16.5, "TeamFormLast5": 17.0,
        "TrackHistory": 6.8, "PointsFinishesLast5": 0,
        "HomeRace": 0, "CumulativeDriverPoints": 0, "CumulativeTeamPoints": 0,
    },
    {
        "Driver": "STR",  # Lance Stroll — Aston Martin — P22
        "GridPosition": 22, "QualiGapToPole": 1.720,
        "DriverFormLast5": 17.2, "TeamFormLast5": 17.0,
        "TrackHistory": 11.5, "PointsFinishesLast5": 0,
        "HomeRace": 0, "CumulativeDriverPoints": 0, "CumulativeTeamPoints": 0,
    },
])

# ── Run predictions ────────────────────────────────────────────────────────
monaco_2026["PodiumProbability"] = rf.predict_proba(
    monaco_2026[feature_cols]
)[:, 1]

results = monaco_2026[["Driver", "GridPosition", "PodiumProbability"]].sort_values(
    "PodiumProbability", ascending=False
).reset_index(drop=True)

results["PodiumProbability"] = results["PodiumProbability"].map("{:.1%}".format)

print("=" * 50)
print("  2026 Monaco GP — Podium Predictions")
print("  Model: Random Forest | Input: Qualifying data")
print("=" * 50)
print(results.to_string(index=False))

print("\n--- MODEL'S PREDICTED PODIUM ---")
print(results.head(3)[["Driver", "GridPosition", "PodiumProbability"]].to_string(index=False))

print("\n--- ACTUAL RACE RESULT ---")
print("  P1: ANT (started P1)  ← model should get this")
print("  P2: HAM (started P3)  ← model should get this")
print("  P3: GAS (started P9)  ← model will likely miss this")
print("      (Gasly benefited from penalties handed to others)")
print("\nNote: Verstappen and Leclerc both DNF'd, which the")
print("model cannot predict — it only sees pre-race information.")
