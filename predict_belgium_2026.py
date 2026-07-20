"""
predict_2026_belgium.py

Tests the trained random forest model on the 2026 Belgian Grand Prix.
Uses real qualifying data from July 18 2026 (Spa-Francorchamps) as
model input to predict who is most likely to reach the podium.

NOTE: Unlike the Monaco script, this is run BEFORE the race (quali was
July 18, race is July 19 2026), so there's no "actual result" to compare
against yet — this is a genuine pre-race prediction.

Run with:
    python predict_2026_belgium.py
"""

import joblib
import pandas as pd

# ── Load saved model ───────────────────────────────────────────────────────
rf = joblib.load("model/random_forest.pkl")
feature_cols = joblib.load("model/feature_cols.pkl")

# ── 2026 Belgian GP — Round 10 (Spa-Francorchamps) ─────────────────────────
#
# HOW EACH COLUMN WAS FILLED:
#
# GridPosition      — FINAL grid order, i.e. qualifying order adjusted for
#                     confirmed grid penalties. Norris qualified P3 but
#                     drops to P13 (power unit component allocation), so
#                     everyone originally between P4-P13 moves up one slot.
#                     Hadjar, Alonso and Stroll are also confirmed to be
#                     carrying grid penalties into the race, but the exact
#                     drop wasn't specified in available reports, so their
#                     qualifying order is kept as a best estimate.
# QualiGapToPole    — seconds behind Antonelli's 1:44.361 pole lap.
#                     P1-P16 confirmed from session reports; Hadjar set no
#                     representative Q3 time (car issue) so his gap is
#                     estimated from his Q2 pace; P17-P22 (Q1 eliminees)
#                     use their reported gaps to the overall pole time.
# DriverFormLast5   — average finishing position over the last 5 rounds
#                     (Canada, Monaco, Spain, Austria, Britain). DNF/DNS
#                     results are treated as 20th for averaging purposes —
#                     an approximation, not an official stat.
# TeamFormLast5     — average finishing position for both team cars
#                     combined over the same 5 rounds.
# TrackHistory      — estimated historical average finish at Spa-
#                     Francorchamps, from known career patterns. Rookies
#                     and drivers with 0-1 prior Spa starts get a neutral
#                     ~10.0 fill (matching the convention from the Monaco
#                     script).
# PointsFinishesLast5 — how many of the last 5 races were finished in the
#                     points (top 10).
# HomeRace          — no driver on the current 2026 grid is Belgian, so
#                     this is 0 for every entry this weekend.
# CumulativeDriverPoints — championship points entering Belgium (after
#                     Round 9, the British GP).
# CumulativeTeamPoints   — constructor points entering Belgium (after
#                     Round 9, the British GP).
#
# Sources: RacingNews365, The Race, Motorsport Week, Formula1.com,
# Crash.net, Total-Motorsport, GPFans, F1-Fansite (July 2026)

belgium_2026 = pd.DataFrame([
    # ── Front of grid ───────────────────────────────────────────────────────
    {
        "Driver": "ANT",  # Kimi Antonelli — Mercedes — POLE
        "GridPosition": 1, "QualiGapToPole": 0.000,
        "DriverFormLast5": 7.2,
        "TeamFormLast5": 7.3,
        "TrackHistory": 10.0,     # limited Spa history (2nd start)
        "PointsFinishesLast5": 3,
        "HomeRace": 0,
        "CumulativeDriverPoints": 179,
        "CumulativeTeamPoints": 333,
    },
    {
        "Driver": "VER",  # Max Verstappen — Red Bull — P2
        "GridPosition": 2, "QualiGapToPole": 0.317,
        "DriverFormLast5": 9.8,
        "TeamFormLast5": 7.5,
        "TrackHistory": 2.5,      # multiple Spa wins historically
        "PointsFinishesLast5": 3,
        "HomeRace": 0,
        "CumulativeDriverPoints": 76,
        "CumulativeTeamPoints": 128,
    },
    {
        "Driver": "RUS",  # George Russell — Mercedes — P3 (was P4 on track)
        "GridPosition": 3, "QualiGapToPole": 0.508,
        "DriverFormLast5": 7.4,
        "TeamFormLast5": 7.3,
        "TrackHistory": 5.0,      # multiple Spa podiums
        "PointsFinishesLast5": 3,
        "HomeRace": 0,
        "CumulativeDriverPoints": 154,
        "CumulativeTeamPoints": 333,
    },
    {
        "Driver": "LEC",  # Charles Leclerc — Ferrari — P4 (was P5 on track)
        "GridPosition": 4, "QualiGapToPole": 0.532,
        "DriverFormLast5": 9.6,
        "TeamFormLast5": 6.1,
        "TrackHistory": 3.5,      # 2019 Spa winner
        "PointsFinishesLast5": 3,
        "HomeRace": 0,
        "CumulativeDriverPoints": 108,
        "CumulativeTeamPoints": 255,
    },
    {
        "Driver": "HAM",  # Lewis Hamilton — Ferrari — P5 (was P6 on track)
        "GridPosition": 5, "QualiGapToPole": 0.534,
        "DriverFormLast5": 2.6,   # excellent recent form
        "TeamFormLast5": 6.1,
        "TrackHistory": 2.8,      # multiple Spa wins historically
        "PointsFinishesLast5": 5,
        "HomeRace": 0,
        "CumulativeDriverPoints": 147,
        "CumulativeTeamPoints": 255,
    },
    {
        "Driver": "PIA",  # Oscar Piastri — McLaren — P6 (was P7 on track)
        "GridPosition": 6, "QualiGapToPole": 0.655,
        "DriverFormLast5": 7.2,
        "TeamFormLast5": 9.0,
        "TrackHistory": 7.0,
        "PointsFinishesLast5": 3,
        "HomeRace": 0,
        "CumulativeDriverPoints": 82,
        "CumulativeTeamPoints": 179,
    },
    {
        "Driver": "LIN",  # Arvid Lindblad — Racing Bulls — P7 (was P8)
        "GridPosition": 7, "QualiGapToPole": 0.782,
        "DriverFormLast5": 10.8,
        "TeamFormLast5": 9.1,
        "TrackHistory": 10.0,     # rookie, no Spa history
        "PointsFinishesLast5": 4,
        "HomeRace": 0,
        "CumulativeDriverPoints": 20,
        "CumulativeTeamPoints": 59,
    },
    {
        "Driver": "BOR",  # Gabriel Bortoleto — Audi — P8 (was P9)
        "GridPosition": 8, "QualiGapToPole": 1.267,
        "DriverFormLast5": 10.8,
        "TeamFormLast5": 13.1,
        "TrackHistory": 10.0,     # rookie, limited history
        "PointsFinishesLast5": 1,
        "HomeRace": 0,
        "CumulativeDriverPoints": 6,
        "CumulativeTeamPoints": 6,
    },
    {
        "Driver": "HAD",  # Isack Hadjar — Red Bull — P9 (was P10, no Q3 time + penalty)
        "GridPosition": 9, "QualiGapToPole": 1.350,
        "DriverFormLast5": 5.2,
        "TeamFormLast5": 7.5,
        "TrackHistory": 9.0,      # limited, rookie-era Spa starts
        "PointsFinishesLast5": 5,
        "HomeRace": 0,
        "CumulativeDriverPoints": 52,
        "CumulativeTeamPoints": 128,
    },
    {
        "Driver": "LAW",  # Liam Lawson — Racing Bulls — P10 (was P11)
        "GridPosition": 10, "QualiGapToPole": 0.978,
        "DriverFormLast5": 7.4,
        "TeamFormLast5": 9.1,
        "TrackHistory": 9.5,
        "PointsFinishesLast5": 5,
        "HomeRace": 0,
        "CumulativeDriverPoints": 39,
        "CumulativeTeamPoints": 59,
    },
    # ── Midfield ────────────────────────────────────────────────────────────
    {
        "Driver": "GAS",  # Pierre Gasly — Alpine — P11 (was P12)
        "GridPosition": 11, "QualiGapToPole": 1.189,
        "DriverFormLast5": 8.2,
        "TeamFormLast5": 9.5,
        "TrackHistory": 7.5,
        "PointsFinishesLast5": 4,
        "HomeRace": 0,
        "CumulativeDriverPoints": 42,
        "CumulativeTeamPoints": 60,
    },
    {
        "Driver": "COL",  # Franco Colapinto — Alpine — P12 (was P13)
        "GridPosition": 12, "QualiGapToPole": 1.250,
        "DriverFormLast5": 10.8,
        "TeamFormLast5": 9.5,
        "TrackHistory": 10.0,
        "PointsFinishesLast5": 3,
        "HomeRace": 0,
        "CumulativeDriverPoints": 18,
        "CumulativeTeamPoints": 60,
    },
    {
        "Driver": "NOR",  # Lando Norris — McLaren — P13 (10-place grid penalty)
        "GridPosition": 13, "QualiGapToPole": 0.440,
        "DriverFormLast5": 10.8,
        "TeamFormLast5": 9.0,
        "TrackHistory": 4.5,      # multiple Spa podiums
        "PointsFinishesLast5": 3,
        "HomeRace": 0,
        "CumulativeDriverPoints": 97,
        "CumulativeTeamPoints": 179,
    },
    {
        "Driver": "HUL",  # Nico Hulkenberg — Audi — P14
        "GridPosition": 14, "QualiGapToPole": 1.529,
        "DriverFormLast5": 15.4,
        "TeamFormLast5": 13.1,
        "TrackHistory": 8.0,
        "PointsFinishesLast5": 0,
        "HomeRace": 0,
        "CumulativeDriverPoints": 0,
        "CumulativeTeamPoints": 6,
    },
    {
        "Driver": "SAI",  # Carlos Sainz — Williams — P15
        "GridPosition": 15, "QualiGapToPole": 1.635,
        "DriverFormLast5": 14.8,
        "TeamFormLast5": 15.9,
        "TrackHistory": 5.5,      # career Spa podiums
        "PointsFinishesLast5": 1,
        "HomeRace": 0,
        "CumulativeDriverPoints": 6,
        "CumulativeTeamPoints": 11,
    },
    {
        "Driver": "BEA",  # Ollie Bearman — Haas — P16
        "GridPosition": 16, "QualiGapToPole": 1.637,
        "DriverFormLast5": 14.6,
        "TeamFormLast5": 13.8,
        "TrackHistory": 10.0,
        "PointsFinishesLast5": 1,
        "HomeRace": 0,
        "CumulativeDriverPoints": 18,
        "CumulativeTeamPoints": 21,
    },
    # ── Back of grid (Q1 eliminees) ────────────────────────────────────────
    {
        "Driver": "ALB",  # Alex Albon — Williams — P17
        "GridPosition": 17, "QualiGapToPole": 1.255,
        "DriverFormLast5": 17.0,
        "TeamFormLast5": 15.9,
        "TrackHistory": 8.5,
        "PointsFinishesLast5": 1,
        "HomeRace": 0,
        "CumulativeDriverPoints": 5,
        "CumulativeTeamPoints": 11,
    },
    {
        "Driver": "OCO",  # Esteban Ocon — Haas — P18
        "GridPosition": 18, "QualiGapToPole": 1.936,
        "DriverFormLast5": 13.0,
        "TeamFormLast5": 13.8,
        "TrackHistory": 8.0,
        "PointsFinishesLast5": 1,
        "HomeRace": 0,
        "CumulativeDriverPoints": 3,
        "CumulativeTeamPoints": 21,
    },
    {
        "Driver": "BOT",  # Valtteri Bottas — Cadillac — P19
        "GridPosition": 19, "QualiGapToPole": 1.958,
        "DriverFormLast5": 18.4,
        "TeamFormLast5": 17.5,
        "TrackHistory": 5.5,      # past Spa podiums with Mercedes
        "PointsFinishesLast5": 0,
        "HomeRace": 0,
        "CumulativeDriverPoints": 0,
        "CumulativeTeamPoints": 0,
    },
    {
        "Driver": "PER",  # Sergio Perez — Cadillac — P20
        "GridPosition": 20, "QualiGapToPole": 2.106,
        "DriverFormLast5": 16.6,
        "TeamFormLast5": 17.5,
        "TrackHistory": 8.5,
        "PointsFinishesLast5": 0,
        "HomeRace": 0,
        "CumulativeDriverPoints": 0,
        "CumulativeTeamPoints": 0,
    },
    {
        "Driver": "ALO",  # Fernando Alonso — Aston Martin — P21 (also carries a grid penalty)
        "GridPosition": 21, "QualiGapToPole": 4.137,
        "DriverFormLast5": 17.2,
        "TeamFormLast5": 18.0,
        "TrackHistory": 4.0,      # historical Spa wins/podiums
        "PointsFinishesLast5": 0,
        "HomeRace": 0,
        "CumulativeDriverPoints": 1,
        "CumulativeTeamPoints": 1,
    },
    {
        "Driver": "STR",  # Lance Stroll — Aston Martin — P22 (also carries a grid penalty)
        "GridPosition": 22, "QualiGapToPole": 4.312,
        "DriverFormLast5": 18.8,
        "TeamFormLast5": 18.0,
        "TrackHistory": 9.5,
        "PointsFinishesLast5": 0,
        "HomeRace": 0,
        "CumulativeDriverPoints": 0,
        "CumulativeTeamPoints": 1,
    },
])

# ── Run predictions ────────────────────────────────────────────────────────
belgium_2026["PodiumProbability"] = rf.predict_proba(
    belgium_2026[feature_cols]
)[:, 1]

results = belgium_2026[["Driver", "GridPosition", "PodiumProbability"]].sort_values(
    "PodiumProbability", ascending=False
).reset_index(drop=True)

results["PodiumProbability"] = results["PodiumProbability"].map("{:.1%}".format)

print("=" * 55)
print("  2026 Belgian GP (Spa-Francorchamps) — Podium Predictions")
print("  Model: Random Forest | Input: Qualifying data, July 18 2026")
print("=" * 55)
print(results.to_string(index=False))

print("\n--- MODEL'S PREDICTED PODIUM ---")
print(results.head(3)[["Driver", "GridPosition", "PodiumProbability"]].to_string(index=False))

print("\nNote: The race hasn't been run yet (Sunday, July 19 2026), so")
print("this is a genuine pre-race prediction, not a backtest. Spa is a")
print("high-variance track (long straights, changeable weather, high")
print("chance of Safety Cars), so treat probabilities as a guide to")
print("relative strength rather than a hard forecast — the model also")
print("has no way to account for in-race incidents, strategy calls, or")
print("weather changes.")