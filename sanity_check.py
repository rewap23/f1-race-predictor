"""
sanity_check.py

Quick checks on data/raw_race_results.csv before moving on to feature
engineering. Run after build_race_dataset.py has finished.

    python sanity_check.py
"""

import pandas as pd

df = pd.read_csv("data/raw_race_results.csv")

print("Shape:", df.shape)
print("\nRows per season:")
print(df["Season"].value_counts().sort_index())

print("\nStatus breakdown (Finished vs various DNF reasons):")
print(df["Status"].value_counts())

print("\nSample of non-finishers (no numeric FinishPosition):")
print(df[df["FinishPosition"].isna()][["Driver", "Status", "ClassifiedPosition"]].head(10))
