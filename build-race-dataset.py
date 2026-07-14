
import os # telling python to interacte with your operating system and directory, so it can read files in the system
import time # 

import pandas as pd
import fastf1
import fastf1.exceptions #

# Add a retry helper function at the top of the file, right after your imports
def load_session_with_retry(year, round_num, session_type, max_retries=3):
    """Load a session, retrying with a wait if rate limited."""
    for attempt in range(max_retries):
        try:
            session = fastf1.get_session(year, round_num, session_type)
            session.load(laps=False, telemetry=False, weather=False, messages=False)
            return session
        except fastf1.exceptions.RateLimitExceededError:
            wait = 120 * (attempt + 1)  # 2 min, then 4 min, then 6 min
            print(f"    [rate limit] waiting {wait}s before retry {attempt + 1}/{max_retries}...")
            time.sleep(wait)
        except Exception as e:
            print(f"    [error] {e}")
            return None
    print(f"    [skip] gave up after {max_retries} retries")
    return None

# important to cache the data, since we are dealing with a large dataset, and it will slow 

CACHE_DIR = 'fastf1 cache' # this is the directory where the data will be stored
os.makedirs(CACHE_DIR, exist_ok=True) # this creates the directory if it doesn't exist
fastf1.Cache.enable_cache(CACHE_DIR) 
# this enables the cache, so that the data is stored in the fastf1 cache directory we made earlier

SEASONS = range(2018, 2026) # this is the range of seasons we want to get data for, from 2018 to 2025

rows = [] # creating empty list to store the data we get from the fastf1 library

#loop over each year/season
for year in SEASONS:
    print(f"\n ====> {year} season <====")
    # so for each year in seasons we need to get the event schedule, which is get_event_schedule, and we need to pass the year as an argument
    schedule = fastf1.get_event_schedule(year)

    # loop over each event in the schedule
    # For each event in that schedule, call fastf1.get_session(year, round_number, "R") to get the race session data, and then call session.load() to load the data. Then we can get the laps data by calling session.laps, and then we can loop over each lap in the laps data, and for each lap we can get the driver, lap time, and lap number. We can then append that data to our rows list as a dictionary.
    for index, event in schedule.iterrows(): 
        round_num = int(event["RoundNumber"]) 
        event_name = event["EventName"]

        try: 
            session = fastf1.get_session(year, round_num, "R") 
            session.load(laps=False, telemetry=False, weather=False, messages=False)
            #note: we are not loading laps, telemetry, weather, or messages data to save time and memory, since we only need the results data for this dataset
        except Exception as e:
            print(f"  [skip] round {round_num} ({event_name}): {e}")
            continue
 
        results = session.results
        if results is None or results.empty:
            print(f"  [skip] round {round_num} ({event_name}): no results")
            continue

        for index, row in results.iterrows():
            rows.append({
               "Season": year,
                "Round": round_num,
                "EventName": event_name,
                "Country": event["Country"],
                "DriverId": row["DriverId"],
                "Driver": row["Abbreviation"],
                "FullName": row["FullName"],
                "Team": row["TeamName"],
                "GridPosition": row["GridPosition"],
                "FinishPosition": row["Position"],
                "ClassifiedPosition": row["ClassifiedPosition"],
                "Status": row["Status"],
                "Points": row["Points"],
            })
        print(f"  [ok]   round {round_num}: {event_name} - {len(results)} drivers")
        time.sleep(5)  # be polite to the API
 
df = pd.DataFrame(rows)
 
os.makedirs("data", exist_ok=True)
out_path = "data/raw_race_results.csv"
df.to_csv(out_path, index=False)
 
print(f"\nSaved {len(df)} rows across {df['Season'].nunique()} seasons to {out_path}")

#building a flat list of dictionaries (rows = [])
# appending one dict per driver per race inside the inner loop
# then creating a pandas dataframe from that list of dicts, and saving it to a csv file

# ── Qualifying data pull ──────────────────────────────────────────────────
print("\n=== Pulling qualifying data ===")
quali_rows = []

for year in SEASONS:
    print(f"\n  {year} season")
    schedule = fastf1.get_event_schedule(year, include_testing=False)

    for _, event in schedule.iterrows():
        round_num = int(event["RoundNumber"])
        event_name = event["EventName"]

        try:
            session = fastf1.get_session(year, round_num, "Q")
            session.load(laps=False, telemetry=False, weather=False, messages=False)
        except Exception as e:
            print(f"    [skip] round {round_num} ({event_name}): {e}")
            continue

        results = session.results
        if results is None or results.empty:
            continue

        # Pole time is the fastest Q time among all drivers
        pole_time = results["Q3"].dropna().min()
        if pd.isna(pole_time):
            pole_time = results["Q2"].dropna().min()
        if pd.isna(pole_time):
            pole_time = results["Q1"].dropna().min()

        for _, r in results.iterrows():
            # Use the best qualifying time available for each driver
            driver_time = r["Q3"]
            if pd.isna(driver_time):
                driver_time = r["Q2"]
            if pd.isna(driver_time):
                driver_time = r["Q1"]

            # Gap in seconds — NaN if driver has no qualifying time
            if pd.notna(driver_time) and pd.notna(pole_time):
                gap = (driver_time - pole_time).total_seconds()
            else:
                gap = None

            quali_rows.append({
                "Season": year,
                "Round": round_num,
                "DriverId": r["DriverId"],
                "QualiGapToPole": gap,
            })

        print(f"    [ok] round {round_num}: {event_name}")
        time.sleep(5)

quali_df = pd.DataFrame(quali_rows)
quali_path = "data/qualifying_data.csv"
quali_df.to_csv(quali_path, index=False)
print(f"\nSaved {len(quali_df)} qualifying rows to {quali_path}")

