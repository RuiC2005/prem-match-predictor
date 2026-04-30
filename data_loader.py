"""
data_loader.py — Data loading and pre-processing for the EPL Match Predictor.

Responsibilities:
  - Read all season CSVs from ./data/
  - Normalize dates and drop invalid rows
  - Compute per-match Elo ratings (cross-season, no leakage)

[WHY] Separated from feature engineering so we can swap data sources
(e.g. an API feed) without touching the feature pipeline.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR   = Path("data")
US_FILE    = Path("data/understat_raw.csv")

REQUIRED_COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]

# ---------------------------------------------------------------------------
# ELO constants
# [WHY] K=20 is standard for sports with ~15-40 matches per season.
#        Starting at 1500 is a universal convention (same as chess).
# ---------------------------------------------------------------------------
ELO_K = 20
ELO_START = 1500
# [WHY split K?] Home results are more "controllable" (crowd, rest, travel);
# away results carry more variance. Different learning rates capture this.
ELO_K_HOME = 24   # home performance updates faster
ELO_K_AWAY = 16   # away performance updates slower


# ===========================================================================
# 1. CSV loading
# ===========================================================================

def load_csvs(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """
    Reads all season CSVs from *data_dir*, tags each row with the source
    filename and returns a single concatenated DataFrame.

    [WHY] .copy() BEFORE .assign() fixes the PerformanceWarning that arises
    because these CSVs have 100+ columns (fragmented pandas block manager).
    Calling .copy() compacts all blocks into one contiguous array so
    the subsequent .assign() / insert() is fast and silent.
    """
    files = [f for f in sorted(data_dir.glob("*.csv")) if f.name != "understat_raw.csv"]
    if not files:
        raise FileNotFoundError(f"No match CSV files found in {data_dir}.")

    dfs = []
    for f in files:
        df = pd.read_csv(f, encoding="utf-8-sig").copy()
        df = df.assign(source_file=f.name)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True).copy()

    missing = [col for col in REQUIRED_COLUMNS if col not in combined.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return combined


# ===========================================================================
# 2. Date normalization
# ===========================================================================

def normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parses the Date column (DD/MM/YYYY format used by EPL CSVs),
    drops rows with unparseable dates or missing key columns,
    and sorts chronologically.

    [WHY] dayfirst=True → DD/MM/YYYY format.
          errors='coerce' converts bad dates to NaT so we can drop them
          safely instead of crashing mid-pipeline.
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTR"])
    return df.sort_values("Date").reset_index(drop=True)


# ===========================================================================
# 3. Elo ratings
# ===========================================================================

def compute_elo_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes per-match pre-match Elo ratings for home and away teams.

    [WHY] Pre-match Elo (before the result is known) avoids data leakage.
    Elo persists across seasons and captures cumulative team quality better
    than short-window rolling form alone.

    Expected score formula: E_A = 1 / (1 + 10^((R_B - R_A) / 400))
    This is the standard chess Elo formula — works well for football.

    Also maintains split home/away Elo tracks:
      - elo_home[team]: updated only when that team plays at home (K=24)
      - elo_away[team]: updated only when that team plays away  (K=16)
    [WHY split?] Home advantage is large and consistent in the EPL. Separate
    tracks let the model distinguish a team's home dominance from their away
    frailty — something the single Elo conflates.
    """
    elo: dict[str, float] = {}
    elo_home: dict[str, float] = {}  # home performance Elo
    elo_away: dict[str, float] = {}  # away performance Elo

    home_elo_pre, away_elo_pre = [], []
    home_elo_home_pre, away_elo_away_pre = [], []

    for _, row in df.iterrows():
        home, away = row["HomeTeam"], row["AwayTeam"]

        # --- Unified Elo (backward-compatible) ---
        r_h = elo.get(home, ELO_START)
        r_a = elo.get(away, ELO_START)
        home_elo_pre.append(r_h)
        away_elo_pre.append(r_a)

        # --- Venue-split Elo (new) ---
        r_h_home = elo_home.get(home, ELO_START)
        r_a_away = elo_away.get(away, ELO_START)
        home_elo_home_pre.append(r_h_home)
        away_elo_away_pre.append(r_a_away)

        e_h = 1 / (1 + 10 ** ((r_a - r_h) / 400))
        e_a = 1 - e_h

        if row["FTR"] == "H":
            s_h, s_a = 1.0, 0.0
        elif row["FTR"] == "A":
            s_h, s_a = 0.0, 1.0
        else:
            s_h, s_a = 0.5, 0.5

        # Update unified Elo
        elo[home] = r_h + ELO_K * (s_h - e_h)
        elo[away] = r_a + ELO_K * (s_a - e_a)

        # Update venue-split Elo with venue-specific expected scores
        e_h_home = 1 / (1 + 10 ** ((r_a_away - r_h_home) / 400))
        e_a_away = 1 - e_h_home
        elo_home[home] = r_h_home + ELO_K_HOME * (s_h - e_h_home)
        elo_away[away] = r_a_away + ELO_K_AWAY * (s_a - e_a_away)

    df = df.copy()
    df["home_elo"] = home_elo_pre
    df["away_elo"] = away_elo_pre
    df["elo_diff"] = df["home_elo"] - df["away_elo"]
    df["home_elo_home"] = home_elo_home_pre
    df["away_elo_away"] = away_elo_away_pre
    df["venue_elo_diff"] = df["home_elo_home"] - df["away_elo_away"]
    return df


# ===========================================================================
# 4. Understat xG merge (optional — skipped if CSV not present)
# ===========================================================================

def merge_understat(df: pd.DataFrame, us_file: Path = US_FILE) -> pd.DataFrame:
    """
    Left-join understat per-match xG data onto the main DataFrame.

    Adds columns: us_home_xg, us_away_xg
    Rows with no understat match get NaN — feature_engineering falls back
    to xg_proxy for those entries, so nothing breaks.

    [WHY left join?] We never want to lose matches just because understat
    doesn't cover them (e.g. the very first week of 2021-22 before data
    was fully published).
    """
    if not us_file.exists():
        print(
            "  [understat] understat_raw.csv not found — skipping xG merge.\n"
            "  [understat] Run: python fetch_understat.py"
        )
        return df

    us = pd.read_csv(us_file)
    us["date"] = pd.to_datetime(us["date"])
    us = us.rename(columns={"date": "Date", "home_team": "HomeTeam", "away_team": "AwayTeam"})

    cols_to_merge = [
        "Date", "HomeTeam", "AwayTeam", 
        "us_home_xg", "us_away_xg",
        "us_home_npxg", "us_away_npxg",
        "us_home_ppda", "us_away_ppda",
        "us_home_deep", "us_away_deep",
        "us_home_xpts", "us_away_xpts",
        "us_forecast_w", "us_forecast_d", "us_forecast_l"
    ]
    # Handle cases where the new columns might not exist in an older CSV yet
    available_cols = [c for c in cols_to_merge if c in us.columns]

    df = df.merge(
        us[available_cols],
        on=["Date", "HomeTeam", "AwayTeam"],
        how="left",
    )

    n_merged = df["us_home_xg"].notna().sum()
    print(f"  [understat] Merged xG for {n_merged}/{len(df)} matches.")
    return df


# ===========================================================================
# 5. Public entry point
# ===========================================================================

def load_and_prepare(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """
    Full data-prep pipeline: load → normalize → Elo → understat xG merge.
    Returns a sorted, Elo-enriched DataFrame ready for feature engineering.
    """
    raw = load_csvs(data_dir)
    df = normalize_dates(raw)
    df = compute_elo_ratings(df)
    df = merge_understat(df)
    return df
