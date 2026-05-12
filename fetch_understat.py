"""
fetch_understat.py — Download per-match xG data from understat.com (v2).

Changes from v1
---------------
SEASONS extended: now covers 2018–2025 (all 8 seasons in our CSV dataset).
  [WHY 2018?] Our earliest football-data CSV is EPL_2018_2019.csv. Understat
  has EPL data from 2014/15, but we only need what aligns with our match CSV
  range. Fetching 2018+ gives us xG coverage for every season we train on,
  meaning the model never has to fall back to xg_proxy for the training set.

NEW FIELDS extracted from teamsData.history (all available per-match stats):
  us_home_shots / us_away_shots
    — Total shots taken. Understat stores this in the history object as
      "shots". Different from football-data's HS/AS (which counts all
      attempts); understat's shot count feeds their xG model so it aligns
      with xG values better than football-data's shot count.
      [WHY useful?] Shot volume independent of location — pairs with xG
      to compute a per-match shot quality ratio (xG/shot).

  us_home_xga / us_away_xga
    — Expected goals conceded, taken directly from teamsData.history.
      The opponent's xG in that match — more reliable than inferring it
      from the other team's xG because understat stores it natively.
      [WHY not just swap home/away xG?] This is exactly what we'd do, but
      storing it explicitly keeps data_loader / feature_engineering clean
      — no cross-referencing required.

  us_home_npxg / us_away_npxg  [already present, now from correct field]
    — Non-penalty xG. Removes the variance of penalty awards which are
      partially luck-driven. npxG is a better long-run finisher-quality
      signal than raw xG. Already in v1 but sometimes mapped to None due
      to a key lookup bug — fixed here.

  us_home_ppda / us_away_ppda  [already present, calculation corrected]
    — Passes allowed per defensive action. Lower = more aggressive press.
      v1 had a potential division-by-zero when ppda.def == 0 on a very
      high-press game. Now guarded with max(1, ppda_def).

  us_home_deep / us_away_deep  [already present]
    — Dangerous-area passes (passes into the zone ~18 yards from goal).
      Independent of xG methodology. Correlates with late-game dominance.

  us_home_xpts / us_away_xpts  [already present]
    — Expected points based on xG. A cleaner long-run form signal than
      actual points because it strips out goalkeeper heroics and post-hits.

  us_forecast_w/d/l  [already present]
    — Understat's pre-match win/draw/loss forecast probabilities.
      Derived from their internal xG model, so orthogonal to bookmaker odds.

FIELDS deliberately NOT added (and why):
  xGChain / xGBuildup — player-level stats only, not available per-match
    in teamsData. Would require a separate per-match endpoint call (380
    calls/season × 8 seasons = 3040 extra HTTP requests). Not worth it.
  scored / missed (goals) — already in our football-data CSVs as FTHG/FTAG.
    Duplicating them from understat adds merge risk without new information.
  wins / draws / losses (season totals in league_teams) — season-level,
    not per-match. We compute these ourselves in feature_engineering.

Saves: data/understat_raw.csv

Run once, then re-run to refresh:
    python fetch_understat.py

Requires:
    pip install playwright
    python -m playwright install chromium
"""

import sys
import time
from pathlib import Path

import pandas as pd

try:
    from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
except ImportError:
    sys.exit(
        "Missing dependency. Run:\n"
        "    pip install playwright\n"
        "    python -m playwright install chromium"
    )

LEAGUE   = "EPL"
# [WHY 2018–2025?] Aligns exactly with our football-data CSV range:
#   EPL_2018_2019.csv → season=2018  (Aug 2018 – May 2019)
#   EPL_2019_2020.csv → season=2019  ... etc.
#   EPL_2024_2025.csv → season=2024
#   EPL_2025_2026.csv → season=2025  (current, partial)
# Understat season=2018 means the 2018/19 academic year.
SEASONS  = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
OUT_FILE = Path("data/understat_raw.csv")

BASE_URL = "https://understat.com/league/{league}/{season}"

# ---------------------------------------------------------------------------
# Team name normalisation: understat spelling → football-data.co.uk spelling
# Add entries here whenever a merge fails (check processed_matches.csv for
# unmatched rows — they'll have NaN in all us_* columns).
# ---------------------------------------------------------------------------
TEAM_MAP: dict[str, str] = {
    "Manchester City":         "Man City",
    "Manchester United":       "Man United",
    "Newcastle United":        "Newcastle",
    "Wolverhampton Wanderers": "Wolves",
    "Nottingham Forest":       "Nott'm Forest",
    "Leeds United":            "Leeds",
    "Leicester City":          "Leicester",
    "Luton Town":              "Luton",
    "Sheffield United":        "Sheffield United",
    "West Bromwich Albion":    "West Brom",
    "Queens Park Rangers":     "QPR",
    "Swansea City":            "Swansea",
    "Stoke City":              "Stoke",
    "Huddersfield":            "Huddersfield",
    "Cardiff":                 "Cardiff",
}


def normalize(name: str) -> str:
    return TEAM_MAP.get(name, name)


def fetch_season(page, season: int, retries: int = 3) -> list[dict]:
    """
    Fetch all completed matches for one EPL season from understat.

    Extracts from two JS globals on the page:
      datesData  — one entry per match: xG, goals, forecast
      teamsData  — one entry per team per match: shots, npxG, ppda, deep, xpts, xGA

    [WHY two sources?] datesData gives the match-level xG and forecast.
    teamsData gives the per-team per-match performance history which
    includes shots, npxG, ppda, deep passes, xpts and xGA — these are
    NOT in datesData and must be pulled from teamsData.history.
    """
    url = BASE_URL.format(league=LEAGUE, season=season)

    for attempt in range(1, retries + 1):
        try:
            # "networkidle" waits until JS has finished executing and populated
            # the global variables. "domcontentloaded" fires too early.
            page.goto(url, wait_until="networkidle", timeout=60_000)
            # Belt-and-suspenders: explicitly wait for the variable to exist.
            page.wait_for_function(
                "() => typeof datesData !== 'undefined'", timeout=15_000
            )
        except PlaywrightTimeout:
            print(
                f"\n  WARNING: Timeout on season {season} "
                f"(attempt {attempt}/{retries})",
                end=" ",
            )
            if attempt < retries:
                time.sleep(3)
                continue
            return []
        except Exception as exc:
            print(f"\n  WARNING: Failed to load season {season} — {exc}")
            return []

        try:
            matches    = page.evaluate("() => datesData")
            teams_data = page.evaluate("() => teamsData")
        except Exception as exc:
            print(
                f"\n  WARNING: Could not evaluate JS globals for "
                f"season {season} — {exc}"
            )
            return []

        if not matches or not teams_data:
            print(
                f"\n  WARNING: datesData or teamsData empty/null "
                f"for season {season}"
            )
            return []

        break   # success — exit retry loop

    # ------------------------------------------------------------------
    # Build a lookup: (date_str, team_name) → per-match team stats dict
    # Source: teamsData[team_id].history — one entry per played match.
    # ------------------------------------------------------------------
    team_stats: dict[tuple[str, str], dict] = {}

    for team_id, t_info in teams_data.items():
        t_title = normalize(t_info["title"])

        for h in t_info["history"]:
            match_date = str(pd.to_datetime(h["date"]).date())

            # PPDA: passes allowed per defensive action.
            # ppda is a dict {"att": N, "def": M}; lower ratio = harder press.
            # [WHY max(1, def)?] Prevents division-by-zero on extremely
            # aggressive pressing games where def==0 (rare but observed).
            ppda_obj = h.get("ppda", {})
            ppda_att = float(ppda_obj.get("att", 0) or 0)
            ppda_def = float(ppda_obj.get("def", 1) or 1)
            ppda     = ppda_att / max(1.0, ppda_def)

            # xGA: expected goals conceded (xG allowed to opponent).
            # Stored natively in teamsData as "xGA" — more reliable than
            # inferring it from the match-level xG["a"] or xG["h"].
            xga = h.get("xGA")

            team_stats[(match_date, t_title)] = {
                # Already in v1 (corrected):
                "ppda":  ppda,
                "xpts":  float(h.get("xpts", 0.0) or 0.0),
                "npxg":  float(h.get("npxG", 0.0) or 0.0),
                "deep":  float(h.get("deep", 0.0) or 0.0),
                # New in v2:
                # shots: total shots taken (aligns with understat's xG model).
                "shots": float(h.get("shots", 0.0) or 0.0),
                # xga: expected goals conceded per match.
                "xga":   float(xga) if xga is not None else None,
            }

    # ------------------------------------------------------------------
    # Build per-match rows from datesData.
    # datesData is a flat list (Playwright unwraps the date-keyed dict).
    # ------------------------------------------------------------------
    rows = []
    for m in matches:
        if not m.get("isResult"):
            continue   # skip unplayed fixtures

        try:
            date_str  = str(pd.to_datetime(m["datetime"]).date())
            home_team = normalize(m["h"]["title"])
            away_team = normalize(m["a"]["title"])

            h_stats = team_stats.get((date_str, home_team), {})
            a_stats = team_stats.get((date_str, away_team), {})

            forecast = m.get("forecast") or {}

            rows.append({
                # ── Match identity ─────────────────────────────────────
                "date":      date_str,
                "home_team": home_team,
                "away_team": away_team,

                # ── Match-level xG (from datesData) ────────────────────
                "us_home_xg": float(m["xG"]["h"]),
                "us_away_xg": float(m["xG"]["a"]),

                # ── Pre-match forecast probabilities ────────────────────
                # Derived from understat's own xG model — orthogonal to
                # bookmaker odds (which incorporate bet flow, not just xG).
                "us_forecast_w": float(forecast.get("w", 0.0)) if forecast else None,
                "us_forecast_d": float(forecast.get("d", 0.0)) if forecast else None,
                "us_forecast_l": float(forecast.get("l", 0.0)) if forecast else None,

                # ── Per-team stats (from teamsData) ────────────────────
                # npxG: non-penalty xG — strips penalty luck from xG.
                "us_home_npxg": h_stats.get("npxg"),
                "us_away_npxg": a_stats.get("npxg"),

                # ppda: passes per defensive action — press intensity proxy.
                "us_home_ppda": h_stats.get("ppda"),
                "us_away_ppda": a_stats.get("ppda"),

                # deep: dangerous-area ball entries.
                "us_home_deep": h_stats.get("deep"),
                "us_away_deep": a_stats.get("deep"),

                # xpts: expected points based on xG scoreline.
                "us_home_xpts": h_stats.get("xpts"),
                "us_away_xpts": a_stats.get("xpts"),

                # shots: total shots taken (understat definition, aligns
                # with their xG model; useful for xG/shot quality ratio).
                "us_home_shots": h_stats.get("shots"),
                "us_away_shots": a_stats.get("shots"),

                # xga: expected goals conceded (native from teamsData).
                # Note: us_home_xga ≈ us_away_xg for the same match, but
                # stored explicitly to avoid cross-referencing in feature_engineering.
                "us_home_xga": h_stats.get("xga"),
                "us_away_xga": a_stats.get("xga"),
            })
        except (KeyError, TypeError, ValueError):
            continue   # skip malformed entries

    return rows


def main() -> None:
    print("Fetching understat EPL data (2018–2025) using Playwright...")
    all_rows: list[dict] = []

    with sync_playwright() as p:
        # NOTE: On some Windows machines headless=True can trigger bot-detection.
        # If you get empty results, try headless=False to watch the browser open.
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        )
        page = context.new_page()

        for season in SEASONS:
            label = f"{season}-{str(season + 1)[-2:]}"
            print(f"  Season {label}...", end=" ", flush=True)

            rows = fetch_season(page, season)
            print(f"{len(rows)} matches")
            all_rows.extend(rows)

            # Polite delay between seasons to avoid rate-limiting.
            # [WHY 2s?] Understat uses Cloudflare; rapid sequential page loads
            # from the same IP can trigger a challenge page. 2s is safe.
            time.sleep(2.0)

        browser.close()

    if not all_rows:
        print("\nNo data fetched — check internet connection or SEASONS list.")
        return

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    OUT_FILE.parent.mkdir(exist_ok=True)
    df.to_csv(OUT_FILE, index=False)

    n_seasons   = df["date"].dt.year.nunique()
    xg_coverage = df["us_home_xg"].notna().sum()
    ppda_coverage = df["us_home_ppda"].notna().sum()
    shots_coverage = df["us_home_shots"].notna().sum()

    print(f"\nSaved {len(df)} matches across ~{n_seasons} calendar years → {OUT_FILE}")
    print(f"  xG coverage:   {xg_coverage}/{len(df)} matches")
    print(f"  PPDA coverage: {ppda_coverage}/{len(df)} matches")
    print(f"  Shots coverage:{shots_coverage}/{len(df)} matches")
    print()
    print(
        df[[
            "date", "home_team", "away_team",
            "us_home_xg", "us_away_xg",
            "us_home_npxg", "us_home_ppda",
            "us_home_shots", "us_home_xga",
            "us_home_xpts",
        ]].head(8).to_string(index=False)
    )


if __name__ == "__main__":
    main()