"""
fetch_understat.py — Download per-match xG data from understat.com.

Uses Playwright (a headless browser) to evaluate the page JavaScript and 
extract the `datesData` global variable. This implicitly handles any Cloudflare 
or JS-rendering protections that block standard python requests.

Saves: data/understat_raw.csv

Run once, then re-run whenever you want fresh data:
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
SEASONS  = [2021, 2022, 2023, 2024, 2025]
OUT_FILE = Path("data/understat_raw.csv")

BASE_URL = "https://understat.com/league/{league}/{season}"

# ---------------------------------------------------------------------------
# Team name mapping: understat spelling → football-data.co.uk spelling
# ---------------------------------------------------------------------------
TEAM_MAP: dict[str, str] = {
    "Manchester City":          "Man City",
    "Manchester United":        "Man United",
    "Newcastle United":         "Newcastle",
    "Wolverhampton Wanderers":  "Wolves",
    "Nottingham Forest":        "Nott'm Forest",
    "Leeds United":             "Leeds",
    "Leicester City":           "Leicester",
    "Luton Town":               "Luton",
    "Sheffield United":         "Sheffield United",
}


def normalize(name: str) -> str:
    return TEAM_MAP.get(name, name)


def fetch_season(page, season: int, retries: int = 3) -> list[dict]:
    """Fetch all completed matches for one EPL season from understat."""
    url = BASE_URL.format(league=LEAGUE, season=season)

    for attempt in range(1, retries + 1):
        try:
            # FIX 1: Use "networkidle" instead of "domcontentloaded".
            # Understat is a JS-heavy site — datesData is injected by a script
            # tag AFTER the DOM loads. "networkidle" waits until all network
            # activity has settled, giving the JS time to execute and populate
            # the global variable. "domcontentloaded" fires too early.
            page.goto(url, wait_until="networkidle", timeout=60000)

            # FIX 2: Explicitly wait for the JS variable to exist before reading it.
            # This is a belt-and-suspenders guard: even after "networkidle",
            # the variable may not be defined yet on slower connections.
            page.wait_for_function("() => typeof datesData !== 'undefined'", timeout=15000)

        except PlaywrightTimeout:
            print(f"\n  WARNING: Timeout on season {season} (attempt {attempt}/{retries})", end=" ")
            if attempt < retries:
                time.sleep(3)
                continue
            return []
        except Exception as exc:
            print(f"\n  WARNING: Failed to load season {season} — {exc}")
            return []

        # FIX 3: Safely retrieve the variable, with a fallback to None check.
        # Previously the code would crash with TypeError if evaluate() returned
        # None (e.g. the variable existed but was null). Now we guard explicitly.
        try:
            matches = page.evaluate("() => datesData")
            teams_data = page.evaluate("() => teamsData")
        except Exception as exc:
            print(f"\n  WARNING: Could not evaluate datesData for season {season} — {exc}")
            return []

        if not matches or not teams_data:
            print(f"\n  WARNING: datesData or teamsData is empty or null for season {season}")
            return []

        # Successfully retrieved data — break out of retry loop
        break

    # Build a lookup for team stats by (date, team_title)
    team_stats = {}
    for team_id, t_info in teams_data.items():
        t_title = normalize(t_info["title"])
        for h in t_info["history"]:
            match_date = str(pd.to_datetime(h["date"]).date())
            
            # Safe extraction of ppda
            ppda_obj = h.get("ppda", {})
            ppda_att = float(ppda_obj.get("att", 0))
            ppda_def = float(ppda_obj.get("def", 1)) # avoid div by zero
            ppda = ppda_att / ppda_def if ppda_def > 0 else 0.0

            team_stats[(match_date, t_title)] = {
                "ppda": ppda,
                "xpts": float(h.get("xpts", 0.0)),
                "npxg": float(h.get("npxG", 0.0)),
                "deep": float(h.get("deep", 0.0))
            }

    rows = []
    for m in matches:
        if not m.get("isResult"):
            continue  # skip unplayed fixtures
        try:
            date_str = str(pd.to_datetime(m["datetime"]).date())
            home_team = normalize(m["h"]["title"])
            away_team = normalize(m["a"]["title"])
            
            home_stats = team_stats.get((date_str, home_team), {})
            away_stats = team_stats.get((date_str, away_team), {})
            
            forecast = m.get("forecast", {})
            
            rows.append({
                "date":       date_str,
                "home_team":  home_team,
                "away_team":  away_team,
                "us_home_xg": float(m["xG"]["h"]),
                "us_away_xg": float(m["xG"]["a"]),
                "us_home_npxg": home_stats.get("npxg"),
                "us_away_npxg": away_stats.get("npxg"),
                "us_home_ppda": home_stats.get("ppda"),
                "us_away_ppda": away_stats.get("ppda"),
                "us_home_deep": home_stats.get("deep"),
                "us_away_deep": away_stats.get("deep"),
                "us_home_xpts": home_stats.get("xpts"),
                "us_away_xpts": away_stats.get("xpts"),
                "us_forecast_w": float(forecast.get("w", 0.0)) if forecast else None,
                "us_forecast_d": float(forecast.get("d", 0.0)) if forecast else None,
                "us_forecast_l": float(forecast.get("l", 0.0)) if forecast else None,
            })
        except (KeyError, TypeError, ValueError):
            continue  # skip malformed entries

    return rows


def main() -> None:
    print("Fetching understat EPL data using Playwright...")
    all_rows: list[dict] = []

    with sync_playwright() as p:
        # Launch Chromium invisibly.
        # NOTE: On some Windows machines, headless=True can trigger bot-detection
        # on certain sites. If you keep getting empty results, try headless=False
        # to watch the browser open — it often bypasses fingerprinting checks.
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

            # FIX 4: Small polite delay between seasons.
            # Rapid sequential requests to the same host can trigger rate-limiting
            # or bot detection. A 1-2s pause keeps you under the radar.
            time.sleep(1.5)

        browser.close()

    if not all_rows:
        print("\nNo data fetched — check internet connection or season list.")
        return

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    OUT_FILE.parent.mkdir(exist_ok=True)
    df.to_csv(OUT_FILE, index=False)

    print(f"\nSaved {len(df)} matches -> {OUT_FILE}")
    print(df[["date", "home_team", "away_team", "us_home_xg", "us_home_ppda", "us_home_xpts"]].head(8).to_string(index=False))


if __name__ == "__main__":
    main()