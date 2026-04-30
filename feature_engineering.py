"""
feature_engineering.py — Advanced feature engineering for the EPL Match Predictor.

Features computed (all strictly pre-match — no leakage):
  - Rolling form points, goals, shots, SOT, corners, fouls, cards (5-match window)
  - Venue-specific rolling stats (home-only / away-only form)
  - Head-to-head win rate (last 6 meetings)
  - Elo ratings (from data_loader)
  - Rest days / fatigue proxy
  - xG proxy: avg_sot × scoring_efficiency (goals per SOT)
  - Shot conversion rate (goals / SOT ratio)
  - Consecutive win/loss streak
  - Cumulative season points (positional pressure proxy)
  - Weekend / midweek fixture flag
  - Bookmaker implied probabilities (optional)

[WHY] All features are computed in a single forward pass through the sorted
DataFrame. Histories are updated AFTER the feature row is recorded, so the
current match's result never influences its own features.
"""

import numpy as np
import pandas as pd

USE_BETTING_ODDS = True  # flag used by train_model.py too


# ===========================================================================
# 1. Rolling stats helper
# ===========================================================================

def _rolling(history: dict, team: str, window: int) -> dict:
    """
    Compute rolling stats for *team* using the last *window* entries in
    *history[team]*. Returns zero-filled defaults when history is empty.

    [WHY] Returning zeros (not NaN) for missing history is semantically
    correct: a team at the start of its first season has no prior form.
    The SimpleImputer in the pipeline will also fill NaN with 0, so this
    keeps both paths consistent.
    """
    last = history.get(team, [])[-window:]
    if not last:
        return {
            "form_points": 0,
            "avg_scored": 0.0,
            "avg_conceded": 0.0,
            "avg_shots": 0.0,
            "avg_shots_on_target": 0.0,
            "avg_corners": 0.0,
            "avg_fouls": 0.0,
            "avg_yellows": 0.0,
            "avg_reds": 0.0,
            "clean_sheet_pct": 0.0,
            "scoring_pct": 0.0,
            "avg_goal_diff": 0.0,
            "shot_conversion": 0.0,     # advanced: goals / SOT
            "xg_proxy": 0.0,            # advanced: SOT × conversion rate
            "streak": 0,                # advanced: consecutive W/L streak
            "avg_real_xg": 0.0,
            "avg_real_xga": 0.0,
            "avg_xg_overperf": 0.0,
            "avg_npxg": 0.0,
            "avg_ppda": 0.0,
            "avg_deep": 0.0,
            "avg_xpts": 0.0,
            "avg_forecast_win": 0.0,
            "avg_forecast_draw": 0.0,
            "avg_forecast_loss": 0.0,
        }

    n = len(last)
    sot_total = sum(x["sot"] for x in last)
    goals_total = sum(x["scored"] for x in last)
    shot_conv = goals_total / sot_total if sot_total > 0 else 0.0
    avg_sot = sot_total / n
    xg_proxy = avg_sot * shot_conv

    # Real xG from understat (None entries = match predates understat merge)
    # [WHY partial?] Early-season matches may lack understat data; we average
    #   only what we have and fall back to xg_proxy when nothing is available.
    xg_vals     = [x["xg"]        for x in last if x.get("xg")        is not None]
    xga_vals    = [x["xga"]       for x in last if x.get("xga")       is not None]
    xg_op_vals  = [x["xg_overperf"] for x in last if x.get("xg_overperf") is not None]

    avg_real_xg      = float(np.mean(xg_vals))    if xg_vals    else xg_proxy
    avg_real_xga     = float(np.mean(xga_vals))   if xga_vals   else xg_proxy
    avg_xg_overperf  = float(np.mean(xg_op_vals)) if xg_op_vals else 0.0

    npxg_vals = [x["npxg"] for x in last if x.get("npxg") is not None]
    ppda_vals = [x["ppda"] for x in last if x.get("ppda") is not None]
    deep_vals = [x["deep"] for x in last if x.get("deep") is not None]
    xpts_vals = [x["xpts"] for x in last if x.get("xpts") is not None]
    fw_vals = [x["forecast_win"] for x in last if x.get("forecast_win") is not None]
    fd_vals = [x["forecast_draw"] for x in last if x.get("forecast_draw") is not None]
    fl_vals = [x["forecast_loss"] for x in last if x.get("forecast_loss") is not None]

    avg_npxg = float(np.mean(npxg_vals)) if npxg_vals else avg_real_xg
    avg_ppda = float(np.mean(ppda_vals)) if ppda_vals else 10.0 # 10 is an average PPDA proxy
    avg_deep = float(np.mean(deep_vals)) if deep_vals else 5.0 # average deep passes proxy
    avg_xpts = float(np.mean(xpts_vals)) if xpts_vals else (1.3 if avg_real_xg > avg_real_xga else 1.0)
    avg_forecast_win = float(np.mean(fw_vals)) if fw_vals else 0.33
    avg_forecast_draw = float(np.mean(fd_vals)) if fd_vals else 0.33
    avg_forecast_loss = float(np.mean(fl_vals)) if fl_vals else 0.33

    # Streak: count consecutive W/L from most recent match backwards
    streak = 0
    direction = None
    for x in reversed(last):
        pts = x["points"]
        cur = 1 if pts == 3 else (-1 if pts == 0 else 0)
        if direction is None:
            direction = cur
        if cur == direction and cur != 0:
            streak += cur
        else:
            break

    return {
        "form_points": sum(x["points"] for x in last),
        "avg_scored": np.mean([x["scored"] for x in last]),
        "avg_conceded": np.mean([x["conceded"] for x in last]),
        "avg_shots": np.mean([x["shots"] for x in last]),
        "avg_shots_on_target": avg_sot,
        "avg_corners": np.mean([x["corners"] for x in last]),
        "avg_fouls": np.mean([x["fouls"] for x in last]),
        "avg_yellows": np.mean([x["yellows"] for x in last]),
        "avg_reds": np.mean([x["reds"] for x in last]),
        "clean_sheet_pct": np.mean([x["clean_sheet"] for x in last]),
        "scoring_pct": np.mean([x["scored_any"] for x in last]),
        "avg_goal_diff": np.mean([x["goal_diff"] for x in last]),
        "shot_conversion": shot_conv,
        "xg_proxy": xg_proxy,
        "streak": streak,
        # --- Real xG (understat) ---
        "avg_real_xg":     avg_real_xg,
        "avg_real_xga":    avg_real_xga,
        "avg_xg_overperf": avg_xg_overperf,
        "avg_npxg":        avg_npxg,
        "avg_ppda":        avg_ppda,
        "avg_deep":        avg_deep,
        "avg_xpts":        avg_xpts,
        "avg_forecast_win": avg_forecast_win,
        "avg_forecast_draw": avg_forecast_draw,
        "avg_forecast_loss": avg_forecast_loss,
    }


# ===========================================================================
# 2. Main feature engineering function
# ===========================================================================

def add_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Forward-pass feature engineering over a chronologically sorted DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Sorted, Elo-enriched DataFrame from data_loader.load_and_prepare().
    window : int
        Look-back window for rolling statistics (default: 5 matches).

    Returns
    -------
    pd.DataFrame
        One row per match with all engineered features.
    """
    # Per-team histories stored as lists of entry dicts
    team_history: dict[str, list[dict]] = {}   # all venues
    home_history: dict[str, list[dict]] = {}   # home matches only
    away_history: dict[str, list[dict]] = {}   # away matches only
    h2h_history: dict[tuple, list[dict]] = {}  # (home_team, away_team) pair

    # Cumulative season points (resets per season via source_file tracking)
    season_pts: dict[str, dict[str, int]] = {}   # {season: {team: pts}}
    # Target encoding: season win counts for leakage-free win-rate computation
    season_wins: dict[str, dict[str, int]] = {}   # {season: {team: wins}}
    season_games: dict[str, dict[str, int]] = {}  # {season: {team: games}}
    last_match_date: dict[str, pd.Timestamp] = {}

    # Multi-window windows beyond the default
    # [WHY 3/10/19?] w=3 = hot streak, w=10 ≈ quarter-season, w=19 = half-season.
    EXTRA_WINDOWS = [3, 10, 19]

    rows = []

    for _, row in df.iterrows():
        home = row["HomeTeam"]
        away = row["AwayTeam"]
        date = row["Date"]
        season = row.get("source_file", "unknown")

        # --- Rolling stats (all venues, default window) ---
        h_all = _rolling(team_history, home, window)
        a_all = _rolling(team_history, away, window)

        # --- Multi-window rolling stats (3, 10, 19 games) ---
        # [WHY] w=3 captures hot streaks, w=10 ≈ quarter-season trend,
        # w=19 = half-season quality. Giving the model all three lets it weight
        # short vs long form dynamically per situation.
        multi_win = {}
        for w in EXTRA_WINDOWS:
            multi_win[(home, w)] = _rolling(team_history, home, w)
            multi_win[(away, w)] = _rolling(team_history, away, w)

        # --- Venue-specific rolling stats ---
        # [WHY] Home/away form captures venue-specific performance; a team
        #        performing well at home may struggle away and vice versa.
        h_home = _rolling(home_history, home, window)
        a_away = _rolling(away_history, away, window)

        # --- Head-to-head ---
        # [WHY] Last 6 H2H meetings (≈ 3 seasons). Tactical mismatches
        #        (e.g. low-block vs high press) persist across seasons.
        h2h_key = (home, away)
        h2h_last = h2h_history.get(h2h_key, [])[-6:]
        # [WHY 0.5 default?] No prior H2H info → treat as perfectly uncertain.
        h2h_home_win_rate = (
            np.mean([x["home_won"] for x in h2h_last]) if h2h_last else 0.5
        )
        # [WHY draw rate?] Some fixture pairings (e.g. defensively-minded sides)
        #   draw far more often than average. This is one of the best draw signals.
        h2h_draw_rate = (
            np.mean([x["draw"] for x in h2h_last]) if h2h_last else 0.27
            # 0.27 ≈ historical EPL draw frequency as neutral prior
        )

        # --- Rest / fatigue ---
        # [WHY] A 3rd match in 7 days correlates with drop in performance.
        home_rest = (date - last_match_date[home]).days if home in last_match_date else 7
        away_rest = (date - last_match_date[away]).days if away in last_match_date else 7

        # --- Cumulative season points (pressure proxy) ---
        # [WHY] A team fighting relegation behaves differently to a side
        #        already safe. Cumulative season pts is a quick approximation.
        if season not in season_pts:
            season_pts[season] = {}
            season_wins[season] = {}
            season_games[season] = {}
        home_season_pts = season_pts[season].get(home, 0)
        away_season_pts = season_pts[season].get(away, 0)

        # --- Target-encoded season win rate (leakage-free via prior games) ---
        # [WHY] Gives the model a continuous strength signal per team per season
        # without seeing the current match. Expanding mean uses shift(1) implicitly
        # since we read BEFORE updating.
        def _safe_win_rate(s, team):
            g = season_games[s].get(team, 0)
            w = season_wins[s].get(team, 0)
            return w / g if g > 0 else 0.33
        home_season_win_rate = _safe_win_rate(season, home)
        away_season_win_rate = _safe_win_rate(season, away)

        feature_row = {
            "Date": date,
            "HomeTeam": home,
            "AwayTeam": away,
            "FTR": row["FTR"],

            # --- Overall form (all venues) ---
            "home_form_points":           h_all["form_points"],
            "away_form_points":           a_all["form_points"],
            "form_diff":                  h_all["form_points"] - a_all["form_points"],

            # --- Goals ---
            "home_avg_goals_scored":      h_all["avg_scored"],
            "away_avg_goals_scored":      a_all["avg_scored"],
            "home_avg_goals_conceded":    h_all["avg_conceded"],
            "away_avg_goals_conceded":    a_all["avg_conceded"],
            "goal_scored_diff":           h_all["avg_scored"] - a_all["avg_scored"],
            "goal_conceded_diff":         h_all["avg_conceded"] - a_all["avg_conceded"],
            "home_avg_goal_diff":         h_all["avg_goal_diff"],
            "away_avg_goal_diff":         a_all["avg_goal_diff"],

            # --- Shots ---
            "home_avg_shots":             h_all["avg_shots"],
            "away_avg_shots":             a_all["avg_shots"],
            "home_avg_sot":               h_all["avg_shots_on_target"],
            "away_avg_sot":               a_all["avg_shots_on_target"],
            "sot_diff":                   h_all["avg_shots_on_target"] - a_all["avg_shots_on_target"],

            # --- xG proxies (advanced) ---
            # [WHY keep both?] avg_real_xg uses understat values where available
            #   and falls back to xg_proxy otherwise, so it's always populated.
            #   xg_overperf is the key mean-reversion signal: teams that score
            #   far above their xG tend to regress; those below tend to improve.
            "home_xg_proxy":              h_all["xg_proxy"],
            "away_xg_proxy":              a_all["xg_proxy"],
            "xg_proxy_diff":              h_all["xg_proxy"] - a_all["xg_proxy"],
            "home_shot_conversion":       h_all["shot_conversion"],
            "away_shot_conversion":       a_all["shot_conversion"],
            "home_avg_real_xg":           h_all["avg_real_xg"],
            "away_avg_real_xg":           a_all["avg_real_xg"],
            "real_xg_diff":               h_all["avg_real_xg"] - a_all["avg_real_xg"],
            "home_avg_real_xga":          h_all["avg_real_xga"],
            "away_avg_real_xga":          a_all["avg_real_xga"],
            "home_xg_overperf":           h_all["avg_xg_overperf"],
            "away_xg_overperf":           a_all["avg_xg_overperf"],
            "xg_overperf_diff":           h_all["avg_xg_overperf"] - a_all["avg_xg_overperf"],

            "home_avg_npxg":              h_all["avg_npxg"],
            "away_avg_npxg":              a_all["avg_npxg"],
            "home_avg_ppda":              h_all["avg_ppda"],
            "away_avg_ppda":              a_all["avg_ppda"],
            "home_avg_deep":              h_all["avg_deep"],
            "away_avg_deep":              a_all["avg_deep"],
            "home_avg_xpts":              h_all["avg_xpts"],
            "away_avg_xpts":              a_all["avg_xpts"],
            "home_avg_forecast_w":        h_all["avg_forecast_win"],
            "away_avg_forecast_w":        a_all["avg_forecast_win"],

            # --- Set pieces & fouls ---
            "home_avg_corners":           h_all["avg_corners"],
            "away_avg_corners":           a_all["avg_corners"],
            "home_avg_fouls":             h_all["avg_fouls"],
            "away_avg_fouls":             a_all["avg_fouls"],

            # --- Discipline ---
            "home_avg_yellows":           h_all["avg_yellows"],
            "away_avg_yellows":           a_all["avg_yellows"],
            "home_avg_reds":              h_all["avg_reds"],
            "away_avg_reds":              a_all["avg_reds"],

            # --- Momentum ---
            "home_clean_sheet_pct":       h_all["clean_sheet_pct"],
            "away_clean_sheet_pct":       a_all["clean_sheet_pct"],
            "home_scoring_pct":           h_all["scoring_pct"],
            "away_scoring_pct":           a_all["scoring_pct"],

            # --- Streak (advanced) ---
            # [WHY] A +3 home streak means 3 consecutive wins; -3 means 3 losses.
            #        Captures momentum beyond raw points.
            "home_streak":                h_all["streak"],
            "away_streak":                a_all["streak"],
            "streak_diff":                h_all["streak"] - a_all["streak"],

            # --- Venue-specific form ---
            "home_venue_form":            h_home["form_points"],
            "away_venue_form":            a_away["form_points"],
            "home_venue_scored":          h_home["avg_scored"],
            "away_venue_scored":          a_away["avg_scored"],
            "home_venue_conceded":        h_home["avg_conceded"],
            "away_venue_conceded":        a_away["avg_conceded"],
            "home_venue_xg":              h_home["xg_proxy"],
            "away_venue_xg":              a_away["xg_proxy"],

            # --- Elo ratings ---
            "home_elo":                   row["home_elo"],
            "away_elo":                   row["away_elo"],
            "elo_diff":                   row["elo_diff"],
            # --- Venue-split Elo (home-Elo vs away-Elo tracks) ---
            # [WHY] home_elo_home tracks quality specifically when playing at home
            # (K=24); away_elo_away when playing away (K=16). venue_elo_diff is
            # more predictive than raw elo_diff because it separates venue effects.
            "home_elo_home":              row["home_elo_home"],
            "away_elo_away":              row["away_elo_away"],
            "venue_elo_diff":             row["venue_elo_diff"],

            # --- Head-to-head ---
            "h2h_home_win_rate":          h2h_home_win_rate,
            "h2h_draw_rate":              h2h_draw_rate,

            # --- Fatigue ---
            "home_rest_days":             home_rest,
            "away_rest_days":             away_rest,
            "rest_diff":                  home_rest - away_rest,

            # --- Season pressure (advanced) ---
            # [WHY] Cumulative pts at kickoff proxy for league position / pressure.
            "home_season_pts":            home_season_pts,
            "away_season_pts":            away_season_pts,
            "season_pts_diff":            home_season_pts - away_season_pts,

            # --- Target-encoded season win rate ---
            # [WHY] Continuous team-strength signal per season without leakage.
            "home_team_season_win_rate":  home_season_win_rate,
            "away_team_season_win_rate":  away_season_win_rate,

            # --- Calendar context ---
            "is_weekend":                 1 if date.dayofweek >= 5 else 0,
            "month":                      date.month,  # seasonality

            # --- Multi-window rolling win/form rates ---
            # [WHY] w=3 captures hot streaks; w=10 quarter-season; w=19 half-season.
            **{f"home_form_pts_w{w}": multi_win[(home, w)]["form_points"] for w in EXTRA_WINDOWS},
            **{f"away_form_pts_w{w}": multi_win[(away, w)]["form_points"] for w in EXTRA_WINDOWS},
            **{f"home_avg_xg_w{w}": multi_win[(home, w)]["avg_real_xg"] for w in EXTRA_WINDOWS},
            **{f"away_avg_xg_w{w}": multi_win[(away, w)]["avg_real_xg"] for w in EXTRA_WINDOWS},
            **{f"xg_diff_w{w}": multi_win[(home, w)]["avg_real_xg"] - multi_win[(away, w)]["avg_real_xg"] for w in EXTRA_WINDOWS},
            **{f"home_avg_ppda_w{w}": multi_win[(home, w)]["avg_ppda"] for w in EXTRA_WINDOWS},
            **{f"away_avg_ppda_w{w}": multi_win[(away, w)]["avg_ppda"] for w in EXTRA_WINDOWS},
        }

        # --- Betting odds (optional) ---
        if USE_BETTING_ODDS:
            b365h = row.get("B365H", np.nan)
            b365d = row.get("B365D", np.nan)
            b365a = row.get("B365A", np.nan)
            avgh  = row.get("AvgH",  np.nan)
            avgd  = row.get("AvgD",  np.nan)
            avga  = row.get("AvgA",  np.nan)

            if all(pd.notna([b365h, b365d, b365a])) and b365h > 0:
                raw_h, raw_d, raw_a = 1/b365h, 1/b365d, 1/b365a
                total = raw_h + raw_d + raw_a
                feature_row["b365_prob_h"] = raw_h / total
                feature_row["b365_prob_d"] = raw_d / total
                feature_row["b365_prob_a"] = raw_a / total
            else:
                feature_row["b365_prob_h"] = np.nan
                feature_row["b365_prob_d"] = np.nan
                feature_row["b365_prob_a"] = np.nan

            if all(pd.notna([avgh, avgd, avga])) and avgh > 0:
                raw_h, raw_d, raw_a = 1/avgh, 1/avgd, 1/avga
                total = raw_h + raw_d + raw_a
                feature_row["avg_prob_h"] = raw_h / total
                feature_row["avg_prob_d"] = raw_d / total
                feature_row["avg_prob_a"] = raw_a / total
            else:
                feature_row["avg_prob_h"] = np.nan
                feature_row["avg_prob_d"] = np.nan
                feature_row["avg_prob_a"] = np.nan

        rows.append(feature_row)

        # =================================================================
        # UPDATE HISTORIES — always AFTER appending the feature row
        # [WHY] If we updated before, the current result would leak into
        #        the current match's own features.
        # =================================================================
        home_pts = 3 if row["FTR"] == "H" else 1 if row["FTR"] == "D" else 0
        away_pts = 3 if row["FTR"] == "A" else 1 if row["FTR"] == "D" else 0

        fthg = row["FTHG"]
        ftag = row["FTAG"]

        def gcol(col, fallback=0):
            v = row.get(col, fallback)
            return v if pd.notna(v) else fallback

        def make_entry(pts, scored, conceded, shots, sot, corners, fouls, yellows, reds,
                       xg=None, xga=None, npxg=None, ppda=None, deep=None, xpts=None, forecast_win=None, forecast_draw=None, forecast_loss=None):
            # Compute per-entry shot conversion for this match
            match_conv = scored / sot if sot > 0 else 0.0
            xg_overperf = (scored - xg) if xg is not None else None
            return {
                "points":      pts,
                "scored":      scored,
                "conceded":    conceded,
                "shots":       shots,
                "sot":         sot,
                "corners":     corners,
                "fouls":       fouls,
                "yellows":     yellows,
                "reds":        reds,
                "clean_sheet": int(conceded == 0),
                "scored_any":  int(scored > 0),
                "goal_diff":   scored - conceded,
                "shot_conv":   match_conv,
                # Real xG (None when understat data not available for this match)
                "xg":          xg,
                "xga":         xga,
                "xg_overperf": xg_overperf,
                "npxg":        npxg,
                "ppda":        ppda,
                "deep":        deep,
                "xpts":        xpts,
                "forecast_win": forecast_win,
                "forecast_draw": forecast_draw,
                "forecast_loss": forecast_loss,
            }

        # Read real xG from understat columns (NaN if not merged / not available)
        _hxg = row.get("us_home_xg")
        _axg = row.get("us_away_xg")
        home_real_xg = float(_hxg) if pd.notna(_hxg) else None
        away_real_xg = float(_axg) if pd.notna(_axg) else None

        def get_stat(col):
            val = row.get(col)
            return float(val) if pd.notna(val) else None

        home_entry = make_entry(
            home_pts, fthg, ftag,
            gcol("HS"), gcol("HST"), gcol("HC"), gcol("HF"), gcol("HY"), gcol("HR"),
            xg=home_real_xg, xga=away_real_xg,
            npxg=get_stat("us_home_npxg"), ppda=get_stat("us_home_ppda"),
            deep=get_stat("us_home_deep"), xpts=get_stat("us_home_xpts"),
            forecast_win=get_stat("us_forecast_w"), forecast_draw=get_stat("us_forecast_d"), forecast_loss=get_stat("us_forecast_l")
        )
        away_entry = make_entry(
            away_pts, ftag, fthg,
            gcol("AS"), gcol("AST"), gcol("AC"), gcol("AF"), gcol("AY"), gcol("AR"),
            xg=away_real_xg, xga=home_real_xg,
            npxg=get_stat("us_away_npxg"), ppda=get_stat("us_away_ppda"),
            deep=get_stat("us_away_deep"), xpts=get_stat("us_away_xpts"),
            forecast_win=get_stat("us_forecast_l"), forecast_draw=get_stat("us_forecast_d"), forecast_loss=get_stat("us_forecast_w")
        )

        team_history.setdefault(home, []).append(home_entry)
        team_history.setdefault(away, []).append(away_entry)
        home_history.setdefault(home, []).append(home_entry)
        away_history.setdefault(away, []).append(away_entry)
        h2h_history.setdefault(h2h_key, []).append({
            "home_won": int(row["FTR"] == "H"),
            "draw":     int(row["FTR"] == "D"),
        })

        # Update season cumulative points and win counts (for target encoding)
        season_pts[season][home] = season_pts[season].get(home, 0) + home_pts
        season_pts[season][away] = season_pts[season].get(away, 0) + away_pts
        season_games[season][home] = season_games[season].get(home, 0) + 1
        season_games[season][away] = season_games[season].get(away, 0) + 1
        if row["FTR"] == "H":
            season_wins[season][home] = season_wins[season].get(home, 0) + 1
        elif row["FTR"] == "A":
            season_wins[season][away] = season_wins[season].get(away, 0) + 1

        last_match_date[home] = date
        last_match_date[away] = date

    return pd.DataFrame(rows)


# ===========================================================================
# 3. Feature column definitions (used by train_model.py and app.py)
# ===========================================================================

BASE_NUMERIC_COLS = [
    "home_form_points", "away_form_points", "form_diff",
    "home_avg_goals_scored", "away_avg_goals_scored",
    "home_avg_goals_conceded", "away_avg_goals_conceded",
    "goal_scored_diff", "goal_conceded_diff",
    "home_avg_goal_diff", "away_avg_goal_diff",
    "home_avg_shots", "away_avg_shots",
    "home_avg_sot", "away_avg_sot", "sot_diff",
    "home_xg_proxy", "away_xg_proxy", "xg_proxy_diff",
    "home_shot_conversion", "away_shot_conversion",
    "home_avg_real_xg", "away_avg_real_xg", "real_xg_diff",
    "home_avg_real_xga", "away_avg_real_xga",
    "home_xg_overperf", "away_xg_overperf", "xg_overperf_diff",
    "home_avg_npxg", "away_avg_npxg",
    "home_avg_ppda", "away_avg_ppda",
    "home_avg_deep", "away_avg_deep",
    "home_avg_xpts", "away_avg_xpts",
    "home_avg_forecast_w", "away_avg_forecast_w",
    "home_avg_corners", "away_avg_corners",
    "home_avg_fouls", "away_avg_fouls",
    "home_avg_yellows", "away_avg_yellows",
    "home_avg_reds", "away_avg_reds",
    "home_clean_sheet_pct", "away_clean_sheet_pct",
    "home_scoring_pct", "away_scoring_pct",
    "home_streak", "away_streak", "streak_diff",
    "home_venue_form", "away_venue_form",
    "home_venue_scored", "away_venue_scored",
    "home_venue_conceded", "away_venue_conceded",
    "home_venue_xg", "away_venue_xg",
    "home_elo", "away_elo", "elo_diff",
    "home_elo_home", "away_elo_away", "venue_elo_diff",
    "h2h_home_win_rate", "h2h_draw_rate",
    "home_rest_days", "away_rest_days", "rest_diff",
    "home_season_pts", "away_season_pts", "season_pts_diff",
    "home_team_season_win_rate", "away_team_season_win_rate",
    "is_weekend", "month",
    # Multi-window form points (w=3, 10, 19)
    "home_form_pts_w3", "away_form_pts_w3",
    "home_form_pts_w10", "away_form_pts_w10",
    "home_form_pts_w19", "away_form_pts_w19",
    # Multi-window xG and diff
    "home_avg_xg_w3", "away_avg_xg_w3", "xg_diff_w3",
    "home_avg_xg_w10", "away_avg_xg_w10", "xg_diff_w10",
    "home_avg_xg_w19", "away_avg_xg_w19", "xg_diff_w19",
    # Multi-window PPDA (pressing intensity)
    "home_avg_ppda_w3", "away_avg_ppda_w3",
    "home_avg_ppda_w10", "away_avg_ppda_w10",
    "home_avg_ppda_w19", "away_avg_ppda_w19",
]

ODDS_COLS = [
    "b365_prob_h", "b365_prob_d", "b365_prob_a",
    "avg_prob_h", "avg_prob_d", "avg_prob_a",
]

CAT_COLS = ["HomeTeam", "AwayTeam"]
