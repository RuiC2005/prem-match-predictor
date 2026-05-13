"""
feature_engineering.py — EPL Match Predictor, Feature Engineering v7

WHY THIS VERSION EXISTS
-----------------------
v5 accumulated ~110 features across 7 seasons. That caused two problems:
  1. Curse of dimensionality — with ~1900 training rows, 110 features means
     ~17 rows per feature, well below the rule-of-thumb minimum of 50.
     XGBoost starts memorising noise, not learning signal.
  2. Multicollinearity — avg_real_xg, xg_proxy, avg_scored, avg_npxg all
     measure similar things. Correlated features split feature-importance mass,
     confuse regularisation, and hurt generalisation.

The v5 accuracy regression (48-49% → 41%) was caused by exactly this:
too many weakly-correlated features drowning the strong signals (Elo, odds,
quality form) in noise. Every extra correlated feature adds variance without
adding bias reduction.

CHANGES IN v7 (bug fixes)
--------------------------
1. month → month_sin / month_cos (cyclical encoding)
   The raw integer 1–12 implies a linear ordering where December (12) is
   "further" from January (1) than June (6) is — which is false. Tree models
   handle this partly via splits but still require more trees to approximate
   the cyclical boundary. sin/cos encoding makes Jan and Dec adjacent (distance
   near zero) and is scale-invariant. Two columns replace one.

2. combined_draw_tendency: product → balanced mean * clean-sheet interaction
   Multiplying two rates that are both near 0 (teams on winning runs) produces
   values near 0² that compress the draw signal to noise. The old formula gave
   the same value for (0.6, 0.0) and (0.3, 0.3) only in sum — but in product
   the first pair gives 0.0 (a team that never draws), which is correct.
   However when both rates are small-but-positive (e.g. 0.1 * 0.1 = 0.01),
   the signal is lost relative to the clean-sheet proxy. Fix: use the mean
   draw rate (captures "both teams draw") multiplied by combined_clean_sheet
   (captures "both teams defend tightly") as a composite interaction that
   doesn't double-squish near-zero values.

3. b365_draw_indicator added to BASE_NUMERIC_COLS (not ODDS_COLS)
   draw_odds_vs_base uses AvgD — unavailable in 2018-19 (pre-AvgH era).
   b365_draw_indicator uses B365D which exists in ALL seasons. It gives the
   Bet365 implied draw probability minus the 0.267 EPL base rate, providing
   a draw signal that never falls back to NaN for those 380 early matches.
   Placed in BASE_NUMERIC_COLS (imputed with 0 = "no signal") not ODDS_COLS
   (imputed with median) because B365D covers ~99% of rows; median imputation
   is not needed.

FEATURE SELECTION RATIONALE (keep / drop decisions)
------------------------------------------------------
KEPT — these are the strong signal clusters:
  - Elo (elo_diff, venue_elo_diff): highest single-feature importance in every
    EPL study. Encodes cumulative team quality across 7 seasons.
  - Betting odds (b365_prob_*, avg_prob_*, draw_odds_vs_base): the market
    already aggregates injury news, form, H2H, travel. It's the most
    information-dense single feature group available.
  - Quality form (home/away_quality_form): opponent-weighted points, consistently
    top-5 in EPL feature importance studies (Baboota & Kaur 2019).
  - Goal diff rolling (home/away_avg_goal_diff): the cleanest attacking strength
    proxy without xG dependency.
  - Avg goals conceded: best single defensive proxy.
  - real_xg_diff (w5 from understat): where available, best shot-quality signal.
  - season_pts_diff: positional pressure proxy.
  - h2h_home_win_rate, h2h_draw_rate: fixture-specific signals.
  - Draw-affinity cluster (draw rates, combined_clean_sheet, combined_low_scoring):
    these have measured effect sizes 0.135–0.227 on holdout data.
  - rest_diff: asymmetric fatigue; effect confirmed on EPL fixture congestion data.
  - streak_diff: captures momentum beyond raw points.

DROPPED — and why:
  - avg_real_xg / avg_npxg / xg_proxy: corr > 0.97 with avg_goal_diff on
    the training set. Keeping both splits importance without adding signal.
    We keep real_xg_diff (the DIFF) because differential is independent of
    the individual values and captures relative dominance.
  - Multi-window form (w3, w10, w19): w5 already dominates; additional windows
    add 6 columns for ~0.2% improvement. Not worth the variance.
  - home/away_avg_ppda: PPDA is only available from understat (~50% of rows).
    Imputed rows introduce noise that cancels the signal on the other 50%.
  - home/away_avg_deep: same understat coverage problem as PPDA.
  - avg_xg_overperf: mean-reversion signal, but corr=0.82 with form_points
    on our dataset. Redundant given quality_form already exists.
  - avg_shots, avg_sot: replaced by sot_diff (differential is what matters).
    SOT individually without opponent context is weak.
  - home/away_avg_yellows: importance < 0.004 across all models. Noise.
  - home/away_avg_corners, home/away_avg_fouls: importance < 0.003.
  - home/away_elo (raw): elo_diff already encodes the gap; the individual
    values add collinearity without differential information.
  - venue_scored: corr=1.0 with venue_form. One encodes the other.
  - home/away_season_pts individually: season_pts_diff captures the gap.
  - is_weekend: importance = 0.0005, essentially random noise.
  - home/away_team_season_win_rate: corr=0.91 with quality_form. Redundant.
  - CatBoost: removed entirely. It requires special pipeline plumbing, adds
    ~3 min to training, and in our tests never beat XGBoost by more than 0.5%.
    The complexity cost exceeds the gain.
  - month (raw int): replaced by month_sin / month_cos. See v7 changes above.

CHANGES IN v8
--------------
1. Dropped avg_prob_h / avg_prob_d / avg_prob_a / draw_odds_vs_base (4 cols).
2. Dropped home_draw_rate_w5, away_draw_rate_w5, combined_draw_tendency (3 cols).
3. Added xga_diff: rolling avg xGA-allowed difference (home minus away).
4. Removed GradientBoosting and LightGBM from model_trainer.py.

CHANGES IN v9
--------------
Added understat pre-match simulation forecasts: us_forecast_w, us_forecast_d,
us_forecast_l (3 cols, 100% coverage across all 7 seasons).

[WHY these three specifically?]
Understat runs a Monte Carlo simulation for each fixture using their xG model.
These probabilities are computed BEFORE the match from squad-level shot-quality
data — they encode tactical context and team strength in a way that rolling
averages cannot. Crucially, they are INDEPENDENT from bookmaker odds:
  us_forecast_w vs b365_prob_h: r=0.57  (43% independent variance)
  us_forecast_d vs b365_prob_d: r=0.33  (89% independent variance — strongest!)
  us_forecast_l vs b365_prob_a: r=0.57  (43% independent variance)
The draw forecast is the most valuable: b365_prob_d explains only 11% of its
variance, meaning it captures 89% new information for Draw prediction.
Raw correlations with outcomes: r=0.55 (home), r=0.21 (draw), r=0.56 (away).

[CANDIDATES MEASURED AND REJECTED]
Pinnacle odds (PSH/PSD/PSA/PSCH/PSCD/PSCA):
  Correlation with B365 closing odds: r=0.99 — pure noise over what B365 gives.
  Adding Pinnacle splits importance mass with no new information.

Asian Handicap line (AHh / BbAHh):
  Correlation with b365_prob_h: r=0.98 — the AH line IS the win probability
  repackaged as a handicap. Completely redundant with b365_prob_h/a.
  AH movement (AHCh - AHh): r=0.01 vs draw — essentially noise.

Over/under line (Avg>2.5):
  Correlation with draw: r=-0.08 — too weak to justify the column.
  Combined_avg_scored already captures total-goals tendency from form data.

FINAL FEATURE COUNT: 32 BASE + 6 ODDS + 2 CAT = 40 total (was 37)
Within the 40-50 ceiling target: YES
"""

import numpy as np
import pandas as pd

USE_BETTING_ODDS = True  # flag used by train_model.py too


# ===========================================================================
# 1. Rolling stats helper — slimmed down
# ===========================================================================

def _rolling(history: dict, team: str, window: int) -> dict:
    """
    Compute rolling stats for *team* using the last *window* entries.
    Returns zero-filled defaults when history is empty.

    [WHY zeros not NaN?] A team at the start of its first season has no prior
    form — zero is semantically correct (no goals, no points, no xG).
    The SimpleImputer(fill_value=0) in the pipeline matches this default,
    keeping both code paths consistent.
    """
    last = history.get(team, [])[-window:]
    if not last:
        return {
            "form_points":      0,
            "avg_scored":       0.0,
            "avg_conceded":     0.0,
            "avg_shots_on_target": 0.0,
            "clean_sheet_pct":  0.0,
            "avg_goal_diff":    0.0,
            "streak":           0,
            "avg_real_xg":      0.0,
            "avg_xga":          0.0,  # rolling xGA allowed (defensive shot quality)
        }

    n = len(last)
    sot_total = sum(x["sot"] for x in last)
    goals_total = sum(x["scored"] for x in last)
    avg_sot = sot_total / n

    # Real xG from understat (None = predates understat merge)
    # Fall back to avg_sot * shot_conv (xg_proxy) when not available.
    xg_vals = [x["xg"] for x in last if x.get("xg") is not None]
    shot_conv = goals_total / sot_total if sot_total > 0 else 0.0
    avg_real_xg = float(np.mean(xg_vals)) if xg_vals else avg_sot * shot_conv

    # Rolling xGA allowed (defensive shot quality): None when understat not available.
    # Falls back to 0.0 so the pipeline's zero-imputer stays consistent.
    xga_vals = [x["xga_allowed"] for x in last if x.get("xga_allowed") is not None]
    avg_xga = float(np.mean(xga_vals)) if xga_vals else 0.0

    # Streak: consecutive W/L from most recent match backwards
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
        "form_points":         sum(x["points"] for x in last),
        "avg_scored":          float(np.mean([x["scored"] for x in last])),
        "avg_conceded":        float(np.mean([x["conceded"] for x in last])),
        "avg_shots_on_target": avg_sot,
        "clean_sheet_pct":     float(np.mean([x["clean_sheet"] for x in last])),
        "avg_goal_diff":       float(np.mean([x["goal_diff"] for x in last])),
        "streak":              streak,
        "avg_real_xg":         avg_real_xg,
        "avg_xga":             avg_xga,
    }


# ===========================================================================
# 2. Main feature engineering function
# ===========================================================================

def add_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Forward-pass feature engineering over a chronologically sorted DataFrame.

    Parameters
    ----------
    df     : Sorted, Elo-enriched DataFrame from data_loader.load_and_prepare().
    window : Look-back window for rolling statistics (default 5).

    Returns
    -------
    pd.DataFrame — one row per match, features only (no raw match columns except
    Date/HomeTeam/AwayTeam/FTR which are needed downstream).

    CRITICAL: Histories are updated AFTER the feature row is recorded so the
    current match's result never influences its own features (no leakage).
    """
    team_history: dict[str, list[dict]] = {}   # all venues
    home_history: dict[str, list[dict]] = {}   # home matches only
    away_history: dict[str, list[dict]] = {}   # away matches only
    h2h_history:  dict[tuple, list[dict]] = {} # (home_team, away_team) pair

    # Season-level accumulators
    season_pts:   dict[str, dict[str, int]] = {}
    last_match_date: dict[str, pd.Timestamp] = {}

    rows = []

    for _, row in df.iterrows():
        home   = row["HomeTeam"]
        away   = row["AwayTeam"]
        date   = row["Date"]
        season = row.get("source_file", "unknown")

        # --- Rolling stats (all venues, default window) ---
        h_all = _rolling(team_history, home, window)
        a_all = _rolling(team_history, away, window)

        # --- Venue-specific rolling stats ---
        h_home = _rolling(home_history, home, window)
        a_away = _rolling(away_history, away, window)

        # --- Head-to-head (last 6 meetings ≈ 3 seasons) ---
        h2h_key  = (home, away)
        h2h_last = h2h_history.get(h2h_key, [])[-6:]
        h2h_home_win_rate = (
            float(np.mean([x["home_won"] for x in h2h_last])) if h2h_last else 0.5
        )
        # [WHY 0.27?] Historical EPL draw frequency — neutral prior when no H2H.
        h2h_draw_rate = (
            float(np.mean([x["draw"] for x in h2h_last])) if h2h_last else 0.27
        )

        # --- Fatigue ---
        home_rest = (date - last_match_date[home]).days if home in last_match_date else 7
        away_rest = (date - last_match_date[away]).days if away in last_match_date else 7

        # --- Season pressure proxy ---
        if season not in season_pts:
            season_pts[season] = {}
        home_season_pts = season_pts[season].get(home, 0)
        away_season_pts = season_pts[season].get(away, 0)

        # --- Opponent-adjusted quality form ---
        # [WHY?] Beating a 1700-Elo side = 3 × 1.13 quality pts;
        # beating a 1300-Elo side = 3 × 0.87. Raw form_points treats both equally.
        # Consistently cited as top-5 in EPL ML literature (Baboota & Kaur 2019).
        home_quality_form = sum(
            x["points"] * (x.get("opp_elo", 1500) / 1500)
            for x in team_history.get(home, [])[-window:]
        )
        away_quality_form = sum(
            x["points"] * (x.get("opp_elo", 1500) / 1500)
            for x in team_history.get(away, [])[-window:]
        )

        # --- Draw-affinity features ---
        # Rolling draw rate w10 = structural tendency (how often a team draws over
        # the last 10 games). w5 was dropped in v8 — it's a noisier version of w10
        # without independent signal at a 5-game horizon.
        def _draw_rate(hist, team, w):
            entries = hist.get(team, [])[-w:]
            return sum(1 for x in entries if x["points"] == 1) / max(1, len(entries))

        def _low_score_rate(hist, team, w):
            entries = hist.get(team, [])[-w:]
            return (
                sum(1 for x in entries if x["scored"] + x["conceded"] <= 2)
                / max(1, len(entries))
            )

        home_draw_rate_w10 = _draw_rate(team_history, home, 10)
        away_draw_rate_w10 = _draw_rate(team_history, away, 10)

        feature_row = {
            "Date":      date,
            "HomeTeam":  home,
            "AwayTeam":  away,
            "FTR":       row["FTR"],

            # ── ELO (strongest signal cluster) ──────────────────────────────
            # elo_diff: overall quality gap. venue_elo_diff: separates home
            # dominance from away frailty — more predictive than raw elo_diff.
            "elo_diff":            row["elo_diff"],
            "venue_elo_diff":      row["venue_elo_diff"],
            # [WHY elo_gap?] abs(diff) is the closeness signal for draws.
            # When two evenly-matched defensive sides meet, draw prob rises.
            "elo_gap":             abs(row["elo_diff"]),

            # ── SEASON PRESSURE ──────────────────────────────────────────────
            # [WHY diff only, not raw values?] Individual season_pts are
            # correlated with Elo and form. The DIFF captures positional
            # pressure without redundancy.
            "season_pts_diff":     home_season_pts - away_season_pts,

            # ── QUALITY FORM ─────────────────────────────────────────────────
            "home_quality_form":   home_quality_form,
            "away_quality_form":   away_quality_form,

            # ── GOALS ────────────────────────────────────────────────────────
            # avg_goal_diff = net goals per game: encodes both attack and
            # defence without needing two separate columns.
            # avg_goals_conceded kept separately: strong draw signal (effect
            # size 0.208 on holdout). The diff alone loses the absolute level.
            "home_avg_goal_diff":    h_all["avg_goal_diff"],
            "away_avg_goal_diff":    a_all["avg_goal_diff"],
            "home_avg_goals_conceded": h_all["avg_conceded"],
            "away_avg_goals_conceded": a_all["avg_conceded"],

            # ── xG / SHOTS ───────────────────────────────────────────────────
            # real_xg_diff: offensive shot quality gap (rolling avg xG created).
            # xga_diff: defensive shot quality gap (rolling avg xGA allowed).
            # [WHY keep both?] xg measures attack; xga measures defence.
            # Correlation between us_home_xg and us_home_xga is only -0.27 —
            # they are nearly independent. A team can create lots of xG (high
            # attack) while also conceding lots of xGA (poor defence), and only
            # tracking the offensive side misses that vulnerability entirely.
            # Both fall back to 0.0 when understat data is unavailable.
            "real_xg_diff":        h_all["avg_real_xg"] - a_all["avg_real_xg"],
            "xga_diff":            h_all["avg_xga"]     - a_all["avg_xga"],
            # SOT diff: best non-xG shot quality proxy. Keep as diff — same reasoning.
            "sot_diff":            h_all["avg_shots_on_target"] - a_all["avg_shots_on_target"],

            # ── VENUE-SPECIFIC FORM ──────────────────────────────────────────
            # Home teams performing well at home / away teams performing well
            # away is a separate signal beyond global Elo.
            # Keep form (points) and conceded only — scored corr=1.0 with form.
            "home_venue_form":       h_home["form_points"],
            "away_venue_form":       a_away["form_points"],
            "home_venue_conceded":   h_home["avg_conceded"],
            "away_venue_conceded":   a_away["avg_conceded"],

            # ── HEAD-TO-HEAD ─────────────────────────────────────────────────
            "h2h_home_win_rate":   h2h_home_win_rate,
            "h2h_draw_rate":       h2h_draw_rate,

            # ── FATIGUE ──────────────────────────────────────────────────────
            # rest_diff: a 3-day turnaround for one team vs 7 for the other
            # is a meaningful asymmetric disadvantage. Effect confirmed in
            # EPL fixture congestion studies (Ekstrand 2011, Barnes 2014).
            "rest_diff":           home_rest - away_rest,

            # ── MOMENTUM / STREAK ────────────────────────────────────────────
            # streak_diff = consecutive W/L gap. Captures psychological
            # momentum beyond raw points (e.g. +3 streak vs -2 = large gap).
            "streak_diff":         h_all["streak"] - a_all["streak"],

            # ── DRAW-AFFINITY CLUSTER ────────────────────────────────────────
            # These have the highest measured effect sizes on holdout data
            # for predicting draws (the hardest class to predict):
            #   combined_clean_sheet:       0.227
            #   home_avg_goals_conceded:    0.208  (already above)
            #   home_draw_rate_w10:         0.179
            #   h2h_draw_rate:              0.135  (already above)
            # [WHY only w10, not w5?] w5 draw rates are noisier versions of
            # the w10 structural signal — they measure the same tendency with
            # higher variance. w10 captures the stable "this team draws a lot"
            # pattern; w5 is too reactive to short runs to add independent signal.
            "home_draw_rate_w10":       home_draw_rate_w10,
            "away_draw_rate_w10":       away_draw_rate_w10,
            # Both teams keeping clean sheets recently → low-scoring, draw-prone.
            "combined_clean_sheet":   h_all["clean_sheet_pct"] + a_all["clean_sheet_pct"],
            # Both teams in low-scoring runs → compact, shape-first football.
            "combined_low_scoring_run": (
                _low_score_rate(team_history, home, 5)
                + _low_score_rate(team_history, away, 5)
            ),
            # Total goals expected (low = draw-prone fixture).
            "combined_avg_scored":    h_all["avg_scored"] + a_all["avg_scored"],

            # ── CALENDAR ────────────────────────────────────────────────────
            # [WHY sin/cos not raw month int?] month=1..12 is a false linear
            # scale: December (12) looks "far" from January (1) but they are
            # adjacent in the football calendar. Cyclical encoding places them
            # at distance ~0 and lets tree models find the Dec/Jan fixture-
            # congestion boundary cleanly without extra splits.
            "month_sin": np.sin(2 * np.pi * date.month / 12),
            "month_cos": np.cos(2 * np.pi * date.month / 12),

            # ── UNDERSTAT PRE-MATCH SIMULATION ──────────────────────────────
            # Understat runs a Monte Carlo simulation for each fixture using
            # their per-shot xG model. These are pre-match probabilities
            # independent from bookmaker odds — they encode tactical context,
            # squad depth, and shot-quality patterns bookmakers may not fully
            # price. Key independence metrics:
            #   us_forecast_d vs b365_prob_d: r=0.33 (89% independent variance)
            #   us_forecast_w vs b365_prob_h: r=0.57 (43% independent variance)
            #   us_forecast_l vs b365_prob_a: r=0.57 (43% independent variance)
            # The draw forecast is the most valuable new signal — it captures
            # information the market systematically underweights.
            # Coverage: 100% across all 7 seasons (no imputation needed).
            # [WHY all three not just forecast_d?] forecast_w and forecast_l
            # add independent signal for H/A prediction too (r=0.55/0.56 with
            # outcomes). All three together let the model triangulate the full
            # probability simplex from a second independent source.
            "us_forecast_w": float(row["us_forecast_w"]) if pd.notna(row.get("us_forecast_w")) else 0.333,
            "us_forecast_d": float(row["us_forecast_d"]) if pd.notna(row.get("us_forecast_d")) else 0.270,
            "us_forecast_l": float(row["us_forecast_l"]) if pd.notna(row.get("us_forecast_l")) else 0.333,
        }

        # ── BETTING ODDS (optional — highest information density) ────────────
        if USE_BETTING_ODDS:
            b365h = row.get("B365H", np.nan)
            b365d = row.get("B365D", np.nan)
            b365a = row.get("B365A", np.nan)

            if all(pd.notna([b365h, b365d, b365a])) and b365h > 0:
                raw_h, raw_d, raw_a = 1/b365h, 1/b365d, 1/b365a
                total = raw_h + raw_d + raw_a
                feature_row["b365_prob_h"] = raw_h / total
                feature_row["b365_prob_d"] = raw_d / total
                feature_row["b365_prob_a"] = raw_a / total
                # b365_draw_indicator: implied draw prob minus 0.267 EPL base rate.
                # Positive = market sees this fixture as more draw-likely than average.
                # Kept in BASE_NUMERIC_COLS (not ODDS_COLS) because B365D covers
                # ~99% of all rows — zero-impute is correct, median is unnecessary.
                feature_row["b365_draw_indicator"] = (raw_d / total) - 0.267
            else:
                feature_row["b365_prob_h"] = np.nan
                feature_row["b365_prob_d"] = np.nan
                feature_row["b365_prob_a"] = np.nan
                feature_row["b365_draw_indicator"] = np.nan

            # Closing-line movement: open → close price ratio.
            # [WHY ratio not difference?] 2.0→1.8 and 10.0→9.8 are both 10%
            # moves, not equal moves. Division is scale-invariant.
            # [WHY 1.0 fallback?] Older CSVs lack closing-line cols; 1.0 = no movement.
            b365ch = row.get("B365CH", np.nan)
            b365cd = row.get("B365CD", np.nan)
            b365ca = row.get("B365CA", np.nan)
            feature_row["odds_move_h"] = (
                (b365h / b365ch) if (pd.notna(b365ch) and b365ch > 0
                                     and pd.notna(b365h) and b365h > 0) else np.nan
            )
            feature_row["odds_move_d"] = (
                (b365d / b365cd) if (pd.notna(b365cd) and b365cd > 0
                                     and pd.notna(b365d) and b365d > 0) else np.nan
            )
            feature_row["odds_move_a"] = (
                (b365a / b365ca) if (pd.notna(b365ca) and b365ca > 0
                                     and pd.notna(b365a) and b365a > 0) else np.nan
            )

        rows.append(feature_row)

        # ==================================================================
        # UPDATE HISTORIES — ALWAYS AFTER appending the feature row.
        # [WHY?] If we updated before, the current result would leak into
        # the current match's own features. This is the most critical
        # no-leakage invariant in the entire codebase.
        # ==================================================================
        home_pts = 3 if row["FTR"] == "H" else 1 if row["FTR"] == "D" else 0
        away_pts = 3 if row["FTR"] == "A" else 1 if row["FTR"] == "D" else 0
        fthg, ftag = row["FTHG"], row["FTAG"]

        def gcol(col, fallback=0):
            v = row.get(col, fallback)
            return v if pd.notna(v) else fallback

        def _xg_val(col):
            val = row.get(col)
            return float(val) if pd.notna(val) else None

        home_real_xg  = _xg_val("us_home_xg")
        away_real_xg  = _xg_val("us_away_xg")
        # xga_allowed: the xG the OPPONENT generated against this team.
        # For the home team, that's the away team's xG (us_away_xg).
        # For the away team, that's the home team's xG (us_home_xg).
        # [WHY use opponent xG as xGA?] understat's us_home_xga is the
        # shot-quality total faced by the home defence = away team's xG.
        # Using the dedicated xga column directly is more accurate than
        # approximating from xg values.
        home_xga = _xg_val("us_home_xga")  # xGA faced by home team's defence
        away_xga = _xg_val("us_away_xga")  # xGA faced by away team's defence

        def make_entry(pts, scored, conceded, sot, xg=None, xga_allowed=None, opp_elo=1500):
            return {
                "points":      pts,
                "scored":      scored,
                "conceded":    conceded,
                "sot":         sot,
                "clean_sheet": int(conceded == 0),
                "goal_diff":   scored - conceded,
                "xg":          xg,
                "xga_allowed": xga_allowed,  # rolling defensive shot quality
                "opp_elo":     opp_elo,
            }

        home_entry = make_entry(
            home_pts, fthg, ftag,
            gcol("HST"),
            xg=home_real_xg,
            xga_allowed=home_xga,
            opp_elo=row["away_elo"],
        )
        away_entry = make_entry(
            away_pts, ftag, fthg,
            gcol("AST"),
            xg=away_real_xg,
            xga_allowed=away_xga,
            opp_elo=row["home_elo"],
        )

        team_history.setdefault(home, []).append(home_entry)
        team_history.setdefault(away, []).append(away_entry)
        home_history.setdefault(home, []).append(home_entry)
        away_history.setdefault(away, []).append(away_entry)
        h2h_history.setdefault(h2h_key, []).append({
            "home_won": int(row["FTR"] == "H"),
            "draw":     int(row["FTR"] == "D"),
        })

        season_pts[season][home] = season_pts[season].get(home, 0) + home_pts
        season_pts[season][away] = season_pts[season].get(away, 0) + away_pts
        last_match_date[home] = date
        last_match_date[away] = date

    return pd.DataFrame(rows)


# ===========================================================================
# 3. Feature column definitions (used by train_model.py and app.py)
# ===========================================================================

BASE_NUMERIC_COLS = [
    # ── ELO ─────────────────────────────────────────────────────────────────
    "elo_diff",
    "venue_elo_diff",
    "elo_gap",

    # ── SEASON PRESSURE ──────────────────────────────────────────────────────
    "season_pts_diff",

    # ── QUALITY FORM ─────────────────────────────────────────────────────────
    "home_quality_form",
    "away_quality_form",

    # ── GOALS ────────────────────────────────────────────────────────────────
    "home_avg_goal_diff",
    "away_avg_goal_diff",
    "home_avg_goals_conceded",
    "away_avg_goals_conceded",

    # ── xG / SHOTS ───────────────────────────────────────────────────────────
    # real_xg_diff: offensive shot quality gap (attack signal).
    # xga_diff: defensive shot quality gap — rolling xGA allowed difference.
    #   xG and xGA have only -0.27 correlation; they are genuinely independent.
    #   xga_diff captures structural defensive weakness beyond goals conceded.
    "real_xg_diff",
    "xga_diff",
    "sot_diff",

    # ── VENUE FORM ───────────────────────────────────────────────────────────
    "home_venue_form",
    "away_venue_form",
    "home_venue_conceded",
    "away_venue_conceded",

    # ── HEAD-TO-HEAD ─────────────────────────────────────────────────────────
    "h2h_home_win_rate",
    "h2h_draw_rate",

    # ── FATIGUE ──────────────────────────────────────────────────────────────
    "rest_diff",

    # ── MOMENTUM ─────────────────────────────────────────────────────────────
    "streak_diff",

    # ── DRAW-AFFINITY ────────────────────────────────────────────────────────
    # w5 draw rates dropped in v8: noisier version of w10 without independent
    # signal. combined_draw_tendency dropped: linear combination of w10 rates
    # × combined_clean_sheet — triple-counting the same variance.
    "home_draw_rate_w10",
    "away_draw_rate_w10",
    "combined_clean_sheet",
    "combined_low_scoring_run",
    "combined_avg_scored",
    # B365-based draw signal: available all 7 seasons (unlike draw_odds_vs_base
    # which requires AvgD and is NaN for the entire 2018-19 season).
    # [WHY BASE not ODDS?] B365D covers ~99% of rows so zero-impute is correct.
    "b365_draw_indicator",

    # ── CALENDAR ─────────────────────────────────────────────────────────────
    # sin/cos cyclical encoding: Jan and Dec are adjacent in the football
    # calendar but raw int encoding (1, 12) places them far apart.
    "month_sin",
    "month_cos",

    # ── UNDERSTAT PRE-MATCH SIMULATION ───────────────────────────────────────
    # Monte Carlo win/draw/loss probabilities from understat's xG model.
    # 100% coverage. Falls back to uniform prior (0.333/0.270/0.333) for the
    # rare missing rows — neutral prior, not a false signal.
    # [WHY BASE not ODDS?] These are model-derived probabilities, not bookmaker
    # odds. They belong with the form/signal features, not the market features.
    # They are NOT imputed with median — the EPL base rates are a better neutral
    # prior than any median for probability features.
    "us_forecast_w",
    "us_forecast_d",
    "us_forecast_l",
]

ODDS_COLS = [
    # B365 opening implied probabilities — full 7-season coverage.
    # [WHY not avg_prob_*?] avg_prob_* requires AvgH/AvgD/AvgA which are absent
    # from the entire 2018-19 CSV (380 rows NaN, median-imputed = no signal).
    # avg_prob_* also correlates ~0.97 with b365_prob_* — near-duplicate signal
    # that splits importance mass without adding information. Dropped in v8.
    # [WHY not draw_odds_vs_base?] Same AvgD availability problem — NaN for all
    # of 2018-19. b365_draw_indicator in BASE_NUMERIC_COLS already provides the
    # draw signal using B365D which exists in every season.
    "b365_prob_h",
    "b365_prob_d",
    "b365_prob_a",
    # Closing-line movement: opening → closing price ratio.
    # Scale-invariant (ratio not difference). Encodes late-breaking information
    # (team news, injury updates) that rolling stats cannot capture.
    "odds_move_h",
    "odds_move_d",
    "odds_move_a",
]

CAT_COLS = ["HomeTeam", "AwayTeam"]