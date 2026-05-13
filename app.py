"""
app.py — EPL Match Outcome Predictor (v3)

Updated for v9 feature set:
  - build_input_row now matches the exact 40-column feature set trained in v9
  - get_latest_team_stats reads the correct column names from processed_matches.csv
  - Sidebar shows CV f1-macro (not test_acc) as primary metric, consistent with
    how the best model is selected in train_model.py
  - Bookmaker odds section updated: avg_prob_* removed (dropped in v8),
    odds_move columns added with sensible UI
  - us_forecast_w/d/l exposed as optional Understat simulation inputs
  - Footer updated to reflect current model architecture
"""

import pickle
from pathlib import Path
import datetime

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Paths — all artifacts written to the same directory as app.py
# ---------------------------------------------------------------------------
MODEL_PATH    = Path("model.pkl")
LABEL_ENC_PATH= Path("label_encoder.pkl")
FEATURES_PATH = Path("feature_columns.pkl")
TEAMS_PATH    = Path("teams.pkl")
MATCHES_PATH  = Path("processed_matches.csv")
RESULTS_PATH  = Path("model_results.pkl")

st.set_page_config(
    page_title="EPL Match Predictor",
    page_icon="⚽",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Artifact loaders
# [WHY cache_resource] Heavy objects (model, encoder) — cached across sessions.
# [WHY cache_data] DataFrames — Streamlit can hash and invalidate automatically.
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model_artifacts():
    with open(MODEL_PATH,     "rb") as f: model       = pickle.load(f)
    with open(LABEL_ENC_PATH, "rb") as f: label_enc   = pickle.load(f)
    with open(FEATURES_PATH,  "rb") as f: feature_meta= pickle.load(f)
    with open(TEAMS_PATH,     "rb") as f: teams        = pickle.load(f)
    return model, label_enc, feature_meta, teams

@st.cache_data
def load_processed_matches() -> pd.DataFrame:
    if MATCHES_PATH.exists():
        return pd.read_csv(MATCHES_PATH, parse_dates=["Date"])
    return pd.DataFrame()

@st.cache_data
def load_model_results() -> dict:
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, "rb") as f:
            return pickle.load(f)
    return {}


# ---------------------------------------------------------------------------
# Auto-populate: pull the latest pre-match feature values for a team
# from processed_matches.csv so the user doesn't have to type everything.
#
# [WHY separate home/away lookup?]
# The feature columns are asymmetric — "home_avg_goal_diff" is the home
# team's rolling stat when they played at home, "away_avg_goal_diff" is the
# away team's stat when they played away. We need the right prefix to get
# the right rolling window.
# ---------------------------------------------------------------------------
def get_latest_team_stats(df: pd.DataFrame, team: str, venue: str) -> dict:
    if df.empty:
        return {}

    if venue == "home":
        mask   = df["HomeTeam"] == team
        prefix = "home_"
    else:
        mask   = df["AwayTeam"] == team
        prefix = "away_"

    rows = df[mask].sort_values("Date")
    if rows.empty:
        return {}

    last = rows.iloc[-1]

    def safe(col, fallback):
        v = last.get(col, fallback)
        return float(v) if pd.notna(v) else fallback

    return {
        # Form / quality
        "quality_form":       safe(f"{prefix}quality_form",       7.0),
        "avg_goal_diff":      safe(f"{prefix}avg_goal_diff",       0.0),
        "avg_goals_conceded": safe(f"{prefix}avg_goals_conceded",  1.2),
        # Venue-specific
        "venue_form":         safe(f"{prefix}venue_form",          6.0),
        "venue_conceded":     safe(f"{prefix}venue_conceded",      1.2),
        # Draw affinity
        "draw_rate_w10":      safe(f"{prefix}draw_rate_w10",       0.25),
        # Elo (stored as elo_diff/venue_elo_diff on the row, not raw Elo)
        # We reconstruct approximate raw Elos from the diff when possible.
        # For display only — the model uses the diff directly.
        "elo_diff":           safe("elo_diff",                     0.0),
        "venue_elo_diff":     safe("venue_elo_diff",               0.0),
        # Understat sim (from most recent match this team was in)
        "us_forecast_w":      safe("us_forecast_w",                0.333),
        "us_forecast_d":      safe("us_forecast_d",                0.270),
        "us_forecast_l":      safe("us_forecast_l",                0.333),
        # Odds (opening)
        "b365_prob_h":        safe("b365_prob_h",                  np.nan),
        "b365_prob_d":        safe("b365_prob_d",                  np.nan),
        "b365_prob_a":        safe("b365_prob_a",                  np.nan),
    }


# ---------------------------------------------------------------------------
# Build the exact feature row the trained pipeline expects.
#
# CRITICAL: every column name here must exactly match what feature_engineering
# v9 writes into BASE_NUMERIC_COLS + ODDS_COLS + CAT_COLS.
# The ColumnTransformer selects by name — extra columns are silently dropped
# (remainder="drop"), but MISSING columns raise a KeyError.
# ---------------------------------------------------------------------------
def build_input_row(params: dict, feature_meta: dict) -> pd.DataFrame:
    h        = params["home"]
    a        = params["away"]
    use_odds = feature_meta.get("use_betting_odds", False)
    match_dt = params.get("match_date", datetime.date.today())

    # Cyclical month encoding — same formula as feature_engineering.py
    month      = match_dt.month
    month_sin  = np.sin(2 * np.pi * month / 12)
    month_cos  = np.cos(2 * np.pi * month / 12)

    row = {
        # ── ELO ──────────────────────────────────────────────────────────────
        # elo_diff and venue_elo_diff come directly from the latest match row.
        # elo_gap = abs(elo_diff) — captures match closeness for draw signal.
        "elo_diff":               params["elo_diff"],
        "venue_elo_diff":         params["venue_elo_diff"],
        "elo_gap":                abs(params["elo_diff"]),

        # ── SEASON PRESSURE ───────────────────────────────────────────────────
        "season_pts_diff":        params.get("season_pts_diff", 0.0),

        # ── QUALITY FORM ──────────────────────────────────────────────────────
        "home_quality_form":      h["quality_form"],
        "away_quality_form":      a["quality_form"],

        # ── GOALS ─────────────────────────────────────────────────────────────
        "home_avg_goal_diff":     h["avg_goal_diff"],
        "away_avg_goal_diff":     a["avg_goal_diff"],
        "home_avg_goals_conceded":h["avg_goals_conceded"],
        "away_avg_goals_conceded":a["avg_goals_conceded"],

        # ── xG / SHOTS ────────────────────────────────────────────────────────
        # real_xg_diff and xga_diff: user supplies rolling averages directly.
        # Defaults to 0 (neutral) when no understat data is available.
        "real_xg_diff":           params.get("real_xg_diff",  0.0),
        "xga_diff":               params.get("xga_diff",      0.0),
        "sot_diff":               params.get("sot_diff",      0.0),

        # ── VENUE FORM ────────────────────────────────────────────────────────
        "home_venue_form":        h["venue_form"],
        "away_venue_form":        a["venue_form"],
        "home_venue_conceded":    h["venue_conceded"],
        "away_venue_conceded":    a["venue_conceded"],

        # ── HEAD-TO-HEAD ──────────────────────────────────────────────────────
        "h2h_home_win_rate":      params.get("h2h_home_win_rate", 0.50),
        "h2h_draw_rate":          params.get("h2h_draw_rate",     0.27),

        # ── FATIGUE ───────────────────────────────────────────────────────────
        "rest_diff":              params.get("rest_diff", 0),

        # ── MOMENTUM ─────────────────────────────────────────────────────────
        "streak_diff":            params.get("streak_diff", 0),

        # ── DRAW-AFFINITY ─────────────────────────────────────────────────────
        "home_draw_rate_w10":     h["draw_rate_w10"],
        "away_draw_rate_w10":     a["draw_rate_w10"],
        "combined_clean_sheet":   params.get("combined_clean_sheet",    0.4),
        "combined_low_scoring_run": params.get("combined_low_scoring", 0.3),
        "combined_avg_scored":    params.get("combined_avg_scored",     2.5),
        # b365_draw_indicator: implied draw prob minus EPL base rate (0.267).
        # Uses b365_prob_d if provided, else 0 (= no signal).
        "b365_draw_indicator":    params.get("b365_draw_indicator",     0.0),

        # ── CALENDAR ──────────────────────────────────────────────────────────
        "month_sin":              month_sin,
        "month_cos":              month_cos,

        # ── UNDERSTAT SIMULATION ──────────────────────────────────────────────
        "us_forecast_w":          params.get("us_forecast_w", 0.333),
        "us_forecast_d":          params.get("us_forecast_d", 0.270),
        "us_forecast_l":          params.get("us_forecast_l", 0.333),

        # ── TEAM IDENTITY (OneHotEncoder) ─────────────────────────────────────
        "HomeTeam":               params["home_team"],
        "AwayTeam":               params["away_team"],
    }

    # Betting odds — only included when USE_BETTING_ODDS=True in training
    if use_odds:
        row["b365_prob_h"] = params.get("b365_prob_h", np.nan)
        row["b365_prob_d"] = params.get("b365_prob_d", np.nan)
        row["b365_prob_a"] = params.get("b365_prob_a", np.nan)
        # odds_move = opening/closing ratio. 1.0 = no movement (safe default).
        row["odds_move_h"] = params.get("odds_move_h", np.nan)
        row["odds_move_d"] = params.get("odds_move_d", np.nan)
        row["odds_move_a"] = params.get("odds_move_a", np.nan)

    return pd.DataFrame([row])


# ===========================================================================
# UI
# ===========================================================================

st.title("⚽ Premier League Match Outcome Predictor")
st.caption("v3 · Elo ratings · xG & xGA · Draw-affinity cluster · Understat simulation · RandomForest + XGBoost")

if not MODEL_PATH.exists():
    st.error("❌ model.pkl not found. Run `python train_model.py` first, then copy all .pkl files here.")
    st.stop()

model, label_enc, feature_meta, teams = load_model_artifacts()
matches_df   = load_processed_matches()
model_results= load_model_results()
use_odds     = feature_meta.get("use_betting_odds", False)

# ---------------------------------------------------------------------------
# Sidebar — model comparison
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("🏆 Model Results")
    if model_results:
        # Sort by CV mean — same criterion used for model selection
        for name, r in sorted(model_results.items(), key=lambda x: -x[1]["cv_mean"]):
            st.metric(
                label=name,
                value=f"CV {r['cv_mean']:.1%}",
                delta=f"Test {r['test_acc']:.1%}",
            )
    else:
        st.info("Run train_model.py to populate results.")

    st.divider()
    st.header("⚙️  Settings")
    show_advanced    = st.toggle("Show advanced inputs",      value=False)
    show_understat   = st.toggle("Enter Understat forecasts", value=False)
    if use_odds:
        enter_odds   = st.toggle("Enter bookmaker odds",      value=False)
    else:
        enter_odds   = False

# ---------------------------------------------------------------------------
# Team selection
# ---------------------------------------------------------------------------
col_l, col_r = st.columns(2)
with col_l:
    home_team = st.selectbox("🏠 Home team", teams, index=0)
with col_r:
    away_options = [t for t in teams if t != home_team]
    away_team    = st.selectbox("✈️ Away team", away_options, index=0)

# Match date — needed for cyclical month encoding
match_date = st.date_input("📅 Match date", value=datetime.date.today())

# Auto-load latest rolling stats for selected teams
h_stats = get_latest_team_stats(matches_df, home_team, "home")
a_stats = get_latest_team_stats(matches_df, away_team, "away")

# Elo display — approximate raw Elos from the stored diff
# elo_diff = home_elo - away_elo, assuming average ≈ 1500
avg_elo   = 1500.0
elo_diff  = h_stats.get("elo_diff", 0.0)
home_elo_approx = avg_elo + elo_diff / 2
away_elo_approx = avg_elo - elo_diff / 2

e1, e2, e3 = st.columns(3)
with e1:
    st.metric(f"{home_team} Elo (approx)", f"{home_elo_approx:.0f}")
with e2:
    st.metric("Elo gap", f"{abs(elo_diff):.0f} pts",
              delta=f"{'Home' if elo_diff >= 0 else 'Away'} favoured")
with e3:
    st.metric(f"{away_team} Elo (approx)", f"{away_elo_approx:.0f}")

st.divider()

# ---------------------------------------------------------------------------
# Main form inputs — auto-populated from processed_matches.csv
# ---------------------------------------------------------------------------
st.subheader("📊 Rolling form — auto-populated from last match, adjust if needed")

c1, c2 = st.columns(2)

with c1:
    st.markdown(f"**{home_team} (Home)**")
    home_quality_form = st.slider(
        "Quality form (opponent-weighted pts, last 5)",
        0.0, 20.0,
        round(h_stats.get("quality_form", 7.0), 1), 0.5, key="hqf"
    )
    home_goal_diff = st.slider(
        "Avg goal diff per game",
        -3.0, 3.0,
        round(h_stats.get("avg_goal_diff", 0.0), 2), 0.05, key="hgd"
    )
    home_conceded = st.slider(
        "Avg goals conceded",
        0.0, 4.0,
        round(h_stats.get("avg_goals_conceded", 1.2), 1), 0.1, key="hgc"
    )
    home_venue_form = st.slider(
        "Venue form pts (home games only)",
        0, 15,
        int(h_stats.get("venue_form", 6)), key="hvf"
    )
    home_venue_conceded = st.slider(
        "Venue avg goals conceded",
        0.0, 4.0,
        round(h_stats.get("venue_conceded", 1.2), 1), 0.1, key="hvc"
    )
    home_draw_rate = st.slider(
        "Draw rate last 10 games",
        0.0, 1.0,
        round(h_stats.get("draw_rate_w10", 0.25), 2), 0.05, key="hdr"
    )

with c2:
    st.markdown(f"**{away_team} (Away)**")
    away_quality_form = st.slider(
        "Quality form (opponent-weighted pts, last 5)",
        0.0, 20.0,
        round(a_stats.get("quality_form", 7.0), 1), 0.5, key="aqf"
    )
    away_goal_diff = st.slider(
        "Avg goal diff per game",
        -3.0, 3.0,
        round(a_stats.get("avg_goal_diff", 0.0), 2), 0.05, key="agd"
    )
    away_conceded = st.slider(
        "Avg goals conceded",
        0.0, 4.0,
        round(a_stats.get("avg_goals_conceded", 1.2), 1), 0.1, key="agc"
    )
    away_venue_form = st.slider(
        "Venue form pts (away games only)",
        0, 15,
        int(a_stats.get("venue_form", 6)), key="avf"
    )
    away_venue_conceded = st.slider(
        "Venue avg goals conceded",
        0.0, 4.0,
        round(a_stats.get("venue_conceded", 1.2), 1), 0.1, key="avc"
    )
    away_draw_rate = st.slider(
        "Draw rate last 10 games",
        0.0, 1.0,
        round(a_stats.get("draw_rate_w10", 0.25), 2), 0.05, key="adr"
    )

# ---------------------------------------------------------------------------
# Advanced inputs (hidden by default)
# ---------------------------------------------------------------------------
h2h_home_win_rate = 0.50
h2h_draw_rate     = 0.27
rest_diff         = 0
streak_diff       = 0
season_pts_diff   = 0.0
real_xg_diff      = 0.0
xga_diff          = 0.0
sot_diff          = 0.0

if show_advanced:
    st.subheader("🔬 Advanced inputs")
    adv1, adv2 = st.columns(2)
    with adv1:
        h2h_home_win_rate = st.slider("H2H home win rate (last 6 meetings)", 0.0, 1.0, 0.50, 0.05)
        h2h_draw_rate     = st.slider("H2H draw rate (last 6 meetings)",     0.0, 1.0, 0.27, 0.05)
        rest_diff         = st.slider("Rest diff (home days − away days)",  -14, 14, 0)
        streak_diff       = st.slider("Streak diff (home streak − away)",   -5, 5, 0)
    with adv2:
        season_pts_diff   = st.slider("Season pts diff (home − away)",      -60.0, 60.0, 0.0, 1.0)
        real_xg_diff      = st.slider("xG diff (home avg xG − away)",       -2.0, 2.0, 0.0, 0.05)
        xga_diff          = st.slider("xGA diff (home xGA allowed − away)", -2.0, 2.0, 0.0, 0.05)
        sot_diff          = st.slider("Shots on target diff (home − away)", -6.0, 6.0, 0.0, 0.5)

# ---------------------------------------------------------------------------
# Understat simulation inputs (optional)
# These are the three features that drove the biggest accuracy improvement.
# If the user has access to pre-match Understat forecasts they can enter them.
# ---------------------------------------------------------------------------
us_forecast_w = h_stats.get("us_forecast_w", 0.333)
us_forecast_d = h_stats.get("us_forecast_d", 0.270)
us_forecast_l = h_stats.get("us_forecast_l", 0.333)

if show_understat:
    st.subheader("🔮 Understat pre-match simulation")
    st.caption("From understat.com — the probabilities their Monte Carlo model assigns before kick-off. "
               "Auto-populated from the most recent match involving these teams. Adjust if you have the current fixture's forecast.")
    us1, us2, us3 = st.columns(3)
    with us1:
        us_forecast_w = st.number_input("Home win prob", 0.0, 1.0,
                                         round(us_forecast_w, 3), 0.01, key="usw")
    with us2:
        us_forecast_d = st.number_input("Draw prob",     0.0, 1.0,
                                         round(us_forecast_d, 3), 0.01, key="usd")
    with us3:
        us_forecast_l = st.number_input("Away win prob", 0.0, 1.0,
                                         round(us_forecast_l, 3), 0.01, key="usl")
    total_us = us_forecast_w + us_forecast_d + us_forecast_l
    if abs(total_us - 1.0) > 0.05:
        st.warning(f"These three probabilities sum to {total_us:.3f} — they should sum to ~1.0.")

# ---------------------------------------------------------------------------
# Bookmaker odds (optional)
# ---------------------------------------------------------------------------
b365_prob_h = np.nan
b365_prob_d = np.nan
b365_prob_a = np.nan
odds_move_h = np.nan
odds_move_d = np.nan
odds_move_a = np.nan
b365_draw_indicator = 0.0

if enter_odds and use_odds:
    st.subheader("📈 Bookmaker odds")
    st.caption("Enter Bet365 decimal odds. Closing odds are optional but improve accuracy.")

    oc1, oc2, oc3 = st.columns(3)
    with oc1: b365h_open = st.number_input("B365 Home (opening)", 1.01, 50.0, 2.0, key="bho")
    with oc2: b365d_open = st.number_input("B365 Draw (opening)", 1.01, 50.0, 3.5, key="bdo")
    with oc3: b365a_open = st.number_input("B365 Away (opening)", 1.01, 50.0, 4.0, key="bao")

    raw   = [1/b365h_open, 1/b365d_open, 1/b365a_open]
    total = sum(raw)
    b365_prob_h = raw[0] / total
    b365_prob_d = raw[1] / total
    b365_prob_a = raw[2] / total
    b365_draw_indicator = b365_prob_d - 0.267

    st.caption("Closing odds (optional — leave at 0 to skip)")
    cc1, cc2, cc3 = st.columns(3)
    with cc1: b365h_close = st.number_input("B365 Home (closing)", 0.0, 50.0, 0.0, key="bhc")
    with cc2: b365d_close = st.number_input("B365 Draw (closing)", 0.0, 50.0, 0.0, key="bdc")
    with cc3: b365a_close = st.number_input("B365 Away (closing)", 0.0, 50.0, 0.0, key="bac")

    if b365h_close > 1:
        odds_move_h = b365h_open / b365h_close
        odds_move_d = b365d_open / b365d_close if b365d_close > 1 else np.nan
        odds_move_a = b365a_open / b365a_close if b365a_close > 1 else np.nan

# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------
st.divider()
if st.button("🔮 Predict outcome", type="primary", use_container_width=True):

    params = {
        "home_team":   home_team,
        "away_team":   away_team,
        "match_date":  match_date,

        # Elo differentials — taken directly from most recent match row
        "elo_diff":        h_stats.get("elo_diff",       0.0),
        "venue_elo_diff":  h_stats.get("venue_elo_diff", 0.0),

        # Season pressure
        "season_pts_diff": season_pts_diff,

        # Home team rolling stats
        "home": {
            "quality_form":      home_quality_form,
            "avg_goal_diff":     home_goal_diff,
            "avg_goals_conceded":home_conceded,
            "venue_form":        home_venue_form,
            "venue_conceded":    home_venue_conceded,
            "draw_rate_w10":     home_draw_rate,
        },

        # Away team rolling stats
        "away": {
            "quality_form":      away_quality_form,
            "avg_goal_diff":     away_goal_diff,
            "avg_goals_conceded":away_conceded,
            "venue_form":        away_venue_form,
            "venue_conceded":    away_venue_conceded,
            "draw_rate_w10":     away_draw_rate,
        },

        # Advanced
        "h2h_home_win_rate":    h2h_home_win_rate,
        "h2h_draw_rate":        h2h_draw_rate,
        "rest_diff":            rest_diff,
        "streak_diff":          streak_diff,
        "real_xg_diff":         real_xg_diff,
        "xga_diff":             xga_diff,
        "sot_diff":             sot_diff,

        # Draw affinity computed features
        "combined_clean_sheet": (
            (1 - home_conceded / 3) * 0.5 + (1 - away_conceded / 3) * 0.5
        ),
        "combined_low_scoring": home_draw_rate * 0.5 + away_draw_rate * 0.5,
        "combined_avg_scored":  max(0.0, home_goal_diff + 1.3) + max(0.0, away_goal_diff + 1.2),

        # Understat simulation
        "us_forecast_w": us_forecast_w,
        "us_forecast_d": us_forecast_d,
        "us_forecast_l": us_forecast_l,

        # Betting odds
        "b365_draw_indicator": b365_draw_indicator,
        "b365_prob_h":         b365_prob_h,
        "b365_prob_d":         b365_prob_d,
        "b365_prob_a":         b365_prob_a,
        "odds_move_h":         odds_move_h,
        "odds_move_d":         odds_move_d,
        "odds_move_a":         odds_move_a,
    }

    X = build_input_row(params, feature_meta)

    pred_int  = model.predict(X)[0]
    pred_str  = label_enc.inverse_transform([pred_int])[0]
    probs     = model.predict_proba(X)[0]
    int_labels= list(model.classes_)
    str_labels= list(label_enc.inverse_transform(int_labels))

    label_map = {"H": "🏠 Home Win", "D": "🤝 Draw", "A": "✈️ Away Win"}
    outcome   = label_map.get(pred_str, pred_str)

    st.success(f"### Prediction: {outcome}")
    st.caption(f"{home_team} vs {away_team}  ·  {match_date}")

    prob_df = pd.DataFrame({
        "Outcome":     [label_map.get(l, l) for l in str_labels],
        "Probability": [round(p, 4) for p in probs],
    }).sort_values("Probability", ascending=False)

    st.dataframe(prob_df, hide_index=True, use_container_width=True)
    st.bar_chart(prob_df.set_index("Outcome"))

    with st.expander("🔍 Feature values sent to model"):
        st.dataframe(X.T.rename(columns={0: "value"}), use_container_width=True)

st.divider()
st.markdown("""
**About this model:**  
Trained on 7 EPL seasons (2018–2026) · 3,019 matches · 40 features  
Best model: **RandomForest** (CV f1-macro 0.56) selected by TimeSeriesSplit cross-validation  
Key signals: Elo ratings · Bet365 implied probabilities · Understat xG simulation forecasts  
Features reduced from 110 → 40 to prevent curse of dimensionality  
Temporal sample weights (decay=0.998) downweight the COVID-era 2020-21 season
""")