"""
app.py  —  Premier League Match Outcome Predictor (v2)

Changes from v1:
  - Auto-populates form sliders from processed_matches.csv (latest rolling stats)
  - Accepts and displays all new features (Elo, shots, corners, etc.)
  - Shows Elo ratings for each selected team
  - Handles the new LabelEncoder so predictions decode to H/D/A correctly
  - Shows model comparison results from training
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_PATH        = Path("model.pkl")
LABEL_ENC_PATH    = Path("label_encoder.pkl")
FEATURES_PATH     = Path("feature_columns.pkl")
TEAMS_PATH        = Path("teams.pkl")
MATCHES_PATH      = Path("processed_matches.csv")
RESULTS_PATH      = Path("model_results.pkl")

st.set_page_config(
    page_title="EPL Match Predictor v2",
    page_icon="⚽",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Load all artifacts
# [WHY] @st.cache_resource caches the loaded objects across user sessions.
#        Without it, every widget interaction re-loads the pickle files from
#        disk — very slow. cache_resource is for heavy objects (models, data).
#        cache_data is for pure functions that return DataFrames.
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model_artifacts():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(LABEL_ENC_PATH, "rb") as f:
        label_enc = pickle.load(f)
    with open(FEATURES_PATH, "rb") as f:
        feature_meta = pickle.load(f)
    with open(TEAMS_PATH, "rb") as f:
        teams = pickle.load(f)
    return model, label_enc, feature_meta, teams


@st.cache_data
def load_processed_matches() -> pd.DataFrame:
    """
    [WHY] cache_data is used here (not cache_resource) because this function
    returns a DataFrame — a pure data object that Streamlit can hash and
    invalidate automatically if the file changes.
    """
    if MATCHES_PATH.exists():
        df = pd.read_csv(MATCHES_PATH, parse_dates=["Date"])
        return df
    return pd.DataFrame()


@st.cache_data
def load_model_results() -> dict:
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, "rb") as f:
            return pickle.load(f)
    return {}


def get_latest_team_stats(df: pd.DataFrame, team: str, venue: str) -> dict:
    """
    Pull the most recent row for a team playing at home or away.
    Used to auto-populate slider defaults.

    [WHY] We filter by venue (home/away) because home_* columns reflect
    the HOME team's stats in that row, and away_* reflect the AWAY team's.
    We always want the most recent game that has the stats we need.
    """
    if df.empty:
        return {}

    if venue == "home":
        mask = df["HomeTeam"] == team
        prefix = "home_"
    else:
        mask = df["AwayTeam"] == team
        prefix = "away_"

    rows = df[mask].sort_values("Date")
    if rows.empty:
        return {}

    last = rows.iloc[-1]

    def safe(col, fallback):
        v = last.get(col, fallback)
        return float(v) if pd.notna(v) else fallback

    return {
        "form_points":    safe(f"{prefix}form_points", 7),
        "avg_scored":     safe(f"{prefix}avg_goals_scored", 1.3),
        "avg_conceded":   safe(f"{prefix}avg_goals_conceded", 1.2),
        "avg_shots":      safe(f"{prefix}avg_shots", 12.0),
        "avg_sot":        safe(f"{prefix}avg_sot", 4.5),
        "avg_corners":    safe(f"{prefix}avg_corners", 5.0),
        "avg_fouls":      safe(f"{prefix}avg_fouls", 11.0),
        "avg_yellows":    safe(f"{prefix}avg_yellows", 1.5),
        "elo":            safe(f"{prefix}elo", 1500.0),
        "venue_form":     safe(f"{prefix}venue_form", 6),
        "venue_scored":   safe(f"{prefix}venue_scored", 1.3),
        "venue_conceded": safe(f"{prefix}venue_conceded", 1.2),
        "clean_sheet_pct":safe(f"{prefix}clean_sheet_pct", 0.2),
        "scoring_pct":    safe(f"{prefix}scoring_pct", 0.7),
        "avg_goal_diff":  safe(f"{prefix}avg_goal_diff", 0.0),
    }


def build_input_row(params: dict, feature_meta: dict) -> pd.DataFrame:
    """
    Constructs the exact feature row the trained pipeline expects.
    All columns must match what train_model.py produced — in the same order.

    [WHY] We build the row as a dict and wrap it in pd.DataFrame([...])
    so the pipeline's ColumnTransformer can select columns by name.
    Column ORDER doesn't matter for ColumnTransformer (it selects by name),
    but all required column NAMES must be present.
    """
    h = params["home"]
    a = params["away"]
    use_odds = feature_meta.get("use_betting_odds", False)

    row = {
        # Overall form
        "home_form_points":        h["form_points"],
        "away_form_points":        a["form_points"],
        "form_diff":               h["form_points"] - a["form_points"],

        # Goals
        "home_avg_goals_scored":   h["avg_scored"],
        "away_avg_goals_scored":   a["avg_scored"],
        "home_avg_goals_conceded": h["avg_conceded"],
        "away_avg_goals_conceded": a["avg_conceded"],
        "goal_scored_diff":        h["avg_scored"] - a["avg_scored"],
        "goal_conceded_diff":      h["avg_conceded"] - a["avg_conceded"],
        "home_avg_goal_diff":      h["avg_goal_diff"],
        "away_avg_goal_diff":      a["avg_goal_diff"],

        # Shots
        "home_avg_shots":          h["avg_shots"],
        "away_avg_shots":          a["avg_shots"],
        "home_avg_sot":            h["avg_sot"],
        "away_avg_sot":            a["avg_sot"],
        "sot_diff":                h["avg_sot"] - a["avg_sot"],

        # Set pieces & discipline
        "home_avg_corners":        h["avg_corners"],
        "away_avg_corners":        a["avg_corners"],
        "home_avg_fouls":          h["avg_fouls"],
        "away_avg_fouls":          a["avg_fouls"],
        "home_avg_yellows":        h["avg_yellows"],
        "away_avg_yellows":        a["avg_yellows"],
        "home_avg_reds":           0.05,
        "away_avg_reds":           0.05,

        # Momentum
        "home_clean_sheet_pct":    h["clean_sheet_pct"],
        "away_clean_sheet_pct":    a["clean_sheet_pct"],
        "home_scoring_pct":        h["scoring_pct"],
        "away_scoring_pct":        a["scoring_pct"],

        # Venue-specific form
        "home_venue_form":         h["venue_form"],
        "away_venue_form":         a["venue_form"],
        "home_venue_scored":       h["venue_scored"],
        "away_venue_scored":       a["venue_scored"],
        "home_venue_conceded":     h["venue_conceded"],
        "away_venue_conceded":     a["venue_conceded"],

        # Elo
        "home_elo":                h["elo"],
        "away_elo":                a["elo"],
        "elo_diff":                h["elo"] - a["elo"],

        # Head-to-head (user can't easily provide this; use neutral 0.5)
        "h2h_home_win_rate":       params.get("h2h", 0.5),

        # Rest (default: 7 days each — typical mid-week gap)
        "home_rest_days":          params.get("home_rest", 7),
        "away_rest_days":          params.get("away_rest", 7),
        "rest_diff":               params.get("home_rest", 7) - params.get("away_rest", 7),

        "is_weekend":              params.get("is_weekend", 1),

        # Team identity (for OneHotEncoder)
        "HomeTeam":                params["home_team"],
        "AwayTeam":                params["away_team"],
    }

    # Betting odds columns — filled with NaN if not provided;
    # SimpleImputer in the pipeline will fill with 0 (start-of-season default)
    if use_odds:
        row["b365_prob_h"] = params.get("b365_prob_h", np.nan)
        row["b365_prob_d"] = params.get("b365_prob_d", np.nan)
        row["b365_prob_a"] = params.get("b365_prob_a", np.nan)
        row["avg_prob_h"]  = params.get("avg_prob_h",  np.nan)
        row["avg_prob_d"]  = params.get("avg_prob_d",  np.nan)
        row["avg_prob_a"]  = params.get("avg_prob_a",  np.nan)

    return pd.DataFrame([row])


# ===========================================================================
# UI
# ===========================================================================

# --- Header -----------------------------------------------------------------
st.title("⚽ Premier League Match Outcome Predictor")
st.caption("v2 — Elo ratings · Shot & corner stats · Venue-specific form · Gradient Boosted Trees")

if not MODEL_PATH.exists():
    st.error("model.pkl not found. Run `python train_model.py` first.")
    st.stop()

model, label_enc, feature_meta, teams = load_model_artifacts()
matches_df = load_processed_matches()
model_results = load_model_results()
use_odds = feature_meta.get("use_betting_odds", False)

# --- Sidebar: model comparison & config ------------------------------------
with st.sidebar:
    st.header("🏆 Model Results")
    if model_results:
        for name, r in sorted(model_results.items(), key=lambda x: -x[1]["test_acc"]):
            st.metric(
                label=name,
                value=f"{r['test_acc']:.1%} test",
                delta=f"CV {r['cv_mean']:.1%}",
            )
    else:
        st.info("Run train_model.py to populate results.")

    st.divider()
    st.header("⚙️ Settings")
    show_advanced = st.toggle("Show advanced sliders", value=False)
    if use_odds:
        enter_odds = st.toggle("Enter bookmaker odds", value=False)
    else:
        enter_odds = False

# --- Team selection ---------------------------------------------------------
col_l, col_r = st.columns(2)
with col_l:
    home_team = st.selectbox("🏠 Home team", teams, index=0)
with col_r:
    away_options = [t for t in teams if t != home_team]
    away_team = st.selectbox("✈️ Away team", away_options, index=0)

# Auto-load latest stats for selected teams
h_stats = get_latest_team_stats(matches_df, home_team, "home")
a_stats = get_latest_team_stats(matches_df, away_team, "away")

# Show Elo ratings as a quick strength indicator
if h_stats and a_stats:
    elo_col1, elo_col2, elo_col3 = st.columns(3)
    with elo_col1:
        st.metric(f"{home_team} Elo", f"{h_stats['elo']:.0f}")
    with elo_col2:
        diff = h_stats["elo"] - a_stats["elo"]
        st.metric("Elo advantage", f"{abs(diff):.0f} pts",
                  delta=f"{'Home' if diff >= 0 else 'Away'} favoured")
    with elo_col3:
        st.metric(f"{away_team} Elo", f"{a_stats['elo']:.0f}")

st.divider()

# --- Form inputs ------------------------------------------------------------
st.subheader("📊 Recent form (last 5 matches) — auto-populated, adjust if needed")

c1, c2 = st.columns(2)

with c1:
    st.markdown(f"**{home_team} (Home)**")
    home_form     = st.slider("Form points",         0, 15,  int(h_stats.get("form_points", 7)),   key="hfp")
    home_scored   = st.slider("Avg goals scored",    0.0, 5.0, round(h_stats.get("avg_scored", 1.3), 1), 0.1, key="hgs")
    home_conceded = st.slider("Avg goals conceded",  0.0, 5.0, round(h_stats.get("avg_conceded", 1.2), 1), 0.1, key="hgc")
    home_sot      = st.slider("Avg shots on target", 0.0, 12.0, round(h_stats.get("avg_sot", 4.5), 1), 0.1, key="hsot")

with c2:
    st.markdown(f"**{away_team} (Away)**")
    away_form     = st.slider("Form points",         0, 15,  int(a_stats.get("form_points", 7)),   key="afp")
    away_scored   = st.slider("Avg goals scored",    0.0, 5.0, round(a_stats.get("avg_scored", 1.3), 1), 0.1, key="ags")
    away_conceded = st.slider("Avg goals conceded",  0.0, 5.0, round(a_stats.get("avg_conceded", 1.2), 1), 0.1, key="agc")
    away_sot      = st.slider("Avg shots on target", 0.0, 12.0, round(a_stats.get("avg_sot", 4.5), 1), 0.1, key="asot")

# --- Advanced sliders (hidden by default) -----------------------------------
if show_advanced:
    st.subheader("🔬 Advanced features")
    ac1, ac2 = st.columns(2)
    with ac1:
        home_corners = st.slider("Home avg corners",  0.0, 12.0, round(h_stats.get("avg_corners", 5.0), 1), 0.1)
        home_fouls   = st.slider("Home avg fouls",    0.0, 20.0, round(h_stats.get("avg_fouls", 11.0), 1), 0.5)
        home_yellows = st.slider("Home avg yellows",  0.0, 5.0,  round(h_stats.get("avg_yellows", 1.5), 1), 0.1)
        home_rest    = st.slider("Home rest days",    1, 21, 7)
    with ac2:
        away_corners = st.slider("Away avg corners",  0.0, 12.0, round(a_stats.get("avg_corners", 5.0), 1), 0.1)
        away_fouls   = st.slider("Away avg fouls",    0.0, 20.0, round(a_stats.get("avg_fouls", 11.0), 1), 0.5)
        away_yellows = st.slider("Away avg yellows",  0.0, 5.0,  round(a_stats.get("avg_yellows", 1.5), 1), 0.1)
        away_rest    = st.slider("Away rest days",    1, 21, 7)
    h2h_rate = st.slider("H2H home win rate (last 6 meetings)", 0.0, 1.0, 0.5, 0.05)
    is_weekend = st.toggle("Weekend fixture?", value=True)
else:
    home_corners = h_stats.get("avg_corners", 5.0)
    home_fouls   = h_stats.get("avg_fouls", 11.0)
    home_yellows = h_stats.get("avg_yellows", 1.5)
    away_corners = a_stats.get("avg_corners", 5.0)
    away_fouls   = a_stats.get("avg_fouls", 11.0)
    away_yellows = a_stats.get("avg_yellows", 1.5)
    home_rest    = 7
    away_rest    = 7
    h2h_rate     = 0.5
    is_weekend   = True

# --- Bookmaker odds (optional) ----------------------------------------------
b365_probs = {}
if enter_odds and use_odds:
    st.subheader("📈 Bookmaker odds (optional)")
    st.caption("Enter decimal odds (e.g. 2.10). Leave at 0 to skip.")
    oc1, oc2, oc3 = st.columns(3)
    with oc1:
        b365h = st.number_input("Bet365 Home odds", 1.01, 50.0, 0.0)
    with oc2:
        b365d = st.number_input("Bet365 Draw odds", 1.01, 50.0, 0.0)
    with oc3:
        b365a = st.number_input("Bet365 Away odds", 1.01, 50.0, 0.0)

    if b365h > 1 and b365d > 1 and b365a > 1:
        raw = [1/b365h, 1/b365d, 1/b365a]
        total = sum(raw)
        b365_probs = {
            "b365_prob_h": raw[0] / total,
            "b365_prob_d": raw[1] / total,
            "b365_prob_a": raw[2] / total,
            "avg_prob_h":  raw[0] / total,
            "avg_prob_d":  raw[1] / total,
            "avg_prob_a":  raw[2] / total,
        }

# --- Predict ----------------------------------------------------------------
st.divider()
if st.button("🔮 Predict outcome", type="primary", use_container_width=True):

    params = {
        "home_team": home_team,
        "away_team": away_team,
        "home": {
            "form_points":    home_form,
            "avg_scored":     home_scored,
            "avg_conceded":   home_conceded,
            "avg_shots":      h_stats.get("avg_shots", 12.0),
            "avg_sot":        home_sot,
            "avg_corners":    home_corners,
            "avg_fouls":      home_fouls,
            "avg_yellows":    home_yellows,
            "elo":            h_stats.get("elo", 1500.0),
            "venue_form":     h_stats.get("venue_form", home_form),
            "venue_scored":   h_stats.get("venue_scored", home_scored),
            "venue_conceded": h_stats.get("venue_conceded", home_conceded),
            "clean_sheet_pct":h_stats.get("clean_sheet_pct", 0.2),
            "scoring_pct":    h_stats.get("scoring_pct", 0.7),
            "avg_goal_diff":  h_stats.get("avg_goal_diff", 0.0),
        },
        "away": {
            "form_points":    away_form,
            "avg_scored":     away_scored,
            "avg_conceded":   away_conceded,
            "avg_shots":      a_stats.get("avg_shots", 12.0),
            "avg_sot":        away_sot,
            "avg_corners":    away_corners,
            "avg_fouls":      away_fouls,
            "avg_yellows":    away_yellows,
            "elo":            a_stats.get("elo", 1500.0),
            "venue_form":     a_stats.get("venue_form", away_form),
            "venue_scored":   a_stats.get("venue_scored", away_scored),
            "venue_conceded": a_stats.get("venue_conceded", away_conceded),
            "clean_sheet_pct":a_stats.get("clean_sheet_pct", 0.2),
            "scoring_pct":    a_stats.get("scoring_pct", 0.7),
            "avg_goal_diff":  a_stats.get("avg_goal_diff", 0.0),
        },
        "h2h":       h2h_rate,
        "home_rest": home_rest,
        "away_rest": away_rest,
        "is_weekend": int(is_weekend),
        **b365_probs,
    }

    X = build_input_row(params, feature_meta)

    # [WHY] We call label_enc.inverse_transform() on the integer prediction
    #        to get back 'H', 'D', or 'A'. This is the reverse of what
    #        train_model.py did with label_enc.fit_transform(y).
    pred_int  = model.predict(X)[0]
    pred_str  = label_enc.inverse_transform([pred_int])[0]
    probs     = model.predict_proba(X)[0]
    int_labels = list(model.classes_)
    str_labels = list(label_enc.inverse_transform(int_labels))

    label_map = {"H": "🏠 Home Win", "D": "🤝 Draw", "A": "✈️ Away Win"}

    outcome = label_map.get(pred_str, pred_str)
    st.success(f"### Prediction: {outcome}")
    st.caption(f"{home_team} vs {away_team}")

    prob_df = pd.DataFrame({
        "Outcome": [label_map.get(lbl, lbl) for lbl in str_labels],
        "Probability": [round(p, 4) for p in probs],
    }).sort_values("Probability", ascending=False)

    st.dataframe(prob_df, hide_index=True, use_container_width=True)
    st.bar_chart(prob_df.set_index("Outcome"))

    with st.expander("🔍 Feature values sent to model"):
        st.dataframe(X.T.rename(columns={0: "value"}), use_container_width=True)

st.divider()
st.markdown("""
**Instructor notes:**  
This v2 model uses Elo ratings, shot quality, venue-specific form, head-to-head history,
rest days, discipline stats, and optionally bookmaker implied probabilities — all computed
with strict temporal ordering to prevent data leakage. Trained with TimeSeriesSplit CV and
compared across XGBoost, Gradient Boosted Trees, and SVM (RBF kernel).
""")