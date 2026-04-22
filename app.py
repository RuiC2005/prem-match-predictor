import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

MODEL_PATH = Path("model.pkl")
FEATURES_PATH = Path("feature_columns.pkl")
TEAMS_PATH = Path("teams.pkl")

st.set_page_config(page_title="Football Match Predictor", page_icon="ronaldo.jpg.webp", layout="centered")


@st.cache_resource
def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(FEATURES_PATH, "rb") as f:
        feature_columns = pickle.load(f)
    with open(TEAMS_PATH, "rb") as f:
        teams = pickle.load(f)
    return model, feature_columns, teams


def build_input_row(
    home_team,
    away_team,
    home_form,
    away_form,
    home_scored,
    away_scored,
    home_conceded,
    away_conceded,
):
    return pd.DataFrame(
        [
            {
                "home_form_points": home_form,
                "away_form_points": away_form,
                "form_diff": home_form - away_form,
                "home_avg_goals_scored": home_scored,
                "away_avg_goals_scored": away_scored,
                "home_avg_goals_conceded": home_conceded,
                "away_avg_goals_conceded": away_conceded,
                "goal_scored_diff": home_scored - away_scored,
                "goal_conceded_diff": home_conceded - away_conceded,
                "is_weekend": 1,
                "HomeTeam": home_team,
                "AwayTeam": away_team,
            }
        ]
    )


col1, col2 = st.columns([3, 5])

with col1:
    st.image("ronaldo.jpg.webp", width=300)

with col2:
    st.title("Premier League Match Outcome Predictor")

st.caption("Basic MVP demo: predicts Home Win / Draw / Away Win from historical EPL data.")

if not MODEL_PATH.exists():
    st.error("model.pkl not found. Run train_model.py first.")
    st.stop()

model, feature_columns, teams = load_artifacts()

home_team = st.selectbox("Home team", teams, index=0)
away_options = [t for t in teams if t != home_team]
away_team = st.selectbox("Away team", away_options, index=0)

st.subheader("Recent form inputs")

col1, col2 = st.columns(2)

with col1:
    home_form = st.slider("Home team form points (last 5 matches)", 0, 15, 8)
    home_scored = st.slider("Home avg goals scored (last 5)", 0.0, 4.0, 1.6, 0.1)
    home_conceded = st.slider("Home avg goals conceded (last 5)", 0.0, 4.0, 1.1, 0.1)

with col2:
    away_form = st.slider("Away team form points (last 5 matches)", 0, 15, 7)
    away_scored = st.slider("Away avg goals scored (last 5)", 0.0, 4.0, 1.3, 0.1)
    away_conceded = st.slider("Away avg goals conceded (last 5)", 0.0, 4.0, 1.4, 0.1)

if st.button("Predict outcome"):
    X = build_input_row(
        home_team,
        away_team,
        home_form,
        away_form,
        home_scored,
        away_scored,
        home_conceded,
        away_conceded,
    )

    pred = model.predict(X)[0]
    probs = model.predict_proba(X)[0]
    labels = list(model.classes_)

    label_map = {"H": "Home Win", "D": "Draw", "A": "Away Win"}

    st.success(f"Prediction: {label_map.get(pred, pred)}")

    prob_df = pd.DataFrame(
        {
            "Outcome": [label_map.get(lbl, lbl) for lbl in labels],
            "Probability": probs,
        }
    ).sort_values("Probability", ascending=False)

    st.dataframe(prob_df, hide_index=True, use_container_width=True)
    st.bar_chart(prob_df.set_index("Outcome"))

st.markdown("---")
st.markdown(
    """
    **Professor-demo talking point:**  
    This MVP uses historical Premier League match data and simple form-based features
    to perform a 3-class classification task: Home Win, Draw, or Away Win.
    """
)