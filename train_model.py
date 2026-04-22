from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_DIR = Path("data")

REQUIRED_COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]


def load_csvs():
    files = sorted(DATA_DIR.glob("*.csv"))

    if not files:
        raise FileNotFoundError("No CSV files found in ./data. Put your EPL season CSVs there.")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df["source_file"] = f.name
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def normalize_dates(df):
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTR"])
    return df.sort_values("Date").reset_index(drop=True)


def add_team_form_features(df, window=5):
    team_stats = {}
    rows = []

    for _, row in df.iterrows():
        home = row["HomeTeam"]
        away = row["AwayTeam"]

        def recent(team):
            history = team_stats.get(team, [])
            last = history[-window:]

            if not last:
                return {
                    "form_points": 0,
                    "avg_scored": 0.0,
                    "avg_conceded": 0.0,
                }

            return {
                "form_points": sum(x["points"] for x in last),
                "avg_scored": np.mean([x["scored"] for x in last]),
                "avg_conceded": np.mean([x["conceded"] for x in last]),
            }

        home_recent = recent(home)
        away_recent = recent(away)

        rows.append(
            {
                "Date": row["Date"],
                "HomeTeam": home,
                "AwayTeam": away,
                "FTR": row["FTR"],
                "home_form_points": home_recent["form_points"],
                "away_form_points": away_recent["form_points"],
                "form_diff": home_recent["form_points"] - away_recent["form_points"],
                "home_avg_goals_scored": home_recent["avg_scored"],
                "away_avg_goals_scored": away_recent["avg_scored"],
                "home_avg_goals_conceded": home_recent["avg_conceded"],
                "away_avg_goals_conceded": away_recent["avg_conceded"],
                "goal_scored_diff": home_recent["avg_scored"] - away_recent["avg_scored"],
                "goal_conceded_diff": home_recent["avg_conceded"] - away_recent["avg_conceded"],
                "is_weekend": 1 if row["Date"].dayofweek >= 5 else 0,
            }
        )

        home_points = 3 if row["FTR"] == "H" else 1 if row["FTR"] == "D" else 0
        away_points = 3 if row["FTR"] == "A" else 1 if row["FTR"] == "D" else 0

        team_stats.setdefault(home, []).append(
            {
                "points": home_points,
                "scored": row["FTHG"],
                "conceded": row["FTAG"],
            }
        )

        team_stats.setdefault(away, []).append(
            {
                "points": away_points,
                "scored": row["FTAG"],
                "conceded": row["FTHG"],
            }
        )

    return pd.DataFrame(rows)


def main():
    raw = load_csvs()

    missing = [col for col in REQUIRED_COLUMNS if col not in raw.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = normalize_dates(raw)
    feat_df = add_team_form_features(df)

    feature_cols_num = [
        "home_form_points",
        "away_form_points",
        "form_diff",
        "home_avg_goals_scored",
        "away_avg_goals_scored",
        "home_avg_goals_conceded",
        "away_avg_goals_conceded",
        "goal_scored_diff",
        "goal_conceded_diff",
        "is_weekend",
    ]

    feature_cols_cat = ["HomeTeam", "AwayTeam"]

    X = feat_df[feature_cols_num + feature_cols_cat]
    y = feat_df["FTR"]

    split_idx = int(len(feat_df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                feature_cols_num,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                feature_cols_cat,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
              LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                )
            ),
        ]
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("Test accuracy:", round(accuracy_score(y_test, preds), 4))
    print("\nClassification report:\n")
    print(classification_report(y_test, preds))

    ohe = model.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
    home_teams = list(ohe.categories_[0])
    away_teams = list(ohe.categories_[1])

    ordered_columns = (
        feature_cols_num
        + [f"home_team_{t}" for t in home_teams]
        + [f"away_team_{t}" for t in away_teams]
    )

    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("feature_columns.pkl", "wb") as f:
        pickle.dump(
            {
                "ordered_columns": ordered_columns,
                "home_team_dummies": home_teams,
                "away_team_dummies": away_teams,
            },
            f,
        )

    teams = sorted(set(feat_df["HomeTeam"]).union(set(feat_df["AwayTeam"])))
    with open("teams.pkl", "wb") as f:
        pickle.dump(teams, f)

    feat_df.to_csv("processed_matches.csv", index=False)
    print("\nSaved: model.pkl, feature_columns.pkl, teams.pkl, processed_matches.csv")


if __name__ == "__main__":
    main()