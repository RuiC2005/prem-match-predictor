"""
train_model.py — EPL Match Predictor, Training Pipeline v3

Orchestrates the full pipeline:
  1. Data loading & Elo computation   (data_loader.py)
  2. Advanced feature engineering     (feature_engineering.py)
  3. Model training, CV & comparison  (model_trainer.py)

Run with:
    python train_model.py

Artifacts written to the working directory:
  model.pkl, label_encoder.pkl, feature_columns.pkl,
  teams.pkl, processed_matches.csv, model_results.pkl
"""

import warnings 

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from data_loader import load_and_prepare
from feature_engineering import (
    USE_BETTING_ODDS,
    BASE_NUMERIC_COLS,
    ODDS_COLS,
    CAT_COLS,
    add_features,
)
from model_trainer import (
    get_models,
    train_and_compare,
    print_summary,
    save_artifacts,
)

warnings.filterwarnings("ignore", category=FutureWarning)


def main() -> None:
    print("=" * 60)
    print("EPL Match Predictor — Training Pipeline v3")
    print(f"Betting odds:     {'ON' if USE_BETTING_ODDS else 'OFF'}")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # 1. Load & prepare data
    # -----------------------------------------------------------------------
    df = load_and_prepare()
    print(
        f"\nLoaded {len(df)} matches "
        f"from {df['Date'].min().date()} to {df['Date'].max().date()}"
    )

    # -----------------------------------------------------------------------
    # 2. Feature engineering
    # -----------------------------------------------------------------------
    print("\nEngineering features...")
    feat_df = add_features(df, window=5)
    print(f"Feature matrix shape: {feat_df.shape}")

    # -----------------------------------------------------------------------
    # 3. Build feature column lists
    # -----------------------------------------------------------------------
    numeric_cols = BASE_NUMERIC_COLS[:]
    if USE_BETTING_ODDS:
        numeric_cols += ODDS_COLS

    # Keep only columns that actually exist (some may be missing from older data)
    numeric_cols = [c for c in numeric_cols if c in feat_df.columns]
    cat_cols = CAT_COLS

    X = feat_df[numeric_cols + cat_cols]

    # -----------------------------------------------------------------------
    # 4. Encode target
    # [WHY global fit] Fit the encoder on ALL y before splitting so that all
    #   three classes (A/D/H) are always known — prevents unseen-class errors
    #   if one class is absent from a small fold.
    # -----------------------------------------------------------------------
    label_enc = LabelEncoder()
    y_encoded = label_enc.fit_transform(feat_df["FTR"])  # A=0, D=1, H=2 (alphabetical)
    y = pd.Series(y_encoded, index=feat_df.index)

    # -----------------------------------------------------------------------
    # 5. Temporal train / test split (80/20)
    # [WHY temporal?] We must never train on future matches. Sorting by Date
    #   and splitting at 80% means the model is always tested on "the future".
    # -----------------------------------------------------------------------
    split_idx = int(len(feat_df) * 0.80)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"\nTrain: {len(X_train)} matches | Test: {len(X_test)} matches")
    print(
        f"Test window: {feat_df['Date'].iloc[split_idx].date()} "
        f"to {feat_df['Date'].iloc[-1].date()}"
    )

    # -----------------------------------------------------------------------
    # 6. Train & evaluate all models
    # -----------------------------------------------------------------------
    models = get_models(numeric_cols, cat_cols)
    results = train_and_compare(
        models, X_train, y_train, X_test, y_test,
        label_enc=label_enc,
        n_cv_splits=5,
    )

    # -----------------------------------------------------------------------
    # 7. Pick best model & save artifacts
    # -----------------------------------------------------------------------
    best_name = print_summary(results)
    print(f"\nSaving best model: {best_name}")

    save_artifacts(
        best_pipeline=results[best_name]["pipeline"],
        label_enc=label_enc,
        numeric_cols=numeric_cols,
        cat_cols=cat_cols,
        use_betting_odds=USE_BETTING_ODDS,
        feat_df=feat_df,
        results=results,
    )


if __name__ == "__main__":
    main()