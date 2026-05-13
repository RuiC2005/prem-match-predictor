"""
train_model.py — EPL Match Predictor, Training Pipeline v6

Orchestrates the full pipeline:
  1. Data loading & Elo computation          (data_loader.py)
  2. Advanced feature engineering            (feature_engineering.py)
  3. Combined sample weight generation       (temporal × class-balance)
  4. XGBoost hyperparameter tuning (Optuna)  (model_trainer.tune_xgboost)
  5. Model training, CV & comparison         (model_trainer.py)

Run with:
    python train_model.py

Artifacts written to the working directory:
  model.pkl, label_encoder.pkl, feature_columns.pkl,
  teams.pkl, processed_matches.csv, model_results.pkl

Tuning knobs (change these before running):
  TUNE_XGBOOST   — set True to run Optuna (~5 min, recommended first run)
  OPTUNA_TRIALS  — number of Optuna trials (80 is a good balance)
  WEIGHT_DECAY   — exponential decay per match for temporal weighting
                   0.998 recommended; lower = more aggressive recency bias.
                   Note: decay=0.999 is safe for XGBoost (oldest weight ≈ 9%
                   of newest — well above float underflow).

CHANGES IN v6 (bug fix: XGBoost Home-only collapse)
----------------------------------------------------
Root cause: XGBoost was predicting Home 100% of the time (precision/recall
0.00 for Away and Draw). The temporal sample weights addressed recency bias
but did NOT correct for class imbalance. In the EPL training set, roughly:
  H (Home win) ≈ 45%  |  A (Away win) ≈ 31%  |  D (Draw) ≈ 24%
XGBoost, unlike RandomForest (which uses balanced_subsample), has no built-in
class reweighting. When temporal weights further downweight COVID-era matches
(which skewed toward draws), the effective Draw frequency in weighted training
drops even lower — pushing XGBoost to collapse to the majority class.

Fix: multiply temporal weights by inverse class-frequency weights, then
renormalise. Each match now carries a weight that corrects for BOTH recency
AND class skew simultaneously. A draw match from last season gets upweighted
on BOTH axes; a home-win match from 2018 gets downweighted on both.

[WHY multiply, not add?]
Addition would let a high temporal weight (recent match) overwhelm a high
class weight (rare outcome). Multiplication keeps both corrections active at
every point in the weight distribution. A recent draw gets high-temporal ×
high-class = doubly upweighted — which is exactly what we want.

[WHY not use XGBoost's scale_pos_weight?]
scale_pos_weight is binary-only. For multiclass the correct XGBoost approach
is per-sample weights, which we already support via sample_weight in fit().

CHANGES IN v5 (retained)
--------------------------
1. All-NaN column drop moved BEFORE Optuna tuning.
2. Model selection changed from test_acc → cv_mean.
3. save_artifacts() receives X_train/y_train for calibration only.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

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
    make_sample_weights,
    tune_xgboost,
    train_and_compare,
    print_summary,
    save_artifacts,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Top-level knobs — edit these to control the training run
# ---------------------------------------------------------------------------
TUNE_XGBOOST  = True   # Run Optuna tuning before training (~5 min)
OPTUNA_TRIALS = 80     # Number of Optuna trials (80 gives TPE enough budget for 9-dim space)
# [WHY 0.999 not 0.97?] With 2405 training rows, decay=0.97 drives the oldest
# matches to weight 0.97^2404 ≈ 10^-32, which underflows float64 and crashes
# XGBoost's multiclass solver ("sum_weight >= kRtEps" error).
# At decay=0.999: oldest match = 9% of newest, oldest 500 matches = 6.4% of
# total weight — enough recency bias to downweight COVID era without numerical issues.
# Intuition: a match from 5 seasons ago carries ~83% the weight of today's match.
# That's gentle but meaningful — the COVID season (2020-21) still gets less signal.
WEIGHT_DECAY  = 0.998  # Temporal decay factor per match (0.998–0.9995 recommended)


def main() -> None:
    print("=" * 60)
    print("EPL Match Predictor — Training Pipeline v6")
    print(f"Betting odds:      {'ON' if USE_BETTING_ODDS else 'OFF'}")
    print(f"Temporal weights:  decay={WEIGHT_DECAY}")
    tuning_label = f"ON ({OPTUNA_TRIALS} trials)" if TUNE_XGBOOST else "OFF"
    print(f"Optuna tuning:     {tuning_label}")
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

    # Drop any numeric column that is entirely NaN in X_train.
    # [WHY BEFORE Optuna?] This is the critical ordering fix (v5 bug).
    # ODDS_COLS like avg_prob_* and odds_move_* are all-NaN in the earliest
    # seasons (2018-20 CSVs lack AvgH / closing-line columns). Previously this
    # drop happened AFTER tune_xgboost(), meaning Optuna's early CV folds ran
    # with a different (broken) feature set than the final training run —
    # producing the Optuna CV f1_macro of 0.20 vs final CV of 0.44 mismatch.
    # Dropping ALL-NaN columns here ensures Optuna and final training see an
    # identical, clean feature set on every CV fold.
    all_nan_cols = [c for c in numeric_cols if X_train[c].isna().all()]
    if all_nan_cols:
        print(f"  [warn] Dropping {len(all_nan_cols)} all-NaN train cols: {all_nan_cols}")
        numeric_cols = [c for c in numeric_cols if c not in all_nan_cols]
        X_train = X_train[numeric_cols + cat_cols]
        X_test  = X_test[numeric_cols + cat_cols]
        X       = X[numeric_cols + cat_cols]

    print(f"\nTrain: {len(X_train)} matches | Test: {len(X_test)} matches")
    print(
        f"Test window: {feat_df['Date'].iloc[split_idx].date()} "
        f"to {feat_df['Date'].iloc[-1].date()}"
    )

    # -----------------------------------------------------------------------
    # 6. Combined sample weights: temporal decay × class-balance correction
    #
    # [WHY combine both?]
    # Temporal weights alone downweight old matches but ignore class imbalance
    # (H≈45%, A≈31%, D≈24% in EPL). XGBoost has no built-in class reweighting
    # for multiclass — unlike RandomForest (balanced_subsample). The result was
    # XGBoost collapsing to predict Home 100% of the time.
    #
    # compute_sample_weight('balanced') gives each sample a weight inversely
    # proportional to its class frequency — Draws get ~1.85×, Away ~1.32×,
    # Home ~0.74× (exact values depend on the train split).
    #
    # [WHY multiply not add?]
    # Addition lets a high temporal weight (very recent match) overwhelm class
    # correction. Multiplication keeps both corrections active simultaneously:
    # a recent Draw is doubly upweighted; a 2018 Home win is doubly downweighted.
    # This is what we want — the gradient signal from rare recent outcomes is
    # the most valuable signal in the dataset.
    #
    # [WHY renormalise after multiplying?]
    # XGBoost's multiclass solver requires sum(weights) to be well above the
    # kRtEps floor. Normalising to sum=len(X_train) keeps the effective scale
    # comparable to unweighted training (where each sample implicitly has w=1).
    # -----------------------------------------------------------------------
    print(f"\nGenerating combined sample weights (temporal decay={WEIGHT_DECAY} × class balance)...")
    temporal_w = make_sample_weights(len(X_train), decay=WEIGHT_DECAY)
    balance_w  = compute_sample_weight("balanced", y_train)
    # Normalise balance_w to mean=1 so it acts as a pure multiplier, not a rescaler.
    balance_w  = balance_w / balance_w.mean()
    sample_weights = temporal_w * balance_w
    # Renormalise to sum = len(X_train) — keeps XGBoost's internal scale stable.
    sample_weights = sample_weights / sample_weights.sum() * len(X_train)
    sample_weights = sample_weights.astype(np.float64)

    # Diagnostics: show effective weight by class so the correction is auditable.
    class_names = {v: k for k, v in zip(label_enc.classes_, range(len(label_enc.classes_)))}
    print(f"  Temporal weight ratio newest/oldest: {temporal_w[-1]/temporal_w[0]:.1f}x")
    for cls_int in sorted(class_names):
        mask = (y_train == cls_int).values
        mean_w = sample_weights[mask].mean()
        print(f"  Mean weight [{class_names[cls_int]}]: {mean_w:.3f}  "
              f"(n={mask.sum()}, raw freq={mask.mean():.3f})")

    # -----------------------------------------------------------------------
    # 7. Optuna hyperparameter tuning (XGBoost only)
    # [WHY only XGBoost?] It's consistently our best performer and has the
    #   most impactful hyperparameters for this dataset size. Tuning all
    #   models would take 4x as long with diminishing returns.
    # [WHY after all-NaN drop?] Optuna must see the same feature set as final
    #   training. See the all_nan_cols comment above for the full explanation.
    # -----------------------------------------------------------------------
    xgb_params: dict = {}
    if TUNE_XGBOOST:
        print(f"\nRunning Optuna tuning ({OPTUNA_TRIALS} trials)...")
        xgb_params = tune_xgboost(
            numeric_cols=numeric_cols,
            cat_cols=cat_cols,
            X_train=X_train,
            y_train=y_train,
            sample_weights=sample_weights,
            n_trials=OPTUNA_TRIALS,
            odds_cols=ODDS_COLS,
        )

    # -----------------------------------------------------------------------
    # 8. Train & evaluate all models
    # -----------------------------------------------------------------------
    models = get_models(numeric_cols, cat_cols, xgb_params=xgb_params, odds_cols=ODDS_COLS)
    results = train_and_compare(
        models, X_train, y_train, X_test, y_test,
        label_enc=label_enc,
        n_cv_splits=5,
        sample_weights=sample_weights,
    )

    # -----------------------------------------------------------------------
    # 9. Pick best model & save artifacts
    # [WHY cv_mean for selection?] See model_trainer.print_summary docstring.
    #   Short answer: selecting by test_acc is test-set leakage.
    # [WHY pass X_train/y_train to save_artifacts?] Calibration must be fitted
    #   on training data only. See save_artifacts docstring for full rationale.
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
        X_train=X_train,
        y_train=y_train,
    )


if __name__ == "__main__":
    main()