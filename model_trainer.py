"""
model_trainer.py — Model definitions, evaluation and persistence.

Responsibilities:
  - Define sklearn Pipelines for each model family
  - Time-series cross-validation helper
  - Model comparison and best-model selection
  - Saving artifacts (model.pkl, label_encoder.pkl, etc.)

[WHY] Isolated from feature engineering and data loading so we can swap
classifiers (e.g. add LightGBM, CatBoost) without touching the rest.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
import xgboost as xgb

try:
    import lightgbm as lgb
    _LGBM_AVAILABLE = True
except ImportError:
    _LGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    _CATBOOST_AVAILABLE = False
except ImportError:
    _CATBOOST_AVAILABLE = False


# ===========================================================================
# 1. Pipeline builder
# ===========================================================================

def build_pipeline(
    classifier,
    numeric_cols: list[str],
    cat_cols: list[str],
) -> Pipeline:
    """
    Wraps a classifier in a sklearn Pipeline with:
      - SimpleImputer (fill_value=0) + StandardScaler for numeric features
      - SimpleImputer (most_frequent) + OneHotEncoder for categorical features

    [WHY] Pipeline ensures:
      1. Preprocessing always runs before the classifier.
      2. model.predict(raw_X) works without manual transformation.
      3. Scaler is fit only on X_train — no train/test leakage.
    [WHY fill_value=0] Rolling features default to 0 for teams with no history,
      so imputing 0 is semantically correct.
    [WHY scale trees?] Trees don't need scaling but it does no harm, and keeping
      a uniform preprocessor simplifies pipeline comparison.
    [WHY scale SVM?] SVM is distance-based — unscaled features will dominate.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                    ("scaler", StandardScaler()),
                ]),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]),
                cat_cols,
            ),
        ],
        remainder="drop",
    )
    preprocessor.set_output(transform="pandas")

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


def build_catboost_pipeline(
    numeric_cols: list[str],
    cat_cols: list[str],
    **catboost_kwargs,
) -> Pipeline:
    """
    Dedicated pipeline for CatBoost that preserves its native categorical handling.

    [WHY a separate builder?] If we pass HomeTeam/AwayTeam through OneHotEncoder
    (as build_pipeline() does for other models), CatBoost never sees the raw
    categorical signal — it just receives a sparse binary matrix like any other
    model. CatBoost's advantage comes from its ordered target statistics on raw
    categoricals, so we must pass them as integer-encoded columns and declare
    their indices via cat_features.

    [HOW] OrdinalEncoder converts team names → stable integers. CatBoost is
    told which output columns are categorical via their positional index
    (numerics first, then cats — matching ColumnTransformer output order).
    unknown_value=-1 safely handles unseen teams at prediction time.
    """
    from sklearn.preprocessing import OrdinalEncoder

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                    # [WHY no scaler?] CatBoost is a tree ensemble — scaling
                    # has zero effect on splits and only wastes a tiny bit of time.
                ]),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer",  SimpleImputer(strategy="most_frequent")),
                    ("ordinal",  OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    )),
                ]),
                cat_cols,
            ),
        ],
        remainder="drop",
    )

    # After ColumnTransformer, numerics occupy indices 0…len(numeric_cols)-1
    # and categoricals occupy the next len(cat_cols) indices.
    cat_feature_indices = list(range(len(numeric_cols), len(numeric_cols) + len(cat_cols)))

    defaults = dict(
        iterations=500,
        depth=6,
        learning_rate=0.03,
        loss_function="MultiClass",
        cat_features=cat_feature_indices,
        random_seed=42,
        verbose=0,
    )
    defaults.update(catboost_kwargs)

    clf = CatBoostClassifier(**defaults)

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   clf),
    ])


# ===========================================================================
# 2. Model catalogue
# ===========================================================================

def get_models(
    numeric_cols: list[str],
    cat_cols: list[str],
) -> dict[str, Pipeline]:
    """
    Returns a dictionary of named sklearn Pipelines to compare.

    Models
    ------
    XGBoost        — State-of-the-art GBT with L1/L2 regularization.
    GradientBoosting — sklearn native GBT; slower but no extra dependency.
    LightGBM       — Leaf-wise GBT; faster than XGBoost on many datasets.
                     Skipped if lightgbm is not installed.
    SVM (RBF)      — Kernel SVM for nonlinear decision boundaries.
                     [WHY RBF?] Football features are not linearly separable.
                     [WARNING] O(n²) training — fine for 1500 rows, slow beyond 10k.
    """
    models: dict[str, Pipeline] = {}

    models["XGBoost"] = build_pipeline(
        xgb.XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
        ),
        numeric_cols, cat_cols,
    )

    models["GradientBoosting"] = build_pipeline(
        GradientBoostingClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.8,
            min_samples_leaf=5,
            random_state=42,
        ),
        numeric_cols, cat_cols,
    )

    if _LGBM_AVAILABLE:
        models["LightGBM"] = build_pipeline(
            lgb.LGBMClassifier(
                n_estimators=400,
                max_depth=5,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=10,
                num_leaves=31,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            ),
            numeric_cols, cat_cols,
        )

    if _CATBOOST_AVAILABLE:
        # [WHY build_catboost_pipeline?] CatBoost handles HomeTeam/AwayTeam
        # natively via ordered statistics — wrapping it with OneHotEncoder
        # (as build_pipeline does) would discard that advantage entirely.
        models["CatBoost"] = build_catboost_pipeline(numeric_cols, cat_cols)

    models["SVM (RBF)"] = build_pipeline(
        SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=42,
        ),
        numeric_cols, cat_cols,
    )

    return models


# ===========================================================================
# 3. Time-series cross-validation
# ===========================================================================

def evaluate_with_tscv(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
) -> dict:
    """
    Evaluate *model* using TimeSeriesSplit cross-validation.

    [WHY TimeSeriesSplit?] Regular KFold randomly shuffles data, so test
    folds can contain matches from BEFORE training folds — leakage.
    TimeSeriesSplit ensures the test set is always AFTER training, mimicking
    real deployment: train on past, predict future.
    """
    if _CATBOOST_AVAILABLE:
        from catboost import CatBoostClassifier
    else:
        CatBoostClassifier = type("Dummy", (), {})

    # [WHY] sklearn's clone() breaks on CatBoost because its constructor
    # modifies cat_features internally. We manually run the folds instead.
    classifier = model.named_steps["classifier"]
    if isinstance(classifier, CatBoostClassifier):
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []
        for train_idx, test_idx in tscv.split(X):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
            model_clone = pickle.loads(pickle.dumps(model))  # deep copy via pickle
            model_clone.fit(X_tr, y_tr)
            scores.append(accuracy_score(y_te, model_clone.predict(X_te)))
        scores = np.array(scores)
    else:
        tscv = TimeSeriesSplit(n_splits=n_splits)
        # [WHY f1_macro?] Accuracy rewards predicting the majority class (Home
        # wins). f1_macro weights Draws, Away, and Home equally — forcing the
        # model to actually learn all three outcomes rather than ignoring draws.
        scores = cross_val_score(model, X, y, cv=tscv, scoring="f1_macro", n_jobs=1)

    return {
        "cv_scores": scores,
        "cv_mean": float(scores.mean()),
        "cv_std": float(scores.std()),
    }


# ===========================================================================
# 4. Training, comparison, and model selection
# ===========================================================================

def train_and_compare(
    models: dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    label_enc: LabelEncoder,
    n_cv_splits: int = 5,
) -> dict:
    """
    Train every model in *models*, run TSCV on training data, evaluate on
    holdout test set, print a report, and return all results.

    Returns
    -------
    dict keyed by model name, each value containing:
      - pipeline, test_acc, cv_mean, cv_std, cv_scores
    """
    results: dict = {}

    for name, pipeline in models.items():
        print(f"\n{'-'*50}")
        print(f"Training: {name}")

        cv_result = evaluate_with_tscv(pipeline, X_train, y_train, n_splits=n_cv_splits)
        print(f"  CV f1_macro: {cv_result['cv_mean']:.4f} ± {cv_result['cv_std']:.4f}")
        print(f"  CV per-fold: {[round(s, 4) for s in cv_result['cv_scores'].tolist()]}")

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        test_acc = accuracy_score(y_test, preds)
        print(f"  Holdout test accuracy: {test_acc:.4f}")

        print(classification_report(
            label_enc.inverse_transform(y_test),
            label_enc.inverse_transform(preds),
            zero_division=0,
        ))

        results[name] = {
            "pipeline": pipeline,
            "test_acc": test_acc,
            "cv_mean": cv_result["cv_mean"],
            "cv_std": cv_result["cv_std"],
            "cv_scores": cv_result["cv_scores"],
        }

    return results


def print_summary(results: dict) -> str:
    """Print and return the name of the best model by holdout test accuracy."""
    best_name = max(results, key=lambda k: results[k]["test_acc"])

    print(f"\n{'='*60}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*60}")
    for name, r in results.items():
        marker = " <-- BEST" if name == best_name else ""
        print(
            f"  {name:<35}  CV: {r['cv_mean']:.4f}  Test: {r['test_acc']:.4f}{marker}"
        )

    return best_name


# ===========================================================================
# 5. Save artifacts
# ===========================================================================

def save_artifacts(
    best_pipeline: Pipeline,
    label_enc: LabelEncoder,
    numeric_cols: list[str],
    cat_cols: list[str],
    use_betting_odds: bool,
    feat_df: pd.DataFrame,
    results: dict,
    out_dir: Path = Path("."),
) -> None:
    """
    Persist all artifacts needed by app.py.

    Files saved
    -----------
    model.pkl           — Best sklearn Pipeline (preprocessor + classifier).
    label_encoder.pkl   — LabelEncoder mapping int ↔ 'A'/'D'/'H'.
    feature_columns.pkl — Dict with numeric_cols, cat_cols, use_betting_odds.
    teams.pkl           — Sorted list of all team names seen in training.
    processed_matches.csv — Feature DataFrame for app auto-population.
    model_results.pkl   — Summary dict {name: {test_acc, cv_mean}} for UI.

    [WHY save the whole Pipeline?] app.py can call model.predict(raw_X)
    without any preprocessing code — the pipeline handles everything.
    """
    with open(out_dir / "model.pkl", "wb") as f:
        pickle.dump(best_pipeline, f)

    with open(out_dir / "label_encoder.pkl", "wb") as f:
        pickle.dump(label_enc, f)

    with open(out_dir / "feature_columns.pkl", "wb") as f:
        pickle.dump({
            "numeric_cols": numeric_cols,
            "cat_cols": cat_cols,
            "use_betting_odds": use_betting_odds,
        }, f)

    teams = sorted(set(feat_df["HomeTeam"]).union(set(feat_df["AwayTeam"])))
    with open(out_dir / "teams.pkl", "wb") as f:
        pickle.dump(teams, f)

    feat_df.to_csv(out_dir / "processed_matches.csv", index=False)

    results_summary = {
        k: {"test_acc": v["test_acc"], "cv_mean": v["cv_mean"]}
        for k, v in results.items()
    }
    with open(out_dir / "model_results.pkl", "wb") as f:
        pickle.dump(results_summary, f)

    print(
        "\nSaved: model.pkl, label_encoder.pkl, feature_columns.pkl, "
        "teams.pkl, processed_matches.csv, model_results.pkl"
    )
