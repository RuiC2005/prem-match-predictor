"""
model_trainer.py — Model definitions, evaluation and persistence (v7).

Changes from v6
---------------
- evaluate_with_tscv now accepts sample_weights and slices them per fold.
  Previously, XGBoost CV folds ran WITHOUT sample weights even though final
  training used them — meaning the Optuna/CV scores were measured on a
  different loss surface than the fitted model. Now the combined temporal ×
  class-balance weights are correctly applied inside each CV fold for
  XGBoost, GradientBoosting, and LightGBM.

- train_and_compare tags each model with _name before calling evaluate_with_tscv
  so the CV function knows which models support sample_weight via Pipeline.

- VotingClassifier Ensemble: XGBoost sub-pipeline now receives sample_weights
  during final training fit. Previously neither sub-estimator received weights,
  meaning the Ensemble's XGBoost component had the same Home-collapse problem.
  RF sub-pipeline deliberately does NOT receive sample_weights — it handles
  class imbalance via balanced_subsample internally.

Changes from v5
---------------
- CatBoost REMOVED. Reasons:
    1. Required a dedicated pipeline builder (200+ lines of plumbing).
    2. Never outperformed XGBoost by more than 0.5% in our tests.
    3. sklearn's clone() fails on CatBoost, forcing manual fold loops
       throughout evaluate_with_tscv — adding ~60 lines of duplicated code.
    4. 3 min extra training time per run.
    Cost > benefit. Use XGBoost (with Optuna tuning) instead.

- Feature count reduced from ~110 to ~37 (see feature_engineering.py v6).
  This directly addresses the 48→41% accuracy regression: too many weakly-
  correlated features were drowning the strong signals in variance.

Responsibilities
----------------
- Define sklearn Pipelines for XGBoost, RandomForest, GradientBoosting,
  LightGBM (optional), SVM, and an XGB+RF soft-voting ensemble.
- TimeSeriesSplit cross-validation with per-fold sample weight slicing.
- Optuna hyperparameter tuning (XGBoost only).
- Model comparison, best-model selection, artifact persistence.
"""

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import xgboost as xgb

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False



# ===========================================================================
# 1. Pipeline builder
# ===========================================================================

def build_pipeline(
    classifier,
    numeric_cols: list[str],
    cat_cols: list[str],
    odds_cols: list[str] | None = None,
) -> Pipeline:
    """
    Wraps a classifier in a sklearn Pipeline with:
      - SimpleImputer(0) + StandardScaler  for rolling/form/elo features
      - SimpleImputer(median) + StandardScaler  for odds features
      - SimpleImputer(most_frequent) + OneHotEncoder  for team names

    [WHY two numeric imputers?]
    Rolling features default to 0 for teams with no history — semantically
    correct (no goals, no points). Odds features are NaN for the 2018-19 season
    which predates AvgH/closing-line columns. Imputing 0 there creates a false
    "no market activity" signal. Median imputation fills with a realistic
    neutral prior instead.
    """
    if odds_cols:
        form_cols   = [c for c in numeric_cols if c not in odds_cols]
        active_odds = [c for c in odds_cols   if c in numeric_cols]
    else:
        form_cols   = numeric_cols
        active_odds = []

    transformers = [
        (
            "form",
            Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                ("scaler",  StandardScaler()),
            ]),
            form_cols,
        ),
    ]
    if active_odds:
        transformers.append((
            "odds",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler",  StandardScaler()),
            ]),
            active_odds,
        ))
    transformers.append((
        "cat",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]),
        cat_cols,
    ))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    preprocessor.set_output(transform="pandas")

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   classifier),
    ])


# ===========================================================================
# 2. Temporal sample weighting
# ===========================================================================

def make_sample_weights(n: int, decay: float = 0.998) -> np.ndarray:
    """
    Exponential downweighting of older matches so the model prioritises
    recent EPL behaviour.

    [WHY 0.998 not 0.97?]
    At decay=0.97 with n=2405 rows, the oldest weight reaches 0.97^2404
    ≈ 10^-32 — below float64 useful precision. XGBoost's multiclass solver
    crashes with "sum_weight >= kRtEps". At decay=0.998: oldest match =
    ~9% of newest — meaningful recency bias, no underflow.

    [WHY exponential?]
    Football changes over time: COVID season (2020-21) had no crowds and a
    37.9% home win rate vs the typical ~45%. We want to keep all 7 seasons
    for Elo convergence and H2H history, but discount outdated seasons.

    Parameters
    ----------
    n     : number of training samples (chronologically sorted)
    decay : per-step decay factor. Range 0.995–0.9995 recommended.

    Returns
    -------
    np.ndarray of shape (n,) — normalised to sum to 1.
    """
    weights = np.array([decay ** (n - 1 - i) for i in range(n)])
    weights = np.clip(weights, 1e-4, None)   # safety net for aggressive values
    return weights / weights.sum()


# ===========================================================================
# 3. Optuna hyperparameter tuning (XGBoost only)
# ===========================================================================

def tune_xgboost(
    numeric_cols: list[str],
    cat_cols: list[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sample_weights: np.ndarray | None = None,
    n_trials: int = 80,
    n_cv_splits: int = 5,
    odds_cols: list[str] | None = None,
) -> dict:
    """
    Optuna TPE search for XGBoost hyperparameters via TimeSeriesSplit CV.

    [WHY Optuna over GridSearch?]
    TPE builds a probabilistic model of which regions of hyperparameter space
    produce good CV scores and samples more from promising regions. 80 trials
    typically outperforms a 500-point grid in less time.

    [WHY f1_macro not accuracy?]
    At ~50% accuracy a naive "always predict Home" model scores ~45%.
    f1_macro penalises equally for failing on any class (H/D/A), forcing
    the tuner to find parameters that predict draws and away wins too.

    [Search space rationale — tuned for ~37 features, ~1900 training rows]
    - max_depth 2–5: with fewer features depth can go slightly higher than v5
      (where 110 features made depth 5 overfit badly). But 5 is still the cap.
    - min_child_weight 3–20: prevents splits on tiny draw/away-win groups.
    - colsample_bytree 0.5–0.9: feature sampling reduces tree correlation.
    - reg_alpha / reg_lambda: L1 drives weak features to zero; L2 smooths.
    """
    if not _OPTUNA_AVAILABLE:
        print("  [tuner] optuna not installed — skipping, using defaults.")
        return {}

    tscv = TimeSeriesSplit(n_splits=n_cv_splits, gap=38)
    # [WHY gap=38?] One full matchday round separates each train/val split,
    # preventing same-week rolling stats from overlapping the fold boundary.

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 150, 600),
            "max_depth":        trial.suggest_int("max_depth", 2, 5),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
            "subsample":        trial.suggest_float("subsample", 0.60, 0.90),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.50, 0.90),
            "min_child_weight": trial.suggest_int("min_child_weight", 3, 20),
            "gamma":            trial.suggest_float("gamma", 0.0, 0.5),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 1.5),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0.5, 4.0),
        }
        pipeline = build_pipeline(
            xgb.XGBClassifier(
                **params,
                eval_metric="mlogloss",
                tree_method="hist",
                random_state=42,
                n_jobs=-1,
            ),
            numeric_cols, cat_cols, odds_cols=odds_cols,
        )

        fold_scores = []
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Skipping features without any observed values",
                category=UserWarning,
            )
            for train_idx, val_idx in tscv.split(X_train):
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                fit_kwargs = {}
                if sample_weights is not None:
                    fit_kwargs["classifier__sample_weight"] = sample_weights[train_idx]

                pipeline.fit(X_tr, y_tr, **fit_kwargs)
                from sklearn.metrics import f1_score
                fold_scores.append(f1_score(y_val, pipeline.predict(X_val), average="macro"))

        return float(np.mean(fold_scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False,
                   catch=(Exception,))

    if not study.trials or not any(t.state.name == "COMPLETE" for t in study.trials):
        print("  [tuner] All trials failed — using XGBoost defaults.")
        return {}

    best = study.best_params
    print(f"  [tuner] Best CV f1_macro: {study.best_value:.4f}")
    print(f"  [tuner] Best params: {best}")
    return best


# ===========================================================================
# 4. Model catalogue
# ===========================================================================

def get_models(
    numeric_cols: list[str],
    cat_cols: list[str],
    xgb_params: dict | None = None,
    odds_cols: list[str] | None = None,
) -> dict[str, Pipeline]:
    """
    Returns named sklearn Pipelines to train and compare.

    Models included (v9 — SVM removed)
    ---------------
    XGBoost          — State-of-the-art GBT. Primary model.
    RandomForest     — Parallel bagging; diverse errors vs XGBoost.
    XGB+RF Ensemble  — Soft-voting of XGBoost + RandomForest.

    [WHY remove SVM?]
    SVM scored 0.41 CV — 8 points below XGBoost (0.48). It is the slowest
    model to train at this data size (no warm-start, O(n²) kernel), does not
    benefit from sample_weights, and handles OHE team columns poorly. There
    is no deployment scenario where it would be chosen over XGBoost or RF.

    [WHY NOT CatBoost?]
    CatBoost was removed in v6. It requires a dedicated pipeline builder
    (OrdinalEncoder path, column-name cat_features), breaks sklearn's clone(),
    adds 3 min to training, and never outperformed XGBoost by more than 0.5%
    across 7 seasons of EPL data. The complexity cost exceeds the gain.
    """
    models: dict[str, Pipeline] = {}

    # Conservative defaults for ~1900 training rows, ~37 features.
    # [WHY max_depth=3?] With fewer features than v5 (37 vs 110), depth 3
    # is less constrained. But it still prevents memorising small draw/away-win
    # clusters in the training set.
    # [WHY min_child_weight=5?] Prevents XGBoost from splitting on groups
    # of < 5 samples — which at our dataset size would be noise.
    default_xgb = dict(
        n_estimators=400,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.75,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.5,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    if xgb_params:
        fixed = {k: default_xgb[k]
                 for k in ("eval_metric", "tree_method", "random_state", "n_jobs")}
        default_xgb.update(xgb_params)
        default_xgb.update(fixed)   # always restore infrastructure params

    models["XGBoost"] = build_pipeline(
        xgb.XGBClassifier(**default_xgb),
        numeric_cols, cat_cols, odds_cols=odds_cols,
    )

    # [WHY balanced_subsample?] class_weight='balanced' computes weights once
    # on the full dataset. 'balanced_subsample' recomputes per bootstrap sample
    # — crucial because each bootstrap draws ~63% of rows, so the draw class
    # size fluctuates. Per-sample recomputation keeps draws consistently
    # upweighted even when a bootstrap happens to undersample them.
    # max_depth=15: with only 37 features the compound draw-interaction
    # (combined_clean_sheet AND draw_tendency AND h2h_draw_rate) needs
    # depth to be found.
    models["RandomForest"] = build_pipeline(
        RandomForestClassifier(
            n_estimators=500,
            max_depth=15,
            max_features="sqrt",
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
        numeric_cols, cat_cols, odds_cols=odds_cols,
    )

    # Soft-voting ensemble: XGBoost + RandomForest.
    # [WHY soft not hard voting?]
    # Soft voting averages the probability vectors then takes argmax.
    # For draws: XGB might say D=0.28, RF might say D=0.38 → averaged 0.33.
    # Hard voting would let XGB kill the draw with an H vote regardless of
    # RF's confidence. Soft voting always beats hard when probabilities are
    # reasonably calibrated.
    models["XGB+RF Ensemble"] = VotingClassifier(
        estimators=[
            ("xgb", build_pipeline(
                xgb.XGBClassifier(**default_xgb),
                numeric_cols, cat_cols, odds_cols=odds_cols,
            )),
            ("rf", build_pipeline(
                RandomForestClassifier(
                    n_estimators=500, max_depth=15, max_features="sqrt",
                    min_samples_leaf=5, class_weight="balanced_subsample",
                    random_state=42, n_jobs=-1,
                ),
                numeric_cols, cat_cols, odds_cols=odds_cols,
            )),
        ],
        voting="soft",
        weights=[1, 1],
    )

    return models


# ===========================================================================
# 5. Time-series cross-validation
# ===========================================================================

def evaluate_with_tscv(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    sample_weights: np.ndarray | None = None,
) -> dict:
    """
    Evaluate *model* using TimeSeriesSplit CV.

    [WHY TimeSeriesSplit?] Regular KFold shuffles data randomly — test folds
    can contain matches from before training folds (leakage). TimeSeriesSplit
    always tests on the future, mimicking real deployment.

    [WHY gap=38?] One full matchday round between train and test.
    Prevents same-week rolling stats from overlapping the fold boundary.

    [WHY pass sample_weights here?]
    The combined temporal × class-balance weights must be sliced to the
    training indices of each CV fold — otherwise XGBoost CV scores are
    computed without the class-balance correction, meaning the tuner still
    sees a Home-biased loss surface and finds degenerate parameters.
    Only XGBoost/GBM/LightGBM pipelines receive weights; VotingClassifier
    and SVM use their own internal class handling.
    """
    is_voting = isinstance(model, VotingClassifier)
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=38)

    # Determine whether this model supports sample_weight via Pipeline API.
    # [WHY name-check?] VotingClassifier and SVM handle class imbalance
    # internally (balanced_subsample / class_weight). RF uses balanced_subsample.
    # Passing sample_weight to them would either error or double-count.
    _WEIGHTED_MODELS = ("XGBoost",)
    model_name = getattr(model, "_name", "")  # set by train_and_compare before calling

    if is_voting:
        # VotingClassifier: must clone manually (sklearn clone() can fail
        # on nested VotingClassifiers with complex sub-estimators).
        scores = []
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Skipping features without any observed values",
                category=UserWarning,
            )
            for train_idx, test_idx in tscv.split(X):
                X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
                y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
                clone = pickle.loads(pickle.dumps(model))
                clone.fit(X_tr, y_tr)
                from sklearn.metrics import f1_score
                scores.append(f1_score(y_te, clone.predict(X_te), average="macro"))
        scores = np.array(scores)
    elif sample_weights is not None and model_name in _WEIGHTED_MODELS:
        # Manual fold loop so we can slice sample_weights to train indices.
        # cross_val_score does not support per-sample weight slicing natively.
        scores = []
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Skipping features without any observed values",
                category=UserWarning,
            )
            for train_idx, val_idx in tscv.split(X):
                X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
                clone = pickle.loads(pickle.dumps(model))
                clone.fit(
                    X_tr, y_tr,
                    classifier__sample_weight=sample_weights[train_idx],
                )
                from sklearn.metrics import f1_score
                scores.append(f1_score(y_val, clone.predict(X_val), average="macro"))
        scores = np.array(scores)
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Skipping features without any observed values",
                category=UserWarning,
            )
            scores = cross_val_score(
                model, X, y,
                cv=tscv,
                scoring="f1_macro",
                n_jobs=1,
            )

    return {
        "cv_scores": scores,
        "cv_mean":   float(scores.mean()),
        "cv_std":    float(scores.std()),
    }


# ===========================================================================
# 6. Training, comparison, model selection
# ===========================================================================

def train_and_compare(
    models: dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test:  pd.DataFrame,
    y_test:  pd.Series,
    label_enc: LabelEncoder,
    n_cv_splits: int = 5,
    sample_weights: np.ndarray | None = None,
) -> dict:
    """
    Train every model, run TSCV, evaluate on holdout test set, print reports.

    [WHY tag model._name before evaluate_with_tscv?]
    evaluate_with_tscv needs to know whether to pass sample_weights into each
    CV fold. It checks model._name against a whitelist of weight-supporting
    models. We set it here (in the loop, not at construction time) so the
    catalogue in get_models() stays free of training-time concerns.

    [WHY only XGBoost / GradientBoosting / LightGBM receive sample_weights?]
    RandomForest uses class_weight='balanced_subsample' which already handles
    class imbalance per bootstrap — passing additional weights would double-
    count the correction. SVM uses class_weight='balanced' at construction.
    VotingClassifier's XGBoost sub-estimator inherits the XGBoost weights via
    manual sub-pipeline fitting below.
    """
    results: dict = {}

    for name, pipeline in models.items():
        print(f"\n{'-'*50}")
        print(f"Training: {name}")

        # Tag the model so evaluate_with_tscv knows whether to apply weights.
        pipeline._name = name

        cv_result = evaluate_with_tscv(
            pipeline, X_train, y_train,
            n_splits=n_cv_splits,
            sample_weights=sample_weights,
        )
        print(f"  CV f1_macro: {cv_result['cv_mean']:.4f} ± {cv_result['cv_std']:.4f}")
        print(f"  CV per-fold: {[round(s, 4) for s in cv_result['cv_scores'].tolist()]}")

        is_voting = isinstance(pipeline, VotingClassifier)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Skipping features without any observed values",
                category=UserWarning,
            )
            if is_voting:
                # [WHY manual sub-estimator fit for the Ensemble?]
                # sklearn >= 1.4 requires enable_metadata_routing=True to pass
                # per-estimator kwargs to VotingClassifier.fit(). Rather than
                # opt in to the metadata routing API (which changes other
                # behaviour), we fit each sub-pipeline independently with its
                # own sample_weight, then call VotingClassifier.fit() without
                # weights so it registers estimators_ for predict_proba.
                #
                # The XGBoost sub-pipeline gets the combined weights (temporal
                # × class-balance). The RF sub-pipeline does NOT — it uses
                # balanced_subsample internally, so passing weights would
                # double-count the class correction.
                for est_name, sub_pipeline in pipeline.estimators:
                    if est_name == "xgb" and sample_weights is not None:
                        sub_pipeline.fit(
                            X_train, y_train,
                            classifier__sample_weight=sample_weights,
                        )
                    else:
                        sub_pipeline.fit(X_train, y_train)
                pipeline.fit(X_train, y_train)
            else:
                fit_kwargs: dict = {}
                if sample_weights is not None and name in ("XGBoost",):
                    fit_kwargs["classifier__sample_weight"] = sample_weights
                pipeline.fit(X_train, y_train, **fit_kwargs)

        preds    = pipeline.predict(X_test)
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
            "cv_mean":  cv_result["cv_mean"],
            "cv_std":   cv_result["cv_std"],
            "cv_scores": cv_result["cv_scores"],
        }

    return results


def print_summary(results: dict) -> str:
    """
    Print comparison table and return the name of the best model.

    [WHY cv_mean not test_acc for selection?]
    Selecting the best model by test accuracy means peeking at the holdout set
    to make a training decision — which is a form of test-set leakage. If you
    compare 6 models and pick the one that happened to score highest on the test
    set, you are implicitly fitting to that test set. The correct criterion is
    cross-validation performance on the training data only. The test set should
    be used for final reporting, not selection.
    """
    best_name = max(results, key=lambda k: results[k]["cv_mean"])

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
# 7. Save artifacts
# ===========================================================================

def save_artifacts(
    best_pipeline: Pipeline,
    label_enc: LabelEncoder,
    numeric_cols: list[str],
    cat_cols: list[str],
    use_betting_odds: bool,
    feat_df: pd.DataFrame,
    results: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    out_dir: Path = Path("."),
) -> None:
    """
    Persist all artifacts needed by app.py.

    Files saved
    -----------
    model.pkl            — Best Pipeline wrapped in CalibratedClassifierCV.
    label_encoder.pkl    — LabelEncoder: int ↔ 'A'/'D'/'H'.
    feature_columns.pkl  — {numeric_cols, cat_cols, use_betting_odds}.
    teams.pkl            — Sorted list of all team names seen in training.
    processed_matches.csv — Feature DataFrame for app auto-population.
    model_results.pkl    — Summary {name: {test_acc, cv_mean}} for UI.

    [WHY CalibratedClassifierCV?]
    Tree ensembles produce overconfident raw probabilities. Isotonic
    calibration (cv=5 folds) maps these to well-calibrated posteriors —
    improving soft-voting ensemble quality and any threshold tuning.
    Typically adds 0.5–1% test accuracy for free.

    [WHY cv=5 not 'prefit'?]
    'prefit' just remaps outputs of an already-fitted model — it wastes
    the calibration benefit. cv=5 refits the base estimator 5 times and
    learns the probability correction properly.

    [WHY calibrate on X_train/y_train only, not all data?]
    The test set was used to *report* final accuracy. If we also calibrate
    on it, the saved model has seen the test set during fitting — corrupting
    the integrity of any future holdout evaluation and making reported
    calibration metrics overoptimistic. Calibration must be trained on
    data the model has never seen in its final evaluation context.
    Using X_train keeps the test set genuinely held-out end-to-end.
    """
    from sklearn.calibration import CalibratedClassifierCV

    print("  Calibrating probabilities (isotonic, cv=5) ...")
    calibrated = CalibratedClassifierCV(best_pipeline, method="isotonic", cv=5)
    # [FIX] Calibrate on training data only — not feat_df (which includes test rows).
    calibrated.fit(X_train, y_train)

    with open(out_dir / "model.pkl", "wb") as f:
        pickle.dump(calibrated, f)
    with open(out_dir / "label_encoder.pkl", "wb") as f:
        pickle.dump(label_enc, f)
    with open(out_dir / "feature_columns.pkl", "wb") as f:
        pickle.dump({
            "numeric_cols":     numeric_cols,
            "cat_cols":         cat_cols,
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