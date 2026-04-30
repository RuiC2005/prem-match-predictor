# Improve Premier League Match Predictor

Current accuracy: **43.5%** (barely above random for a 3-class problem). Goal: **55%+** with better features and model.

## Diagnosis

| Problem | Impact |
|---|---|
| **Only using 6 basic form features** — the CSVs contain 100+ columns including shots, shots on target, corners, fouls, cards, half-time scores, and **bookmaker odds** | High |
| **Logistic Regression** is too weak for nonlinear feature interactions | High |
| **No Elo / power rating** — the model has no concept of team strength beyond 5-match form | High |
| **No head-to-head features** — some matchups have consistent historical patterns | Medium |
| **No home/away-specific form** — a team's home form ≠ away form | Medium |
| **DataFrame fragmentation warning** from inserting `source_file` in a loop | Low (cosmetic) |

## Proposed Changes

### 1. Fix the PerformanceWarning

#### [MODIFY] [train_model.py](file:///c:/Users/noell/Documents/ML/prem-match-predictor/train_model.py)

Assign `source_file` via `.assign()` before appending to the list, avoiding repeated inserts into the same DataFrame.

---

### 2. Richer Feature Engineering

#### [MODIFY] [train_model.py](file:///c:/Users/noell/Documents/ML/prem-match-predictor/train_model.py)

Add these new rolling features computed from existing CSV columns:

| Feature Group | Source Columns | New Features |
|---|---|---|
| **Match stats form** | `HS, AS, HST, AST, HC, AC, HF, AF` | Rolling avg shots, shots on target, corners, fouls (last 5 matches per team) |
| **Discipline** | `HY, AY, HR, AR` | Rolling avg yellow/red cards |
| **Elo rating** | Win/loss results | Dynamic Elo score per team (K=20, start=1500) — captures overall team strength better than 5-match window |
| **Head-to-head** | Historical matchups | H2H win rate for home team vs specific opponent (last 6 meetings) |
| **Home/away-specific form** | Existing data | Separate rolling form for when team plays at home vs away |
| **Goal-scoring momentum** | `FTHG, FTAG` | Rolling goal difference, clean sheet %, scoring streak |
| **Rest days** | `Date` | Days since each team's last match |

> [!IMPORTANT]
> **Betting odds** (`B365H`, `B365D`, `B365A`, `AvgH`, `AvgD`, `AvgA`) are available in the data and are *extremely* powerful predictors — bookmakers already encode team strength, injuries, suspensions, etc. However, **using odds as features feels like "cheating"** since they essentially *are* predictions themselves. I'll add them as an **optional flag** so you can toggle them on/off and compare.

---

### 3. Upgrade the Model

#### [MODIFY] [train_model.py](file:///c:/Users/noell/Documents/ML/prem-match-predictor/train_model.py)

- Replace `LogisticRegression` with **Gradient Boosted Trees** (`GradientBoostingClassifier` from scikit-learn — no new dependencies needed)
- GBT handles nonlinear interactions, mixed feature types, and missing values much better
- Add **time-series cross-validation** (`TimeSeriesSplit`) instead of a single 80/20 split for more robust evaluation
- Print CV scores alongside the final test score

> [!NOTE]
> I'm using scikit-learn's `GradientBoostingClassifier` rather than XGBoost/LightGBM to avoid adding new dependencies. If you want to pull in `xgboost`, we can squeeze out a bit more performance.

---

### 4. Update the Streamlit App

#### [MODIFY] [app.py](file:///c:/Users/noell/Documents/ML/prem-match-predictor/app.py)

- **Auto-populate form inputs** from `processed_matches.csv` — when user selects Home/Away team, pull their latest rolling stats instead of requiring manual slider input
- Add the new features to the prediction input row
- Show Elo ratings for context

---

## Open Questions

> [!IMPORTANT]
> **Include betting odds?** They'll boost accuracy significantly (~55-60%) but the model is basically just learning to read the bookmaker's prediction. Do you want them included as an option, excluded entirely, or on by default?

> [!NOTE]
> **XGBoost dependency?** Scikit-learn's `GradientBoostingClassifier` is solid but `xgboost` is faster and often slightly better. Want me to add it to `requirements.txt`, or keep the dependency list minimal?

## Verification Plan

### Automated Tests
- Run `train_model.py` and compare accuracy before/after
- Verify no warnings are emitted
- Verify all pickle files are saved correctly

### Manual Verification
- Run Streamlit app and confirm auto-populated form values work
- Confirm prediction probabilities are sensible
