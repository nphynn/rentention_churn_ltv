# Customer Analytics on Online Retail II Technical Approach
### Retention × Churn × LTV 

## Overview

- **Business question:** A retail business has finite marketing budget. Where does the next dollar go?
- **Approach:** Build retention, churn, and LTV models in parallel, segment customers two ways (RFM rules + K-Means), then fuse value × risk into a single action grid.
- **Modeling:** Three model families per target: **statistical for narrative and calibration**, **ML for production scoring**, **deep learning to show the sequence-aware upgrade path**.
- **Output:** A `customer_action_list.csv` with predicted 60-day CLV, churn probability, RFM segment, K-Means cluster, recommended action and expected save value per intervention.

## Why retention, churn, and LTV have to be modeled *together*

This is the core insight the project is built around. Each of the three answers a different question and each, on its own mis-allocates budget.

| Question | What it answers alone | What it misses alone |
|---|---|---|
| **Retention** — who comes back? | Forward probability of activity | Says nothing about how *valuable* coming back is |
| **Churn** — who's quietly leaving? | Risk ranking of active customers | Says nothing about whether the customer is worth saving |
| **LTV** — who's worth investing in? | Future revenue per customer | Says nothing about whether they're at risk *right now* |

The action a marketing team takes only makes sense when all three are on the table at once:

- A **high-churn-risk × low-LTV** customer gets a single automated email (cheap, low expected value).
- A **high-churn-risk × high-LTV** customer gets personal outreach from an account manager 
- A **low-churn-risk × high-LTV** customer (a Champion) gets a thank-you and offers for VIP programs. A discount campaign aimed at them is pure margin loss.
- A **low-churn-risk × low-LTV** customer gets ignored. Acquisition replaces them more efficiently than retention does.

That 2×2 is the entire point of the project. Predictive models give scores, segmentation gives names, **combining value (LTV) with risk (churn) on top of a named segment (RFM/cluster) is what turns scores into a campaign brief**. None of the three answers alone is a strategy.

## Why segmentation needs *both* RFM rules and K-Means

| Method | Strength | Blind spot |
|---|---|---|
| **RFM quintile rules** | Fully reproducible; "Champions" means the same thing every quarter | Quintile boundaries are arbitrary as a customer can flip segments because of a one-quintile shift |
| **K-Means on log-transformed RFM** | Lets the data find natural groupings and handles the non-linear shape of revenue distributions when fed log features | Cluster identities drift between training runs |

## Modeling philosophy: three families, three jobs

Every target (retention, churn, LTV) is modeled three ways with each family having a distinct role

| Family | What it's *for* | Role here |
|---|---|---|
| **Statistical** (Cox PH, Weibull AFT, BG/NBD + Gamma-Gamma) | Interesting findings, calibrated probabilities, parametric forecasts | Explains *why* customers churn (hazard ratios, AFT time-multipliers). Executive narrative. |
| **Machine Learning** (LogReg, LightGBM, XGBoost) | Production scoring | Best discrimination/regression on tabular RFM. Tunable, fast to retrain, SHAP-explainable. **What gets deployed.** |
| **Deep Learning** (GRU + Attention, Transformer, LSTM) | Sequence-aware modeling | At RFM-only feature density, parity with ML on this dataset (LSTM nudged XGBoost on LTV). **The architecture documented in code so it's ready when event-level features arrive.** |

## Pipeline walkthrough — model choices, optimizations and trade-offs

### Section 1 — Data quality audit

Standard cleaning: drop missing Customer IDs (guest checkouts can't be modeled at customer level), drop cancellation invoices (`Invoice` starts with `C` — they double-count), drop non-positive quantity/price (modeling artifacts). Single source of truth `df_clean` feeds every downstream model.

### Section 2 — Customer-level feature engineering

Aggregate transaction grain → one row per customer with R/F/M, tenure, AOV, item diversity, country. RFM features typically dominate repeat-purchase prediction in non-contractual retail so they will be used to anchor every downstream model.

### Section 4 — Segmentation (RFM + K-Means)

**Optimizations:**
- K-Means is fit on **log-transformed** frequency and revenue — raw values are heavy-tailed and break the equal-variance assumption clustering relies on.
- `k` is chosen via combined **elbow + silhouette**, not a hard-coded value. The chosen `k` is capped at 5 for business actionability as too many clusters and marketing can't write that many campaign briefs.
- PCA visualization in 2D shows whether clusters are genuinely separated or whether the structure is more continuous than discrete.

**Trade-offs vs alternatives:**
- *Why not Gaussian Mixture Models?* GMM gives soft assignments and works on heteroscedastic clusters but the segments are harder to communicate as "47% Champion, 32% Loyal" doesn't fit on a campaign brief. K-Means'. 
- *Why not hierarchical clustering?* K-Means' centroids are easier to interpret as "average customer" archetypes.

### Section 5 — Retention models (will they come back in the next 60 days?)

#### 5.1 Statistical — Kaplan-Meier + Cox Proportional Hazards

KM is non-parametricand it estimates the survival curve `S(t) = P(active at time t)` with no distributional assumption. Cox PH layers covariates on top to quantify which features extend or shorten customer lifespan. Together they tell you **how long customers stay and what makes them stay**, which is an executive-summary question.

**Optimizations:** Log-transform of `frequency` and `aov` before fitting (heavy-tailed → leverage points distort coefficients); UK indicator collapsed from country to manage degrees of freedom.

**Trade-offs vs alternatives:**
- *Cox vs parametric (Weibull) regression for retention specifically:* Cox doesn't extrapolate beyond observed durations. Fine for the next 60 days, wrong for "what fraction of this cohort is still active in 2 years?" — that question goes to AFT in Section 6.
- *Cox vs random survival forest:* RSF handles non-linearities natively but loses the hazard ratio interpretation that makes Cox the right tool for executive reporting.

#### 5.2 ML — Logistic Regression

Right baseline for binary retention. Fast, interpretable (coefficients → odds ratios), well-calibrated probabilities with proper feature scaling. The right starting point and anything more complex has to beat it before it earns its place in production.

**Optimizations:**
- **Time-aware split**: features computed on data ending `T-60`, label is activity in the last 60 days. Random splits leak: features and labels overlap in time.
- **Regularization strength tuned via cross-validation** (`LogisticRegressionCV`) instead of accepting the default `C=1.0`.
- **Threshold optimization on cost asymmetry**: default 0.5 implicitly assumes false positives and false negatives cost the same. They don't, an email is £0.20 and an at-risk customer is worth £25+. The optimal threshold is `email_cost / customer_value` ≈ 0.008, meaning *email anyone with even modest churn risk*. The default 0.5 leaves 90%+ of expected value on the table.

**Trade-offs vs alternatives:**
- *Why not jump straight to LightGBM here?* LogReg is the calibration baseline and its AUC tells you whether the problem is linear-separable on RFM. LightGBM only beats it by 0.01 on this dataset, therefore the extra complexity is fragile.

#### 5.3 DL — GRU + Attention

RFM aggregates collapse a year of customer history into three numbers. A GRU reads which months were heavy, which were light, whether activity is accelerating or decelerating. Attention pooling weights informative months which is the DL analog of feature importance.

**Optimizations:** Train/val/test split (early stopping needs the val set), early stopping on validation AUC, `ReduceLROnPlateau` learning-rate scheduler, gradient clipping for training stability, best-checkpoint restoration (returning the best val-AUC weights, not the last-epoch weights).

**Trade-offs / honest finding:** On RFM-only monthly sequences, the GRU typically **matches** logistic regression on this dataset. Adding product category embeddings, day-of-week features and session-level events would be better feautes which sequence modeling materially beats tabular ML. **This section exists to document the architecture and discipline so it can ingest those features if a suitable dataset is avaliable and not to claim AUC superiority on the current data.**

### Section 6 — Churn models (who's quietly leaving?)

#### 6.1 Statistical — Weibull AFT

Weibull AFT gives hazard *magnitudes* to measure the actual time-to-churn estimates per customer. Coefficients are multiplicative on time: `exp(coef) = 1.5` means the covariate extends survival by 50%. Output ships directly to the marketing team as "this customer is on track to churn in ~X days."

The Weibull `rho_` shape parameter is the diagnostic gem for retail it's typically <1, meaning the hazard is highest *early* in tenure. This is the empirical fingerprint of the "first 30 days problem" as the largest retention lift comes from onboarding, not win-back.

**Trade-offs vs alternatives:**
- *Weibull vs Log-Normal AFT:* both are parametric. Weibull is the right pick when hazard is monotonic (which it is here — risk decreases with tenure); Log-Normal handles bathtub curves where mid-life is safest.
- *AFT vs deep survival models (DeepSurv):* DeepSurv handles non-linear effects but loses the closed-form time-to-event estimate that makes AFT directly actionable.

#### 6.2 ML — LightGBM + SHAP

Gradient-boosted decision trees handle non-linearity and feature interactions natively (`high recency AND low frequency` is a stronger signal than either alone. SHAP makes the model explainable per-customer which is required for marketing teams to trust and act on individual scores.

**Optimizations vs an out-of-the-box implementation:**
- **`RandomizedSearchCV` hyperparameter tuning** (n_estimators, learning_rate, num_leaves, subsample, colsample_bytree, min_child_samples): accepting LightGBM defaults typically leaves 0.02-0.04 AUC on the table.
- **Stratified CV during search** — class imbalance (most active customers don't churn) makes plain k-fold unstable on AUC.
- **Engineered interaction features** (recency × frequency ratio, revenue-per-day): gradient boosting *can* find these on its own but giving them explicitly tightens the model and makes SHAP attributions cleaner.

**Trade-offs vs alternatives:**
- *LightGBM vs XGBoost for churn:* LightGBM is faster to train at this scale (leaf-wise growth, histogram binning). 
- *LightGBM vs CatBoost:* With one categorical feature (country, encoded as `is_uk`), it's not worth the dependency change.
- *Tree-based vs Neural net:* tree models are more sample-efficient on tabular data at this size.

#### 6.3 DL — Transformer Encoder

Self-attention weights events differently per customer which RFM cannot do. The architecture's natural use case is journey modeling once the data carries event-level richness.

**Optimizations:** Train/val/test split (was train/test only in earlier iterations), early stopping on val AUC + best-checkpoint restoration, `ReduceLROnPlateau`, gradient clipping.


### Section 7 — LTV models (how much will they spend?)

#### 7.1 Statistical — BG/NBD + Gamma-Gamma

Canonical probabilistic LTV for non-contractual settings. **BG/NBD** predicts the *number* of future transactions; **Gamma-Gamma** predicts the *monetary value* per transaction. Their product is expected revenue.

**Optimizations:** Validates the Gamma-Gamma `corr(frequency, monetary_value) < 0.1` assumption explicitly before fitting. If it fails, predicted CLV is biased and the regression-based fallback (Section 7.2) is the right choice instead.

**Trade-offs vs alternatives:**
- *BG/NBD vs Pareto/NBD:* BG/NBD is computationally cheaper and assumes customers churn after a transaction (more realistic for retail than Pareto/NBD's continuous-time churn). 
- *Probabilistic vs ML LTV:* probabilistic models are calibrated but ignore non-RFM features (country, item diversity, seasonality). XGBoost picks those up with Section 7.2 as the complement not the replacement.

#### 7.2 ML — XGBoost Regression

BG/NBD encodes a specific stochastic process. XGBoost makes no such assumption as it learns `(features) → future_revenue` from data including features BG/NBD ignores. Typically wins on **ranking** (Spearman) by 0.05-0.10 over BG/NBD; BG/NBD wins on **calibration** (predicted totals = actual totals).

**Optimizations:**
- **`RandomizedSearchCV`** over n_estimators, max_depth, learning_rate, subsample, colsample_bytree.
- **Log-transformed target** — revenue is heavy-tailed; MSE on raw scale wastes model capacity fitting the long tail rather than the modal customer.
- **Spearman as primary metric** — for budget allocation, ranking matters more than absolute prediction. R² on heavy-tailed targets is structurally low and misleading.

**Trade-offs vs alternatives:**
- *XGBoost vs Quantile regression:* quantile regression gives prediction intervals, useful for risk-adjusted budgets which is not the question here.

#### 7.3 DL — LSTM with proper training discipline:

Same intuition as the churn transformer. LSTMs are competitive with transformers on short sequences and faster to train.

**Optimizations:** Train/val/test split, early stopping, LR scheduler, gradient clipping.

### Section 8 — Robust evaluation: CV + time-based holdout

Every prior section used a single random 75/25 customer split. That's vulnerable to two failure modes:

1. **Lucky/unlucky split variance**: a single split's AUC can swing ±0.02 on this dataset.
2. **Temporal optimism**: random splits leak across time when feature distributions drift, inflating reported performance vs production.

**The two evaluations:**
- **Stratified 5-fold CV on customers**: gives mean AUC ± std (variance estimate).
- **Time-based holdout**: train on data ending `T-120`, test on the latest 60 days. Closest analog to deployed behavior.

**The deployment-honesty check** is the **CV − Holdout gap**. <0.02 → CV is honest about deployment. >0.05 → CV is overstating, usually due to temporal drift. Reporting both is necessary.

### Section 9 — Business action layer (the actual deliverable)

Combines three signals per customer into a `customer_action_list.csv`:

1. **Predicted 60-day CLV** (BG/NBD + Gamma-Gamma) — value
2. **Churn probability** (LightGBM) — risk
3. **RFM segment + K-Means cluster** — identity

Each customer gets a **value tier** × **risk tier** placement, a **recommended action**, and an **expected save value** (`P(churn) × CLV × LIFT − email_cost`). The marketing team works the grid top-down, top-row first, until the marginal expected save falls below cost-per-touch.

**The Top 30 priority list** = High-value × High-risk customers, sorted by expected save value. This is the immediate retention queue.
  

## Tech stack

- **Core:** Python 3, pandas, NumPy, matplotlib, seaborn
- **Statistical:** `lifelines` (KM, Cox, Weibull AFT), `lifetimes` (BG/NBD, Gamma-Gamma), `statsmodels`
- **ML:** scikit-learn (LogReg, K-Means, PCA, CV), `lightgbm`, `xgboost`, `shap`
- **DL:** PyTorch (GRU + Attention, Transformer Encoder, LSTM)
- **Tuning:** `RandomizedSearchCV` with `StratifiedKFold`

## Dataset citation

Chen, D. (2019). Online Retail II. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/502/online+retail+ii

---
