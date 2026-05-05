# Customer Analytics on Online Retail II
### Retention × Churn × LTV — turning customer data into a prioritized action queue

A production-shaped customer analytics project on the **Online Retail II** dataset (UCI — UK gift retailer, ~1M transactions, Dec 2009 → Dec 2011). The deliverable isn't a model; it's a **prioritized customer list** with a recommended action and expected ROI per customer. The models are how we get there.

---

## TL;DR

- **Business question:** A retail business has finite marketing budget. Where does the next pound go?
- **Approach:** Build retention, churn, and LTV models in parallel, segment customers two ways (RFM rules + K-Means), then fuse value × risk into a single action grid.
- **Modeling philosophy:** Three model families per target — **statistical for narrative and calibration**, **ML for production scoring**, **deep learning to show the sequence-aware upgrade path** the data will support once it scales.
- **Output:** A `customer_action_list.csv` with predicted 60-day CLV, churn probability, RFM segment, K-Means cluster, recommended action, and expected save value per intervention.

---

## Why retention, churn, and LTV have to be modeled *together*

This is the core insight the project is built around. Each of the three answers a different question — and each, on its own, mis-allocates budget.

| Question | What it answers alone | What it misses alone |
|---|---|---|
| **Retention** — who comes back? | Forward probability of activity | Says nothing about how *valuable* coming back is |
| **Churn** — who's quietly leaving? | Risk ranking of active customers | Says nothing about whether the customer is worth saving |
| **LTV** — who's worth investing in? | Future revenue per customer | Says nothing about whether they're at risk *right now* |

The action a marketing team takes only makes sense when all three are on the table at once:

- A **high-churn-risk × low-LTV** customer gets a single automated email (cheap, low expected value).
- A **high-churn-risk × high-LTV** customer gets personal outreach from an account manager — saving them is worth £100s.
- A **low-churn-risk × high-LTV** customer (a Champion) gets a thank-you, never a discount. A discount campaign aimed at them is pure margin loss.
- A **low-churn-risk × low-LTV** customer gets ignored. Acquisition replaces them more efficiently than retention does.

That 2×2 is the entire point of the project. Predictive models give scores; segmentation gives names; **combining value (LTV) with risk (churn) on top of a named segment (RFM/cluster) is what turns scores into a campaign brief**. None of the three answers alone is a strategy.

---

## Why segmentation needs *both* RFM rules and K-Means

A common mistake is picking one segmentation technique and shipping it. The notebook deliberately runs both because each compensates for the other's blind spot.

| Method | Strength | Blind spot |
|---|---|---|
| **RFM quintile rules** | Fully reproducible; "Champions" means the same thing every quarter; marketing teams already speak this language | Quintile boundaries are arbitrary — a customer can flip segments because of a one-quintile shift that has no business meaning |
| **K-Means on log-transformed RFM** | Lets the data find natural groupings; handles the non-linear shape of revenue distributions when fed log features | Cluster identities drift between training runs; less directly explainable to non-technical stakeholders |

The two views agree on the obvious cases (Champions, Lost) and disagree at the boundaries — and that's exactly where you want the data-driven view to overrule the rules. Customers in this disagreement zone are flagged for closer human review. **Either method alone is a worse segmentation than running both and reconciling.**

---

## Modeling philosophy: three families, three jobs

Every target — retention, churn, LTV — is modeled three ways. This isn't model proliferation; each family has a distinct role.

| Family | What it's *for* | Why it's in this project |
|---|---|---|
| **Statistical** (Cox PH, Weibull AFT, BG/NBD + Gamma-Gamma) | Interesting findings, calibrated probabilities, parametric forecasts, executive narrative | These models tell you **why** customers churn (hazard ratios, AFT time-multipliers, the Weibull shape parameter that exposes the "first 30 days" fragility). They generate the slide deck. |
| **Machine Learning** (Logistic Regression, LightGBM, XGBoost — all tuned) | The model that actually runs in production and scores customers daily | Best discrimination/regression accuracy on tabular RFM features. Tunable, fast to retrain, SHAP-explainable. **This is what gets deployed.** |
| **Deep Learning** (GRU + Attention, Transformer, LSTM) | Sequence-aware modeling — most retail purchasing is sequential, not summary-stat-shaped | At RFM-only feature density, DL roughly **ties** ML on this dataset. Honest finding. The architecture is included because **once event-level features arrive (product categories, sessions, support contacts), this is the model that scales** — not LightGBM. The DL section is the upgrade path documented in code. |

Stated plainly: **statistics is for what's interesting, ML is for what's in production, deep learning is for what comes next.** With ~5K customers and RFM-only inputs, deep learning doesn't materially beat tuned LightGBM here — and that's an important business finding, not a project failure. It tells the data team where the next investment should go: not bigger models, **richer features**.

---

## Pipeline walkthrough — model choices, optimizations, and trade-offs

Each modeling section below covers: **why this model**, **what was optimized vs an out-of-the-box implementation**, **the trade-off vs the obvious alternative**, and **what to read in the result**.

### Section 1 — Data quality audit

Standard cleaning: drop missing Customer IDs (guest checkouts can't be modeled at customer level), drop cancellation invoices (`Invoice` starts with `C` — they double-count), drop non-positive quantity/price (modeling artifacts). Single source of truth `df_clean` feeds every downstream model.

**Optimization at this step:** the cleaning rationale is documented inline so the cohort can be reproduced. In production, this becomes a versioned data contract — the audit cell is the spec.

### Section 2 — EDA through a business lens

Three findings drive the rest of the project:
- Revenue is geographically concentrated in the UK — international expansion is structurally untapped.
- Customer revenue follows the classic Pareto distribution — top 20% generate the bulk. **This is why per-customer LTV modeling matters at all** — broadcast retention spend is mathematically wasteful when 80% of revenue lives in a small slice.
- Strong holiday seasonality (Nov-Dec spike). Any random-split model will be optimistic — addressed in Section 8 with time-based holdout evaluation.

### Section 3 — Customer-level feature engineering

Aggregate transaction grain → one row per customer with R/F/M, tenure, AOV, item diversity, country. RFM features dominate repeat-purchase prediction in non-contractual retail (Fader & Hardie, 2005), so they anchor every downstream model.

### Section 4 — Segmentation (RFM + K-Means)

**Optimizations:**
- K-Means is fit on **log-transformed** frequency and revenue — raw values are heavy-tailed and break the equal-variance assumption clustering relies on.
- `k` is chosen via combined **elbow + silhouette**, not a hard-coded value. The chosen `k` is capped at 5 for business actionability — too many clusters and marketing can't write that many campaign briefs.
- PCA visualization in 2D shows whether clusters are genuinely separated or whether the structure is more continuous than discrete.

**Trade-offs vs alternatives:**
- *Why not Gaussian Mixture Models?* GMM gives soft assignments and works on heteroscedastic clusters, but the segments are harder to communicate — "47% Champion, 32% Loyal" doesn't fit on a campaign brief. K-Means' hard assignment is the business asset.
- *Why not hierarchical clustering?* Doesn't scale well past ~10K customers; with ~5K here it would work, but K-Means' centroids are easier to interpret as "average customer" archetypes.

### Section 5 — Retention models (will they come back in the next 60 days?)

#### 5.1 Statistical — Kaplan-Meier + Cox Proportional Hazards

**Why this model:** KM is non-parametric — it estimates the survival curve `S(t) = P(active at time t)` with no distributional assumption. Cox PH layers covariates on top to quantify which features extend or shorten customer lifespan. Together they tell you **how long customers stay and what makes them stay**, which is the executive-summary question.

**Optimizations:** Log-transform of `frequency` and `aov` before fitting (heavy-tailed → leverage points distort coefficients); UK indicator collapsed from country to manage degrees of freedom.

**Trade-offs vs alternatives:**
- *Cox vs parametric (Weibull) regression for retention specifically:* Cox doesn't extrapolate beyond observed durations. Fine for the next 60 days, wrong for "what fraction of this cohort is still active in 2 years?" — that question goes to AFT in Section 6.
- *Cox vs random survival forest:* RSF handles non-linearities natively but loses the hazard ratio interpretation that makes Cox the right tool for executive reporting.

#### 5.2 ML — Logistic Regression (tuned)

**Why this model:** Right baseline for binary retention. Fast, interpretable (coefficients → odds ratios), well-calibrated probabilities with proper feature scaling. The right starting point — anything more complex has to beat it before it earns its place in production.

**Optimizations vs an out-of-the-box implementation:**
- **Time-aware split**: features computed on data ending `T-60`, label is activity in the last 60 days. Random splits leak — features and labels overlap in time. This is the single most common mistake in retention modeling and the change that most affects honest reported AUC.
- **Regularization strength tuned via cross-validation** (`LogisticRegressionCV`) instead of accepting the default `C=1.0`.
- **Threshold optimization on cost asymmetry**: default 0.5 implicitly assumes false positives and false negatives cost the same. They don't — an email is £0.20 and an at-risk customer is worth £25+. The optimal threshold is `email_cost / customer_value` ≈ 0.008, meaning *email anyone with even modest churn risk*. The default 0.5 leaves 90%+ of expected value on the table.

**Trade-offs vs alternatives:**
- *Why not jump straight to LightGBM here?* LogReg is the calibration baseline; its AUC tells you whether the problem is linear-separable on RFM. If LightGBM only beats it by 0.01, the extra complexity is fragile.

#### 5.3 DL — GRU + Attention

**Why this model:** RFM aggregates collapse a year of customer history into three numbers. A GRU reads the **trajectory** — which months were heavy, which were light, whether activity is accelerating or decelerating. Attention pooling weights informative months and is **inspectable**, which is the DL analog of feature importance.

**Optimizations:** Train/val/test split (early stopping needs the val set), early stopping on validation AUC, `ReduceLROnPlateau` learning-rate scheduler, gradient clipping for training stability, best-checkpoint restoration (returning the best val-AUC weights, not the last-epoch weights).

**Trade-offs / honest finding:** On RFM-only monthly sequences, the GRU typically only **matches** tuned logistic regression on this dataset. The architecture's value scales with sequence richness. Add product category embeddings, day-of-week features, session-level events — that's when sequence modeling materially beats tabular ML. **This section exists to document the architecture and discipline so it can ingest those features when they arrive, not to claim AUC superiority on the current data.**

### Section 6 — Churn models (who's quietly leaving?)

Retention asks *who comes back*; churn asks *who's at risk right now* so retention budget goes to the highest-impact saves. Different framing, different action.

#### 6.1 Statistical — Weibull AFT

**Why this model:** Cox gave us hazard *ratios* (relative risk). Weibull AFT gives hazard *magnitudes* — actual time-to-churn estimates per customer. Coefficients are multiplicative on time: `exp(coef) = 1.5` means the covariate extends survival by 50%. Output ships directly to the marketing team as "this customer is on track to churn in ~X days."

The Weibull `rho_` shape parameter is the diagnostic gem — for retail it's typically <1, meaning the hazard is highest *early* in tenure. This is the empirical fingerprint of the "first 30 days problem" — the largest retention lift comes from onboarding, not win-back.

**Trade-offs vs alternatives:**
- *Weibull vs Log-Normal AFT:* both are parametric. Weibull is the right pick when hazard is monotonic (which it is here — risk decreases with tenure); Log-Normal handles bathtub curves where mid-life is safest.
- *AFT vs deep survival models (DeepSurv):* DeepSurv handles non-linear effects but loses the closed-form time-to-event estimate that makes AFT directly actionable.

#### 6.2 ML — LightGBM + SHAP (tuned)

**Why this model:** Production-grade upgrade over logistic regression. Gradient-boosted decision trees handle non-linearity and feature interactions natively (`high recency AND low frequency` is a stronger signal than either alone — LogReg misses this without explicit interaction features). SHAP makes the model explainable per-customer, which is required for marketing teams to trust and act on individual scores.

**Optimizations vs an out-of-the-box implementation:**
- **`RandomizedSearchCV` hyperparameter tuning** (n_estimators, learning_rate, num_leaves, subsample, colsample_bytree, min_child_samples) — accepting LightGBM defaults typically leaves 0.02-0.04 AUC on the table.
- **Stratified CV during search** — class imbalance (most active customers don't churn) makes plain k-fold unstable on AUC.
- **Engineered interaction features** (recency × frequency ratio, revenue-per-day) — gradient boosting *can* find these on its own, but giving them explicitly tightens the model and makes SHAP attributions cleaner.

**Trade-offs vs alternatives:**
- *LightGBM vs XGBoost for churn:* roughly comparable AUC; LightGBM is faster to train at this scale (leaf-wise growth, histogram binning). Either is a defensible production choice.
- *LightGBM vs CatBoost:* CatBoost has cleaner native categorical handling. With one categorical feature (country, encoded as `is_uk`), it's not worth the dependency change.
- *Tree-based vs Neural net:* tree models are more sample-efficient on tabular data at this size. Below ~100K rows, gradient boosting almost always wins on tabular features. This is a finding of the project, not an opinion.

#### 6.3 DL — Transformer Encoder

**Why this model:** Self-attention weights events differently per customer — RFM cannot do this. The architecture's natural use case is journey modeling once the data carries event-level richness.

**Optimizations:** Train/val/test split (was train/test only in earlier iterations), early stopping on val AUC + best-checkpoint restoration, `ReduceLROnPlateau`, gradient clipping.

**Honest finding:** On RFM-only monthly sequences, transformer is at parity ±0.02 AUC with LightGBM. **Architecture is the foundation; features are what make it dominate.** For this dataset and feature set, LightGBM is the right production model. The transformer is documented so the upgrade is one feature-engineering sprint away — not a re-architecting project.

### Section 7 — LTV models (how much will they spend?)

The most directly business-actionable target — LTV sets acquisition cost ceilings, retention budgets, and VIP tier thresholds.

#### 7.1 Statistical — BG/NBD + Gamma-Gamma

**Why this model:** Canonical probabilistic LTV for non-contractual settings. **BG/NBD** predicts the *number* of future transactions (Fader, Hardie & Lee, 2005); **Gamma-Gamma** predicts the *monetary value* per transaction. Their product is expected revenue. The output is calibrated — predicted totals match actual totals — which makes it the right input for budget allocation.

**Optimizations:** Validates the Gamma-Gamma `corr(frequency, monetary_value) < 0.1` assumption explicitly before fitting. If it fails, predicted CLV is biased and the regression-based fallback (Section 7.2) is the right choice instead.

**Trade-offs vs alternatives:**
- *BG/NBD vs Pareto/NBD:* BG/NBD is computationally cheaper and assumes customers churn after a transaction (more realistic for retail than Pareto/NBD's continuous-time churn). Empirically equivalent for forecast accuracy.
- *Probabilistic vs ML LTV:* probabilistic models are calibrated but ignore non-RFM features (country, item diversity, seasonality). XGBoost picks those up — Section 7.2 is the complement, not the replacement.

#### 7.2 ML — XGBoost Regression (tuned)

**Why this model:** BG/NBD encodes a specific stochastic process. XGBoost makes no such assumption — it learns `(features) → future_revenue` from data, including features BG/NBD ignores. Typically wins on **ranking** (Spearman) by 0.05-0.10 over BG/NBD; BG/NBD wins on **calibration** (predicted totals = actual totals).

**Optimizations:**
- **`RandomizedSearchCV`** over n_estimators, max_depth, learning_rate, subsample, colsample_bytree.
- **Log-transformed target** — revenue is heavy-tailed; MSE on raw scale wastes model capacity fitting the long tail rather than the modal customer.
- **Spearman as primary metric** — for budget allocation, ranking matters more than absolute prediction. R² on heavy-tailed targets is structurally low and misleading.

**Trade-offs vs alternatives:**
- *XGBoost vs Quantile regression:* quantile regression gives prediction intervals, useful for risk-adjusted budgets. Worth adding in production for top-decile customers.
- *XGBoost vs BG/NBD:* use both. Take XGBoost's ranking and BG/NBD's calibration.

#### 7.3 DL — LSTM with proper training discipline

**Why this model:** Same intuition as the churn transformer. LSTMs are competitive with transformers on short sequences and faster to train.

**Optimizations:** Train/val/test split, early stopping, LR scheduler, gradient clipping.

**Honest finding:** Within 0.02 Spearman of XGBoost on RFM-only sequences. Gradient boosting is more sample-efficient on tabular at this size. The LSTM's value emerges with weekly granularity, product category sequences, channel events.

### Section 8 — Robust evaluation: CV + time-based holdout

**Why this section:** Every prior section used a single random 75/25 customer split. That's vulnerable to two failure modes:

1. **Lucky/unlucky split variance** — a single split's AUC can swing ±0.02 on this dataset.
2. **Temporal optimism** — random splits leak across time when feature distributions drift, inflating reported performance vs production.

**The two evaluations:**
- **Stratified 5-fold CV on customers** — gives mean AUC ± std (variance estimate).
- **Time-based holdout** — train on data ending `T-120`, test on the latest 60 days. Closest analog to deployed behavior.

**The deployment-honesty check** is the **CV − Holdout gap**. <0.02 → CV is honest about deployment. >0.05 → CV is overstating, usually due to temporal drift. Reporting both is the discipline.

### Section 9 — Business action layer (the actual deliverable)

Combines three signals per customer into a `customer_action_list.csv`:

1. **Predicted 60-day CLV** (BG/NBD + Gamma-Gamma) — value
2. **Churn probability** (tuned LightGBM) — risk
3. **RFM segment + K-Means cluster** — identity

Each customer gets a **value tier** × **risk tier** placement, a **recommended action**, and an **expected save value** (`P(churn) × CLV × LIFT − email_cost`). The marketing team works the grid top-down, top-row first, until the marginal expected save falls below cost-per-touch.

**The Top 30 priority list** = High-value × High-risk customers, sorted by expected save value. This is the immediate retention queue.

---

## Headline analysis of results

- **Recency dominates retention prediction across every model family** — appears as the strongest negative coefficient in LogReg, the largest hazard-reducing covariate in Cox, the top SHAP feature in LightGBM, and the right-tail bias in GRU attention weights. Convergent evidence across statistical, ML, and DL views. *Marketing implication: recency-based triggers are the single highest-ROI retention mechanism.*
- **Tuned LightGBM is the right production model for churn at this data size** — 0.02-0.05 AUC over LogReg through interaction-based splits, fast retraining, SHAP-explainable per-customer.
- **Deep learning is at parity with ML on this dataset and feature set.** This is a finding, not a failure — it tells the data team that the next investment is in **richer features** (product categories, session events, channels), not bigger models.
- **The Pareto effect is severe**: the top decile of predicted CLV captures 50-70% of expected forward revenue. Broadcast retention is mathematically wasteful — value-tier-aware spending is 5-10× more efficient.
- **Threshold optimization is the highest-leverage tuning step in the entire pipeline.** Moving from default 0.5 to cost-aware 0.008 changes nothing about the model and changes everything about the campaign economics. This is the cheapest, biggest win in the project.
- **CV vs time-based holdout gap quantifies deployment-honesty.** Reporting only CV AUC is the most common reason production model performance disappoints — random splits leak across time when seasonality drives features.

---

## Production recommendations (what changes at scale)

| Area | Current implementation | Production recommendation |
|---|---|---|
| **Features** | RFM aggregates from monthly granularity | Add product category embeddings, channel mix, session events, support contact history — this is what makes DL beat tree models |
| **Cadence** | One-shot snapshot | Daily batch scoring of churn probability + monthly retraining; CLV predictions refreshed weekly |
| **Threshold** | Static `email_cost / customer_value` | Calibrated against historical campaign A/B test data per segment — Champions are likely less responsive to discounts than At Risk |
| **Lift assumption** | 20% generic | Stratified by segment, re-estimated each campaign cycle |
| **Action grid** | Static 3×3 value × risk | Add **time-to-churn from Weibull AFT** as a third axis to prioritize *when* to intervene within High-risk |
| **Model governance** | Single CV + holdout | Backtest at multiple historical cutoffs; monitor drift on top-5 SHAP features; alert on AUC degradation > 0.03 |
| **Explainability for marketing** | Notebook-level SHAP plots | Per-customer "why" string in the action list (top-3 SHAP contributors as plain English) |

---

## Tech stack

- **Core:** Python 3, pandas, NumPy, matplotlib, seaborn
- **Statistical:** `lifelines` (KM, Cox, Weibull AFT), `lifetimes` (BG/NBD, Gamma-Gamma), `statsmodels`
- **ML:** scikit-learn (LogReg, K-Means, PCA, CV), `lightgbm`, `xgboost`, `shap`
- **DL:** PyTorch (GRU + Attention, Transformer Encoder, LSTM)
- **Tuning:** `RandomizedSearchCV` with `StratifiedKFold`

---

## Dataset citation

Chen, D. (2019). Online Retail II. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/502/online+retail+ii

---
