# Customer Analytics on Online Retail II
### Retention × Churn × LTV — turning customer data into a prioritized action queue

A production-shaped customer analytics project on the **Online Retail II** dataset (UCI — UK gift retailer, 525,461 raw transactions, Dec 2009 → Dec 2010). The deliverable isn't a model; it's a **prioritized customer list** with recommended actions and expected ROI. The models are how we get there.

After cleaning: **407,664 transactions, 4,312 customers, 19,213 invoices, £8.83M revenue** — the working cohort for every model below.

---

## TL;DR — what the data said

- **Top 30 priority customers identified** with £11,735 in net expected save value vs £6 in email cost — a **~1,956× ROI** on the highest-priority retention queue.
- **Logistic regression beat tuned LightGBM on churn** (CV AUC **0.721 ± 0.014** vs **0.689 ± 0.016**) — counter-intuitive, driven by small-data dynamics. *Ship the simpler model.*
- **All four retention/churn classifiers converged to AUC 0.72–0.74** (LogReg 0.742, LightGBM 0.733, GRU 0.730, Transformer 0.724). RFM-only features have a discrimination ceiling at ~0.74 on this dataset — bigger architectures don't break through it.
- **LSTM nudged XGBoost on LTV ranking** (Spearman **0.497 vs 0.448**) — the one place sequence modeling moved the needle. Worth tracking as feature richness grows.
- **Top 20% of customers = 73.5% of revenue. Top 1% = 28.7%.** Pareto is severe and the entire reason per-customer LTV is worth modeling.
- **Time-based holdout AUC matched CV AUC within 0.005** for both models — features generalize cleanly across time. The time-aware split worked.

---

## Why retention, churn, and LTV must be modeled *together*

Each answers a different question, and each — alone — mis-allocates budget.

| Question | What it answers alone | What it misses alone |
|---|---|---|
| **Retention** — who comes back? | Forward probability of activity | Says nothing about how *valuable* coming back is |
| **Churn** — who's quietly leaving? | Risk ranking of active customers | Says nothing about whether the customer is worth saving |
| **LTV** — who's worth investing in? | Future revenue per customer | Says nothing about whether they're at risk *right now* |

The action only makes sense when all three sit on the table at once:

- **High churn-risk × high LTV** → personal outreach from an account manager (worth £100s).
- **High churn-risk × low LTV** → single automated email (cheap, low EV).
- **Low churn-risk × high LTV** (Champions) → thank-you, never a discount. *A discount campaign aimed at Champions is pure margin loss.*
- **Low churn-risk × low LTV** → ignore. Acquisition replaces them more efficiently than retention does.

Predictive models give scores; segmentation gives names. **Combining value (LTV) with risk (churn) on top of a named segment (RFM/cluster) is what turns scores into a campaign brief.** No single output gets you there.

---

## Why segmentation needs *both* RFM rules and K-Means

| Method | Strength | Blind spot |
|---|---|---|
| **RFM quintile rules** | Reproducible across quarters; "Champions" means the same thing every time; marketing teams already speak this language | Quintile boundaries are arbitrary — a customer can flip segments on a one-quintile shift with no business meaning |
| **K-Means on log RFM** | Lets data find natural groupings; handles the heavy-tailed shape of revenue when fed log features | Cluster identities drift between training runs; less directly explainable to non-technical stakeholders |

**Result on this dataset:** RFM gave 9 segments; K-Means picked **k=3** (silhouette = 0.412). The two views agree on the obvious cases (Champions overlap heavily with the High-value/Active cluster) and disagree at the boundaries — which is exactly where you want a data-driven view to overrule the rules. Customers in the disagreement zone get flagged for closer review.

---

## Modeling philosophy: three families, three jobs

| Family | What it's *for* | Role here |
|---|---|---|
| **Statistical** (Cox PH, Weibull AFT, BG/NBD + Gamma-Gamma) | Interesting findings, calibrated probabilities, parametric forecasts | Explains *why* customers churn (hazard ratios, AFT time-multipliers). Generates the executive narrative. |
| **Machine Learning** (LogReg, LightGBM, XGBoost — all tuned) | Production scoring | Best discrimination/regression on tabular RFM. Tunable, fast to retrain, SHAP-explainable. **What gets deployed.** |
| **Deep Learning** (GRU + Attention, Transformer, LSTM) | Sequence-aware modeling | At RFM-only feature density, parity with ML on this dataset (LSTM nudged XGBoost on LTV). **The architecture documented in code so it's ready when event-level features arrive.** |

In short: **statistics is for what's interesting, ML is for what's in production, deep learning is for what comes next.** With ~4K customers and RFM-only inputs, deep learning ties tabular ML — and that's an important business finding, not a project failure. It tells the data team that the next investment should go into **richer features**, not bigger models.

---

## Headline findings — what each model actually said

### Data quality
- 525,461 raw rows → **407,664 clean rows (77.6% retained)**.
- **20.5% of raw rows had missing Customer ID** — the largest data quality issue. These are typically guest checkouts or B2B bulk imports and can't be modeled at customer level.
- 2.0% cancellation invoices, 2.4% returns dropped.

### EDA — the Pareto evidence
- **Top 20% of customers = 73.5% of revenue. Top 1% = 28.7%.** Even more concentrated than the textbook 80/20.
- Revenue dominated by the UK home market — international expansion structurally untapped.
- Holiday seasonality (Nov–Dec spike) — the reason every classifier uses a time-aware split.

### RFM segmentation
The 9-segment view, ranked by revenue share:

| Segment | % of customers | % of revenue | Avg revenue per customer | Action |
|---|---:|---:|---:|---|
| Champions | 21.1% | **64.1%** | £6,219 | Thank, never discount |
| Loyal Customers | 11.3% | 10.1% | £1,836 | Maintain |
| At Risk | 5.1% | **7.8%** | £3,112 | Highest retention priority |
| Needs Attention | 18.8% | 5.3% | £579 | Re-engage |
| Potential Loyalists | 9.0% | 5.2% | £1,171 | Convert to Loyal |
| Hibernating | 4.4% | 2.6% | £1,200 | Modest win-back only |
| Lost | 19.4% | 2.2% | £227 | Ignore |
| New Customers | 8.3% | 2.0% | £495 | Onboarding sequence |
| Can't Lose Them | 2.5% | 0.7% | £616 | Personal save |

**Reading:** *At Risk* customers are 5.1% of the base but 7.8% of revenue — over-indexed and the textbook retention save target. *Champions* are over a fifth of customers and almost two-thirds of revenue.

### K-Means (k=3, silhouette = 0.412)

| Cluster | % of customers | % of revenue | Avg revenue | Avg recency | Avg frequency |
|---|---:|---:|---:|---:|---:|
| **High-value / Active** | 31.6% | **81.2%** | £5,268 | 34 days | 10.1 orders |
| Low-value / Dormant (mid) | 46.3% | 14.3% | £631 | 54 days | 2.1 orders |
| Low-value / Dormant (deep) | 22.1% | 4.6% | £423 | 250 days | 1.4 orders |

**Reading:** ~32% of the base is the entire revenue engine. The data structure is genuinely 3 groups, not 4 — silhouette picked k=3 over k=4 cleanly.

### Retention models — *will they come back in the next 60 days?*

| Model | AUC (test) | What it added |
|---|---:|---|
| Cox PH | concordance **0.744** | Quantified *why* — log_frequency HR = **0.23** (each unit log_frequency cuts hazard by 77%) |
| Logistic Regression (tuned) | **0.742** | Production-ready baseline; interpretable coefficients |
| GRU + Attention | 0.730 | Sequence architecture; tied LogReg, did not beat it |

**The strongest LogReg standardized coefficients (positive = retention-protective):**

| Feature | Std. coefficient | Odds ratio |
|---|---:|---:|
| n_unique_items | **+0.43** | 1.54 |
| log_frequency | +0.37 | 1.44 |
| log_revenue | +0.20 | 1.22 |
| recency | −0.18 | 0.83 |
| aov | −0.16 | 0.86 |

**Counter to typical retention-modeling intuition: recency is *fourth*, not first.** The dominant signal here is **product diversity** (`n_unique_items`) followed by **order frequency**. Customers who shop *broadly* across the catalog come back; transaction-volume customers come back; recent customers come back. Big-basket customers (high AOV) actually retain *worse*, controlling for the rest — likely a one-off-purchase pattern.

**Threshold optimization — the highest-leverage tuning step in the pipeline:**
- Cost-aware threshold = `email_cost / customer_value` = **£0.20 / £25 = 0.008**.
- The default 0.5 leaves >90% of expected save value on the table. Same model, same coefficients — the threshold change alone reorders the campaign economics.

**GRU finding:** Best val AUC 0.707, test AUC 0.730 — within rounding of LogReg. Early stopping triggered at epoch 17. **The architecture didn't add discrimination on RFM-only sequences — by design. Its value scales with sequence richness (product categories, channels, sessions).**

### Churn models — *who's quietly leaving?*

| Model | AUC | Notes |
|---|---:|---|
| Weibull AFT | — | Time-to-churn estimates per customer |
| Logistic Regression (tuned) | CV **0.721 ± 0.014**, Holdout **0.716** | Winner on robust eval |
| **LightGBM (tuned)** | CV **0.689 ± 0.016**, Holdout **0.689** | Lost to LogReg despite hyperparameter search |
| Transformer | Test 0.724 | Tied with LightGBM on test split |

**The big finding: tuned LightGBM lost to logistic regression on robust evaluation.** This contradicts the default assumption that gradient boosting beats LogReg on tabular data. Why it lost here:

1. **Small dataset** — ~4K customers, ~2.4K in any train fold. LightGBM's CV-tuned 53 leaves overfits at this size.
2. **Few features** — six RFM features. Tree splits on six features quickly run out of useful interactions.
3. **Strong regularization helped LogReg** — best `C = 0.046` (heavy shrinkage). The signal is essentially linear in log-space; shrinkage exploits that.
4. **The CV–Holdout gap was 0.005 (LogReg) and 0.001 (LightGBM)** — features generalize across time; the gap isn't temporal drift, it's genuine model difference.

**Production implication: don't auto-default to gradient boosting on small tabular data.** Run LogReg as the actual baseline, not the strawman, before reaching for trees.

**Weibull AFT — time-to-churn risk segments:**

| Risk segment | Median predicted days-to-churn | Customers |
|---|---:|---:|
| High risk | **157 days** | 823 |
| Med-high | 247 days | 822 |
| Med-low | 425 days | 822 |
| Low risk | 8,145 days (essentially permanent) | 823 |

**AFT log_frequency coefficient = 1.00, exp(coef) = 2.73** — each unit increase in log_frequency *multiplies* expected time-to-churn by **2.73×**. Frequency is the dominant time-extending covariate; AOV and country had no significant effect.

**Survival fundamentals from KM:**
- Survival at 30 days: **97.4%**
- Survival at 60 days: **94.3%**
- Survival at 180 days: **77.0%**
- Overall churn rate (recency > 60 days at snapshot): **27.8%**

The 60-day window may be *too short* for this dataset — most customers haven't had time to churn at that horizon. A 180-day window would give a sharper class signal. Worth a follow-up experiment.

### LTV models — *how much will they spend in the next 60 days?*

| Model | Spearman | MAE | Notes |
|---|---:|---:|---|
| BG/NBD + Gamma-Gamma | — | — | Calibrated; top decile = **53% of predicted CLV** |
| XGBoost (tuned) | 0.448 | £513 | R² = -0.003 on raw scale (heavy-tailed) |
| **LSTM** | **0.497** | £548 | First place sequence modeling actually nudged ahead |

**The Pareto in forward predictions: top decile of customers = 53% of predicted 60-day revenue.** Same Pareto shape as historical revenue, preserved into the forecast — confirming that retention spend should be top-decile-first.

**Honest assumption check:** Gamma-Gamma assumes `corr(frequency, monetary) < 0.10`. **The actual correlation is 0.187 — the assumption is violated.** CLV predictions for high-frequency customers are likely upward-biased. This is why XGBoost (no parametric assumption) is paired with BG/NBD — XGBoost catches the bias the probabilistic model can't.

**LSTM beat XGBoost on Spearman (0.497 vs 0.448)** — modest but real. The trajectory of monthly behavior carries genuine ranking signal that aggregate RFM compresses away. **This is the data point that justifies the DL track in this project: at parity on tabular ranking targets, slight win on temporal ranking.** As feature richness grows, the gap should widen in the LSTM's favor.

**Top-CLV customers (BG/NBD + Gamma-Gamma):**
- Top customer: ID 18102, **predicted £50,677** in 60-day CLV (42 historical orders, avg value £8,420).
- The top 10 alone account for ~£213K in predicted forward revenue — over 10% of total predicted from <0.25% of customers.

### Robust evaluation — CV vs time-based holdout

| Model | CV Mean AUC | CV Std | Time-Holdout AUC | Gap |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.721 | 0.014 | 0.716 | **0.005** |
| LightGBM | 0.689 | 0.016 | 0.689 | **0.001** |

**Both gaps are <0.01 — features generalize cleanly across time.** Time-aware feature/label splits did their job: no temporal leakage inflating CV. This is the deployment-honesty check, and it passed.

### The action layer — value × risk grid

| | Low risk | Mid risk | High risk |
|---|---:|---:|---:|
| **High value** | 785 customers · £36K total EV | **415 customers · £58K total EV** ← largest pocket | 238 customers · £33K total EV |
| **Mid value** | 635 · £10K | 394 · £10K | 408 · £11K |
| **Low value** | 17 · £0 | 628 · £0 | 792 · £0 |

**Subtle but actionable finding:** the largest *total* expected-save pocket is **High value × Mid risk (£58K)**, not High value × High risk (£33K). Mid-risk has more customers, and per-customer EV is actually slightly higher (£141 vs £140) because higher CLV compensates for lower churn probability. *The campaign shouldn't just chase the highest-risk tier — it should saturate the high-value mid-risk pocket too.*

**Top 30 priority list (High value × High risk, ranked by expected save):**
- Total expected save value: **£11,741**
- Email cost (30 × £0.20): £6
- **Net expected return: £11,735** (~1,956× ROI)
- Top customer: ID 13902 (Denmark, At Risk × High-value/Active) — £7,094 predicted CLV, 80% churn probability, **£1,137 expected save value from a single intervention**.

International customers are over-represented in the top 30 (Denmark, Singapore, Sweden, Channel Islands, Switzerland, Italy alongside UK), suggesting that **international high-value customers may be under-served by current retention programs** designed around the UK home market.

---

## Pipeline walkthrough — model rationale, optimizations, trade-offs

Every section below follows the same structure: *Why this model · What was optimized vs an out-of-the-box implementation · Trade-off vs the obvious alternative.*

### 1 — Data quality audit

Drop missing Customer IDs (can't model at customer level), cancellation invoices (`Invoice` starts with `C`, double-counts), non-positive quantity/price (modeling artifacts). Single source of truth `df_clean` feeds every downstream model. **Optimization:** decisions are documented inline so the cohort is reproducible — in production this becomes a versioned data contract.

### 2 — EDA, business lens

Pareto check (73.5% from top 20%), geographic concentration (UK), seasonality (Nov–Dec spike). Drives the time-aware split decision in every classifier.

### 3 — Customer feature engineering

RFM + tenure + AOV + n_unique_items + cadence. RFM features dominate repeat-purchase prediction in non-contractual retail (Fader & Hardie, 2005), so they anchor every downstream model.

### 4 — Segmentation (RFM + K-Means)

**Optimizations:**
- K-Means fit on **log-transformed** frequency and revenue — raw values are heavy-tailed and break K-Means' implicit equal-variance assumption.
- `k` chosen via **combined elbow + silhouette**, not hard-coded — picked k=3 with silhouette 0.412.
- 2D PCA visualization confirms cluster separation visually.

**Trade-offs:**
- *Why not GMM?* Soft assignments hurt actionability. K-Means' hard cluster labels are the marketing asset.
- *Why not hierarchical clustering?* Doesn't scale past ~10K customers. Centroid interpretability is also weaker.

### 5 — Retention models

#### 5.1 Cox Proportional Hazards (statistical)
**Why:** Non-parametric survival. Quantifies which features extend or shorten customer lifespan as **hazard ratios**. Concordance 0.744 — useful, mid-tier.

**Optimizations:** Log-transform frequency and AOV before fitting (heavy tails distort coefficients); UK collapsed from 38 countries to a single indicator to manage degrees of freedom.

**Trade-off:** *Cox vs random survival forest:* RSF handles non-linearities natively but loses the hazard-ratio interpretation that makes Cox the right tool for executive reporting.

#### 5.2 Logistic Regression (tuned ML)
**Why:** Right baseline for binary retention. Interpretable, fast, well-calibrated probabilities.

**Optimizations:**
- **Time-aware split**: features computed on data ending T−60, label = activity in last 60 days. Random splits leak.
- **Regularization via `LogisticRegressionCV`** — best C = 0.046 (strong shrinkage).
- **Cost-aware threshold tuning** — 0.008, not 0.5. The biggest economic lever in the entire pipeline.

**Result:** Test AUC 0.742 — the best of any retention/churn classifier on robust evaluation.

#### 5.3 GRU + Attention (deep learning)
**Why:** RFM aggregates collapse trajectory into static numbers. A GRU reads the *trajectory* of monthly behavior. Attention is the inspectable analog of feature importance.

**Optimizations:** Train/val/test split (early stopping needs val), early stopping on val AUC (triggered at epoch 17), `ReduceLROnPlateau`, gradient clipping, best-checkpoint restoration.

**Honest finding:** Test AUC 0.730 — within rounding of LogReg. The architecture's value scales with sequence richness, not architectural complexity at this feature set.

### 6 — Churn models

#### 6.1 Weibull AFT (statistical)
**Why:** Cox gives ratios; AFT gives time-to-event magnitudes ("this customer churns in ~X days"). Coefficients are multiplicative on time.

**Result:** log_frequency exp(coef) = 2.73 — each unit log_frequency multiplies expected time-to-churn by 2.73×. Risk segments span 157 days (high risk) to 8,145 days (low risk).

**Trade-off:** *AFT vs DeepSurv:* DeepSurv handles non-linear effects but loses the closed-form time-to-event estimate that makes AFT directly actionable.

#### 6.2 LightGBM + SHAP (tuned ML)
**Why:** The expected production upgrade — gradient-boosted trees with interaction handling and SHAP explainability.

**Optimizations:** `RandomizedSearchCV` over 8 hyperparameters, stratified CV, engineered interaction features (recency × frequency ratio, revenue-per-day).

**Result:** **Lost to LogReg on robust evaluation** (CV AUC 0.689 vs 0.721). At ~4K customers and 6 features, the model space LightGBM unlocks doesn't pay off. SHAP explainability is still a win — but on this dataset, the AUC argument fails. *Ship LogReg. Re-evaluate when feature count crosses ~30 or row count crosses ~50K.*

**Trade-offs:**
- *LightGBM vs XGBoost:* roughly equivalent AUC; LightGBM faster to train at this scale.
- *LightGBM vs CatBoost:* CatBoost has cleaner native categorical handling — not worth the dependency change with one categorical feature.

#### 6.3 Transformer Encoder (deep learning)
**Why:** Self-attention weights events differently per customer. Architecture documented for the upgrade path.

**Result:** Test AUC 0.724 — tied with LightGBM. **Architecture is the foundation; features are what make it dominate.**

### 7 — LTV models

#### 7.1 BG/NBD + Gamma-Gamma (statistical)
**Why:** Canonical probabilistic LTV for non-contractual settings. Calibrated — predicted totals match actual totals.

**Result:** Top decile = 53% of predicted CLV. **Assumption check failed: corr(frequency, monetary) = 0.187, above the 0.10 threshold.** CLV is likely upward-biased on high-frequency customers — XGBoost (Section 7.2) is the necessary complement, not just an alternative.

**Trade-off:** *BG/NBD vs Pareto/NBD:* BG/NBD is computationally cheaper and assumes customers churn after a transaction (more realistic for retail). Empirically equivalent for forecast accuracy.

#### 7.2 XGBoost (tuned ML)
**Why:** No parametric assumption — learns `(features) → future_revenue` from data. Captures non-RFM features (country, item diversity, seasonality) that BG/NBD ignores.

**Optimizations:** `RandomizedSearchCV` over 8 hyperparameters; **log-transformed target** (revenue is heavy-tailed); **Spearman as primary metric** (ranking matters more than absolute prediction for budget allocation).

**Result:** Spearman 0.448, MAE £513. R² = -0.003 on raw scale — a structural artifact of heavy tails, not a model failure. Spearman is the metric that matters.

#### 7.3 LSTM (deep learning)
**Why:** Sequence-aware LTV. LSTM is faster to train than transformer on short sequences and competitive in performance.

**Optimizations:** Train/val/test split, early stopping on val MSE (triggered at epoch 27), LR scheduler, gradient clipping.

**Result:** **Spearman 0.497 — beat XGBoost (0.448).** The headline DL win in the project. Trajectory information adds genuine ranking signal beyond aggregated RFM.

### 8 — Robust evaluation

5-fold stratified CV + time-based holdout (train T−180→T−60, test T−60→T). The CV–Holdout gap measures temporal drift. Both models gapped <0.01, confirming the time-aware split closed the leakage.

### 9 — Action layer

Per-customer fusion of (RFM segment, K-Means cluster, predicted 60-day CLV, churn probability). Value tier × risk tier 3×3 grid. Recommended action per cell. `expected_save_value = churn_prob × CLV × LIFT − email_cost`. The Top 30 list is the immediate retention queue; the heatmap is the budget envelope for the broader campaign.

---

## Production recommendations

| Area | Current | Production upgrade |
|---|---|---|
| **Model selection for churn** | LightGBM tuned | **Ship LogReg.** Smaller, faster, more accurate at this size. Re-test boosting only after features cross ~30 or rows cross ~50K. |
| **Features** | RFM aggregates, monthly granularity | Add product category embeddings, channel mix, session events, support contacts — *the actual reason DL would beat ML* |
| **Cadence** | One-shot snapshot | Daily batch churn scoring; monthly retraining; weekly CLV refresh |
| **Threshold** | Static cost-aware (0.008) | Calibrated against historical campaign A/B data per segment |
| **Lift assumption** | Generic 20% | Segment-stratified, re-estimated each campaign cycle (Champions less responsive to discounts than At Risk) |
| **Action grid** | Static 3×3 value × risk | Add **time-to-churn from Weibull AFT** as a third axis to prioritize *when* to intervene within the High-risk tier |
| **CLV correction** | BG/NBD ignoring f-m correlation = 0.187 | Stratify Gamma-Gamma by frequency tier, or use XGBoost-only when assumption fails |
| **Window** | 60-day labels (94% retention rate — class imbalance is mild but signal is weak) | Test 90/180-day windows; longer horizons may yield sharper class signal |
| **Geography** | Single global model | International high-value customers over-represented in priority list — possibly under-served. Consider regional models |
| **Explainability** | Notebook-level SHAP plots | Per-customer "why" string in the action list (top-3 SHAP contributors as plain English) |
| **Governance** | Single CV + holdout | Backtest at multiple historical cutoffs; monitor drift on top-5 SHAP features; alert on AUC degradation > 0.03 |

---

## Tech stack

- **Core:** Python 3, pandas, NumPy, matplotlib, seaborn
- **Statistical:** `lifelines` (KM, Cox, Weibull AFT), `lifetimes` (BG/NBD, Gamma-Gamma)
- **ML:** scikit-learn (LogReg, K-Means, PCA, CV), `lightgbm`, `xgboost`, `shap`
- **DL:** PyTorch (GRU + Attention, Transformer Encoder, LSTM)
- **Tuning:** `RandomizedSearchCV` with `StratifiedKFold`

## Repository structure

```
.
├── Online_Retail_II_Optimized.ipynb   # Full pipeline, 70 cells
├── data/
│   └── online_retail_II.xlsx          # UCI dataset
├── outputs/
│   └── customer_action_list.csv       # Deliverable: prioritized customers
└── README.md
```

## How to run

```bash
pip install -r requirements.txt
jupyter lab Online_Retail_II_Optimized.ipynb
```

Run cells top-to-bottom. Section dependencies are linear: each modeling section reads from `df_clean` and `customer_df` produced in Sections 1–3. The action layer (Section 9) reads from segmentation, LTV, and churn outputs.

## Dataset citation

Chen, D. (2019). *Online Retail II.* UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/502/online+retail+ii

---

*Built by **TechNick Analytics** — predictive health and customer analytics consulting.*
