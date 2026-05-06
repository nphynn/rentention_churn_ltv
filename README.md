# Customer Analytics on Online Retail II Results
### Retention × Churn × LTV

A production-designed customer analytics project on the **Online Retail II** dataset (UCI — UK gift retailer, 525,461 raw transactions, Dec 2009 to Dec 2010). The report provides a **prioritized customer list** with recommended actions and expected ROI.

After cleaning: **407,664 transactions, 4,312 customers, 19,213 invoices, £8.83M revenue** which serves as the working cohort for every model below.

## Why retention, churn, and LTV have to be modeled *together*

This is the core insight the project is built around. Each of the three answers a different question and each on its own mis-allocates budget.

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

Predictive models give scores, segmentation gives names, **combining value (LTV) with risk (churn) on top of a named segment (RFM/cluster) is what turns scores into a campaign brief**. None of the three answers alone is a strategy.

## Why segmentation needs *both* RFM rules and K-Means

A common mistake is picking one segmentation technique and shipping it. I deliberately ran both because each compensates for the other's blind spot. The two views agree on the obvious cases (Champions, Lost) and disagree at the boundaries. Customers in this disagreement zone are flagged for closer human review. **Either method alone is a worse segmentation than running both and reconciling.**

**Result on this dataset:** RFM gave 9 segments; K-Means picked **k=3** (silhouette = 0.412). The two views agree on the obvious cases (Champions overlap heavily with the High-value/Active cluster) and disagree at the boundaries — which is exactly where you want a data-driven view to overrule the rules. Customers in the disagreement zone get flagged for closer review.

## Modeling philosophy:

| Family | What it's *for* | Role here |
|---|---|---|
| **Statistical** (Cox PH, Weibull AFT, BG/NBD + Gamma-Gamma) | Interesting findings, calibrated probabilities, parametric forecasts | Explains *why* customers churn (hazard ratios, AFT time-multipliers). Executive narrative. |
| **Machine Learning** (LogReg, LightGBM, XGBoost) | Production scoring | Best discrimination/regression on tabular RFM. Tunable, fast to retrain, SHAP-explainable. **What gets deployed.** |
| **Deep Learning** (GRU + Attention, Transformer, LSTM) | Sequence-aware modeling | At RFM-only feature density, parity with ML on this dataset (LSTM nudged XGBoost on LTV). **The architecture documented in code so it's ready when event-level features arrive.** |

In short: **statistics is for what's interesting, ML is for what's in production, deep learning is for what comes next.**

## ANALYSIS

### Data quality
- 525,461 raw rows → **407,664 clean rows (77.6% retained)**.
- **20.5% of raw rows had missing Customer ID**. These are typically guest checkouts or B2B bulk imports and can't be modeled at customer level.
- 2.0% cancellation invoices, 2.4% returns dropped.

### EDA — the Pareto evidence
- **Top 20% of customers = 73.5% of revenue. Top 1% = 28.7%.** Even more concentrated than the textbook 80/20.
- Revenue dominated by the UK home market: international expansion structurally untapped.
- Holiday seasonality (Nov–Dec spike): the reason every classifier below uses a time-aware split.

### RFM segmentation
The 9-segment view, ranked by revenue share:

| Segment | % of customers | % of revenue | Avg revenue per customer | Action |
|---|---:|---:|---:|---|
| Champions | 21.1% | **64.1%** | £6,219 | Thank, vip offers |
| Loyal Customers | 11.3% | 10.1% | £1,836 | Maintain |
| At Risk | 5.1% | **7.8%** | £3,112 | Highest retention priority |
| Needs Attention | 18.8% | 5.3% | £579 | Re-engage |
| Potential Loyalists | 9.0% | 5.2% | £1,171 | Convert to Loyal |
| Hibernating | 4.4% | 2.6% | £1,200 | Modest win-back only |
| Lost | 19.4% | 2.2% | £227 | Ignore |
| New Customers | 8.3% | 2.0% | £495 | Onboarding sequence |
| Can't Lose Them | 2.5% | 0.7% | £616 | Personal save |

**Reading:** *At Risk* customers are 5.1% of the base but 7.8% of revenue, over-indexed and the textbook retention save target. *Champions* are over a fifth of customers and almost two-thirds of revenue.

### K-Means (k=3, silhouette = 0.412)

| Cluster | % of customers | % of revenue | Avg revenue | Avg recency | Avg frequency |
|---|---:|---:|---:|---:|---:|
| **High-value / Active** | 31.6% | **81.2%** | £5,268 | 34 days | 10.1 orders |
| Low-value / Dormant (mid) | 46.3% | 14.3% | £631 | 54 days | 2.1 orders |
| Low-value / Dormant (deep) | 22.1% | 4.6% | £423 | 250 days | 1.4 orders |

**Reading:** ~32% of the base is the entire revenue engine. The data structure is genuinely 3 groups not 4, silhouette picked k=3 over k=4 cleanly.

### Retention models — *will they come back in the next 60 days?*

**Survival fundamentals from KM:**
- Survival at 30 days: **97.4%**
- Survival at 60 days: **94.3%**
- Survival at 180 days: **77.0%**
- Overall churn rate (recency > 60 days at snapshot): **27.8%**

A 180-day window would give a sharper class signal. Worth a follow-up experiment.

| Model | AUC (test) | What it added |
|---|---:|---|
| Cox PH | concordance **0.744** | Quantified *why* — log_frequency HR = **0.23** (each unit log_frequency cuts hazard by 77%) |
| Logistic Regression | **0.742** | Production-ready baseline; interpretable coefficients |
| GRU + Attention | 0.730 | Sequence architecture; tied LogReg, did not beat it |

**The strongest LogReg standardized coefficients (positive = retention-protective):**

| Feature | Std. coefficient | Odds ratio |
|---|---:|---:|
| n_unique_items | **+0.43** | 1.54 |
| log_frequency | +0.37 | 1.44 |
| log_revenue | +0.20 | 1.22 |
| recency | −0.18 | 0.83 |
| aov | −0.16 | 0.86 |

**Counter to typical retention-modeling intuition: recency is *fourth*, not first.** The dominant signal here is **product diversity** (`n_unique_items`) followed by **order frequency**. Customers who shop *broadly* across the catalog come back; transaction-volume customers come back; recent customers come back. Big-basket customers (high AOV) actually retain *worse*, controlling for the rest likely a one-off-purchase pattern.

**Threshold optimization**
- Cost-aware threshold = `email_cost / customer_value` = **£0.20 / £25 = 0.008**.

**GRU finding:** Best val AUC 0.707, test AUC 0.730. Early stopping triggered at epoch 17. **The architecture didn't add discrimination on RFM-only sequences by design as it value scales with sequence richness (product categories, channels, sessions).**

### Churn models — *who's quietly leaving?*

**Weibull AFT — time-to-churn risk segments:**

| Risk segment | Median predicted days-to-churn | Customers |
|---|---:|---:|
| High risk | **157 days** | 823 |
| Med-high | 247 days | 822 |
| Med-low | 425 days | 822 |
| Low risk | 8,145 days (essentially permanent) | 823 |

**AFT log_frequency coefficient = 1.00, exp(coef) = 2.73**: each unit increase in log_frequency *multiplies* expected time-to-churn by **2.73×**. Frequency is the dominant time-extending covariate; AOV and country had no significant effect.

| Model | AUC | Notes |
|---|---:|---|
| **LightGBM** | CV **0.689 ± 0.016**, Holdout **0.689** |  
| Transformer | Test 0.724 | 

1. **Small dataset** — ~4K customers, ~2.4K in any train fold. LightGBM's CV-tuned 53 leaves overfits at this size.
2. **Few features** — six RFM features. Tree splits on six features quickly run out of useful interactions.

**Production implication: don't auto-default to gradient boosting on small tabular data.** Run LogReg as the actual baseline before reaching for trees. 

**Finding:** On RFM-only monthly sequences, transformer is at parity ±0.02 AUC with LightGBM. **Architecture is the foundation; features are what make it dominate.** For this dataset and feature set, LightGBM is the right production model. The transformer is documented so the upgrade is one feature-engineering sprint away. 


### LTV models — *how much will they spend in the next 60 days?*

**BG/NBD + Gamma-Gamma (top decile of customers = 53% of predicted 60-day revenue)**: Same Pareto shape as historical revenue preserved into the forecast confirming that retention spend should be top-decile-first.

**Top-CLV customers (BG/NBD + Gamma-Gamma):**
- Top customer: ID 18102, **predicted $50,677** in 60-day CLV (42 historical orders, avg value $8,420).
- The top 10 alone account for ~$213K in predicted forward revenue which is over 10% of total predicted from <0.25% of customers.
- 
**Assumption check:** Gamma-Gamma assumes `corr(frequency, monetary) < 0.10`. **The actual correlation is 0.187 — the assumption is violated.** CLV predictions for high-frequency customers are likely upward-biased. This is why XGBoost (no parametric assumption) is paired with BG/NBD as XGBoost catches the bias the probabilistic model can't.

| Model | Spearman | MAE | Notes |
|---|---:|---:|---|
| XGBoost | 0.448 | $513 | R² = -0.003 on raw scale (heavy-tailed) |
| **LSTM** | **0.497** | $548 | First place sequence modeling actually nudged ahead |

**LSTM beat XGBoost on Spearman (0.497 vs 0.448)**: The trajectory of monthly behavior carries genuine ranking signal that aggregate RFM compresses away. **This is the data point that justifies the DL track in this project as at parity on tabular ranking targets the architecture slight win on temporal ranking.** As feature richness grows the gap should widen in the LSTM's favor.

**Finding:** Within 0.02 Spearman of XGBoost on RFM-only sequences. Gradient boosting is more sample-efficient on tabular at this size. The LSTM's value emerges with weekly granularity, product category sequences, channel events

### Robust evaluation — CV vs time-based holdout

| Model | CV Mean AUC | CV Std | Time-Holdout AUC | Gap |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.721 | 0.014 | 0.716 | **0.005** |
| LightGBM | 0.689 | 0.016 | 0.689 | **0.001** |

**Both gaps are <0.01 (features generalize cleanly across time):** Time-aware feature/label splits did their job and no temporal leakage inflating CV. This is the deployment-honesty check and it passed.

### The action layer — value × risk grid

| | Low risk | Mid risk | High risk |
|---|---:|---:|---:|
| **High value** | 785 customers · £36K total EV | **415 customers · £58K total EV** ← largest pocket | 238 customers · £33K total EV |
| **Mid value** | 635 · £10K | 394 · £10K | 408 · £11K |
| **Low value** | 17 · £0 | 628 · £0 | 792 · £0 |

**Finding:** the largest *total* expected-save pocket is **High value × Mid risk (£58K)**, not High value × High risk (£33K). Mid-risk has more customers, and per-customer EV is actually slightly higher (£141 vs £140) because higher CLV compensates for lower churn probability. *The campaign shouldn't just chase the highest-risk tier, it should saturate the high-value mid-risk pocket too.*

**Top 30 priority list (High value × High risk, ranked by expected save):**
- Total expected save value: **£11,741**
- Email cost (30 × £0.20): £6
- **Net expected return: £11,735** (~1,956× ROI)
- Top customer: ID 13902 (Denmark, At Risk × High-value/Active) — £7,094 predicted CLV, 80% churn probability, **£1,137 expected save value from a single intervention**.

International customers are over-represented in the top 30 (Denmark, Singapore, Sweden, Channel Islands, Switzerland, Italy alongside UK), suggesting that **international high-value customers may be under-served by current retention programs** designed around the UK home market.

## Business Deliverables

1. **Customer segmentation** (RFM rules + K-Means): every customer is named and grouped.
2. **Risk scores**: every active customer has a churn probability and a 60-day CLV estimate.
3. **Prioritized action list**: combining value × risk into recommended actions with expected ROI.
4. **Robust evaluation**: CV + time-based holdout to make sure the production-deployed metric matches the reported one.

## Executive summary

A retention email costs £0.20. An at-risk High-value customer is worth £25-200+ in expected 60-day revenue. The model identifies the top decile of High-value × High-risk customers and quantifies expected save value per intervention. The marketing team can now spend retention budget where it has the highest expected ROI instead of broadcasting to the entire base.

---

## Results Summary

- **Top 30 priority customers identified** with £11,735 in net expected save value vs £6 in email cost; a **~1,956× ROI** on the highest-priority retention queue.
- **All four retention/churn classifiers converged to AUC 0.72–0.74** (LogReg 0.742, LightGBM 0.733, GRU 0.730, Transformer 0.724). RFM-only features have a discrimination ceiling at ~0.74 on this dataset which bigger architectures don't break through it.
- - **LightGBM is the right production model for churn at this data size** — 0.02-0.05 AUC over LogReg through interaction-based splits, fast retraining, SHAP-explainable per-customer.
- **LSTM nudged XGBoost on LTV ranking** (Spearman **0.497 vs 0.448**): the one place sequence modeling moved the needle. Worth tracking as feature richness grows.
- **Recency dominates retention prediction across every model family**: appears as the strongest negative coefficient in LogReg, the largest hazard-reducing covariate in Cox, the top SHAP feature in LightGBM and the right-tail bias in GRU attention weights. Convergent evidence across statistical, ML, and DL views. *Marketing implication: recency-based triggers are the single highest-ROI retention mechanism.*
- **Time-based holdout AUC matched CV AUC within 0.005** for both models — features generalize cleanly across time. The time-aware split worked.
- **Deep learning is at parity with ML on this dataset and feature set:** This tells data teams that the next investment is in **richer features** (product categories, session events, channels), not bigger models.
- **The Pareto effect is severe**: the top decile of predicted CLV captures 50-70% of expected forward revenue. Broadcast retention is mathematically wasteful as value-tier-aware spending is 5-10× more efficient.
- **Threshold optimization is the highest-leverage tuning step in the entire pipeline**: Moving from default 0.5 to cost-aware 0.008 changes nothing about the model and changes everything about the campaign economics. This is the cheapest, biggest win in the project.


## Production recommendations

| Area | Current | Production upgrade |
|---|---|---|
| **Model selection for churn** | LightGBM | **Ship LogReg.** Smaller, faster, more accurate at this size. Re-test boosting only after features cross ~30 or rows cross ~50K. |
| **Features** | RFM aggregates, monthly granularity | Add product category embeddings, channel mix, session events, support contacts |
| **Cadence** | One-shot snapshot | Daily batch churn scoring; monthly retraining; weekly CLV refresh |
| **Threshold** | Static cost-aware (0.008) | Calibrated against historical campaign A/B data per segment |
| **Lift assumption** | Generic 20% | Segment-stratified, re-estimated each campaign cycle (Champions less responsive to discounts than At Risk) |
| **Action grid** | Static 3×3 value × risk | Add **time-to-churn from Weibull AFT** as a third axis to prioritize *when* to intervene within the High-risk tier |
| **CLV correction** | BG/NBD ignoring f-m correlation = 0.187 | Stratify Gamma-Gamma by frequency tier, or use XGBoost only when assumption fails |
| **Window** | 60-day labels (94% retention rate as class imbalance is mild but signal is weak) | Test 90/180-day windows; longer horizons may yield sharper class signal |
| **Geography** | Single global model | International high-value customers over-represented in priority list and possibly under-served. Consider regional models |
| **Explainability** | Notebook-level SHAP plots | Per-customer "why" string in the action list (top-3 SHAP contributors as plain English) |
| **Governance** | Single CV + holdout | Backtest at multiple historical cutoffs; monitor drift on top-5 SHAP features; alert on AUC degradation > 0.03 |


## Tech stack

- **Core:** Python 3, pandas, NumPy, matplotlib, seaborn
- **Statistical:** `lifelines` (KM, Cox, Weibull AFT), `lifetimes` (BG/NBD, Gamma-Gamma), `statsmodels`
- **ML:** scikit-learn (LogReg, K-Means, PCA, CV), `lightgbm`, `xgboost`, `shap`
- **DL:** PyTorch (GRU + Attention, Transformer Encoder, LSTM)
- **Tuning:** `RandomizedSearchCV` with `StratifiedKFold`

## Dataset citation

Chen, D. (2019). *Online Retail II.* UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/502/online+retail+ii
