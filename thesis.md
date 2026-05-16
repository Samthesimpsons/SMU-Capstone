# Thesis: Profile Coherence as a Diagnostic Lens for Financial Asset Recommendation

## Table of Contents

1. [Abstract](#abstract)
2. [Introduction](#1-introduction)
3. [Methodology](#2-methodology)

> **Note**: Findings, discussion, and conclusion are developed in the companion findings notebook (`notebooks/findings.ipynb`).

## Abstract

Financial Asset Recommendation (FAR) literature optimises ranking quality (nDCG, Recall) and realised return (ROI) without exposing whether recommendations are profile-aligned in the regulatory sense required by MiFID II suitability. This thesis introduces **Profile Coherence at k (PC@k)**, an evaluation metric that scores the share of a top-k list whose asset risk class lies within ordinal tolerance of the customer's declared MiFID II risk band, together with a band-conditional random baseline `pi(b)` that makes the metric interpretable across customer segments. Four questions are asked of FAR-Trans (Sanz-Cruzado et al., 2024) under this lens: **RQ1** (diagnostic) audits how prevalent profile-discordance is in the observed Buy record, and whether it is a customer-level trait or transaction-level noise; **RQ2** (quasi-causal) asks whether profile-discordant Buys earn lower realised six-month forward return than profile-coherent ones once asset volatility, customer segment, and year are absorbed in a transaction-level OLS with cluster-robust standard errors on `customerID`; **RQ3** (audit) locates the FAR-Trans baselines (LightGCN, Random Forest) on the PC@10 axis at their primary-metric optima, via a model-level panel regression that conditions on declared band; **RQ4** (method) asks whether a stratified, profile-coherent LightGCN extension that trains one sub-model per declared MiFID risk band with a profile-coherent margin loss on top of BPR can improve PC@10 over the global LightGCN baseline, and what the trade against nDCG@10, Recall@10, and ROI@10 looks like on aggregate and per declared band, with a `lambda = 0` versus `lambda = 1` ablation that isolates the loss-term contribution from the stratified-architecture contribution.

## 1. Introduction

Financial asset recommendation sits at the intersection of personalised ranking and regulatory obligation. The European Markets in Financial Instruments Directive (MiFID II) requires investment firms to assess suitability before recommending a financial product. Suitability means alignment between the product's risk profile and the customer's declared risk tolerance, investment horizon, and capacity to absorb losses. A recommender that maximises predicted return without conditioning on the customer's risk band can be both an excellent ranker on standard benchmarks and a regulatory liability in deployment.

The FAR-Trans dataset (Sanz-Cruzado et al., 2024) is the first open benchmark for FAR that includes both customer profile metadata and a multi-year transaction history. It records 228,913 Buy transactions across 806 assets and 29,090 retail customers, alongside MiFID II declared risk bands for each customer and asset metadata sufficient to derive each asset's risk class through a hierarchical mapping. The accompanying paper benchmarks LightGCN, Random Forest, and several other recommenders on nDCG@10, ROI@10, and Recall@10. None of those metrics measures profile alignment, and none of the published baselines explicitly conditions on the declared risk band.

This thesis introduces a profile-alignment lens for FAR at three levels: a metric (PC@k), an audit of the FAR-Trans transaction record through that metric, and a comparative test of how the existing baselines behave on the new axis. The research questions answered here are:

- **RQ1** *(Diagnostic)*: How prevalent is profile-discordance in observed FAR-Trans Buy transactions, and is it customer-level trait or transaction-level noise?
- **RQ2** *(Quasi-causal)*: Do profile-discordant transactions earn lower realised 6-month return than profile-coherent ones, after adjusting for asset volatility, customer segment, and year?
- **RQ3** *(Audit)*: Where do the FAR-Trans baselines (Random Forest, LightGCN) sit on the PC@10 axis, and does any baseline match or exceed the band-conditional random baseline for every declared band?
- **RQ4** *(Method)*: Can a stratified, profile-coherent LightGCN extension (one sub-model per declared MiFID risk band, trained with a profile-coherent margin loss) improve PC@10 over the global LightGCN baseline, and at what cost to nDCG@10, Recall@10, and ROI@10? Does the gain vary by declared risk segment, and does the realised-return cost concentrate where the regulatory deficit lives or somewhere else?

The contributions of the thesis are: (i) a regulator-aligned coherence metric (PC@k) with a band-conditional random baseline (`pi(b)`) and a scale-invariant normalisation (PC-lift@k); (ii) a customer-level audit of FAR-Trans Buy transactions through the PC@k lens, decomposing discordance by calendar year, customer segment, and declared MiFID risk band; (iii) a transaction-level OLS regression specification that estimates the conditional six-month forward-return gap between coherent and discordant Buys, with cluster-robust standard errors clustered on `customerID`; (iv) a model-level panel regression specification that locates each FAR-Trans baseline on the PC@10 axis by declared band, with cluster-robust standard errors clustered on `customer_id`; (v) a stratified profile-coherent LightGCN extension (one sub-model per declared MiFID risk band, trained with a profile-coherent margin loss on top of the BPR objective), with a clean `lambda = 0` versus `lambda = 1` ablation that isolates the contribution of the coherence loss from the contribution of the stratified architecture; and (vi) a per-band decomposition framework for both PC@10 and ROI@10 across the baseline, the `lambda = 0` ablation, and the `lambda = 1` treatment, over the 69 monthly temporal splits used by the FAR-Trans evaluation protocol.

## 2. Methodology

### 2.1 Profile Coherence at k

Let `b_u in {0, 1, 2, 3}` be customer `u`'s declared MiFID II risk band, ordinally encoded as Conservative (0), Income (1), Balanced (2), Aggressive (3). Customer bands are read from FAR-Trans's `riskLevel` field: the four declared strings (`Conservative`, `Income`, `Balanced`, `Aggressive`) map to bands directly, and the four regression-imputed strings (`Predicted_Conservative` ... `Predicted_Aggressive`) get the same ordinal mapping with an `is_predicted` flag carried alongside; customers with `Not_Available` or a missing `riskLevel` have `b_u = None` and are excluded from the metric.

Each asset `i` is mapped to a risk band `b_i` via a hierarchical three-step rule (`build_asset_risk_classes` in `src/utils/profile_coherence.py`):

1. **Subcategory metadata first.** Mutual-fund (MTF) subcategories map as `Money Market → Conservative`, `Bond / Bonds → Income`, `Balanced → Balanced`, `Equity / Large Cap → Aggressive`. Bond subcategories map as `Government → Conservative`, `Corporate → Income`, with any other bond subcategory defaulting to Income.
2. **Volatility-quartile fallback for stocks.** A trailing 252-day annualised log-return standard deviation is computed per ISIN; the stock-only volatility distribution gives quartile cutoffs `(q1, q2, q3)`, and each stock is bucketed `Conservative` / `Income` / `Balanced` / `Aggressive` accordingly.
3. **Balanced default.** Assets that fall through both rules (typically stocks without enough price history to compute volatility) are assigned `Balanced`, the centre-of-distribution prior.

Pairwise discordance is the absolute ordinal distance:

```
d(u, i) = |b_u - b_i| in {0, 1, 2, 3}
```

A recommendation is **profile-coherent** under the default tolerance iff `d(u, i) <= 1`; a stricter variant uses `d(u, i) == 0` (exact band match) and is reported only as a sensitivity row in the EDA. PC@k for one user is the share of the top-k that is coherent:

```
PC@k(u) = (1/k) * |{i in top_k(u) : d(u, i) <= 1}|
```

Three implementation rules close out the per-user definition (`compute_profile_coherence_at_k` in `src/utils/metrics.py`). First, **truncation rather than padding**: a recommender that returns fewer than k items is scored on what it returned, so `len(top_k)` replaces `k` in the denominator if the slate is short. Second, **assets with no resolvable band count as discordant**: an asset whose `asset_id` is missing from the asset-band lookup contributes nothing to `coherent_count`. The Balanced default in step 3 of the asset-band rule means this is rare in practice, but the rule is conservative against unclassifiable items. Third, **customers without a usable MiFID risk band (declared or regression-imputed) are excluded** by setting their per-customer PC@k to zero, so the aggregate metric draws only on customers with a profile signal.

The aggregate **PC@k for a split** is the unweighted mean of per-customer PC@k across the eligible customers in that split; no slate-volume or band-population reweighting is applied. The thesis-level numerics then average over the 69 splits.

A uniformly-random recommender that samples assets ignoring customer profile achieves a band-conditional baseline that depends only on the asset-universe distribution:

```
pi(b) = |{i in A : |b - b_i| <= 1}| / |A|
```

On FAR-Trans, `pi(b)` is computed once from the empirical asset-band distribution; the per-band values are reported with the findings. PC-lift@k normalises PC@k by the customer's band baseline:

```
PC-lift@k(u) = PC@k(u) / pi(b_u)
```

A lift of 1.0 means the recommender is no better than uniform sampling for that customer's band; values above 1.0 indicate above-random skill. The thesis reports PC@k and PC-lift@k jointly because aggregate PC@k can be inflated by a recommender that simply tilts toward the largest asset band, while PC-lift@k makes that inflation visible.

### 2.2 Evaluation protocol for RQ3 and RQ4

Following FAR-Trans, the model-level analysis uses 69 monthly temporal splits between August 2019 and April 2025. For each split, models are trained on interactions whose timestamp falls before the split's `time_point` and evaluated on test interactions in `[time_point, test_end]`. Per-split metrics are nDCG@10, ROI@10 (geometric 30-day-rescaled forward return averaged over the top-10), Recall@10, PC@10, and PC-lift@10. A "trial" is one fully specified hyperparameter configuration evaluated as 69 fresh train-evaluate cycles. Random Forest is tuned on a 12-trial grid over `(number_of_estimators, max_depth)` with `prediction_horizon_months = 6` and primary metric ROI@10; LightGCN is tuned on an 8-trial grid over `(embedding_dimension, number_of_layers, learning_rate)` with primary metric nDCG@10. The asset risk classification used by PC@10 is computed once on the full FAR-Trans price history; we treat this as a methodological caveat documented alongside the findings. Note also the validation/evaluation window overlap caveat documented in [README.md](README.md#validationevaluation-window-overlap-known-caveat); it affects all models symmetrically and therefore does not bias relative model comparisons.

### 2.3 Transaction-level regression for RQ2

For RQ2, the unit of observation is one Buy transaction. Each Buy is joined to the closest available close price at its transaction timestamp (a `merge_asof` with backward direction and 7-day tolerance) and to the closest available close price exactly six calendar months later under the same convention. The realised return is

```
realised_return = (end_price - start_price) / start_price
```

Each transaction is annotated with `is_coherent = 1[d(u, i) <= 1]`, asset volatility (the asset's trailing 252-day annualised volatility, computed once over the full history), customer segment (`customer_type`), and the calendar year of the transaction. The fit is

```
realised_return ~ is_coherent + asset_volatility + C(customer_type) + C(year)
```

with cluster-robust standard errors clustered on `customerID`. The coefficient on `is_coherent` is the headline: it gives the conditional six-month return gap between coherent and discordant Buys for a typical customer-year-volatility cell. Buys that lack a forward price within tolerance, a customer band, an asset band, a customer segment, or a computable asset volatility are dropped; the resulting panel size and unique-customer count are reported with the findings. RQ2 is *quasi-causal* in the sense that observable confounders are adjusted for through the controls, but unobservable factors (customer attention, exogenous information shocks, channel-level effects) are not, so the coefficient should be read as the conditional association rather than as a causal effect.

### 2.4 Model panel regression for RQ3

The model-level regression takes each baseline's best-trial recommendation parquet, reduces it to one row per (customer, split) carrying the customer's coherent share for that split, and stacks the two model panels. The fit is

```
coherent_share ~ C(declared_band) * C(model) + C(split_index)
```

with cluster-robust standard errors clustered on `customer_id`. The interaction coefficients `C(declared_band)[T.b]:C(model)[T.random_forest]` quantify how Random Forest shifts each band's coherence relative to the reference cell (Conservative band, LightGCN). Predictions are reported at the median split index of the unbalanced panel (the eligible-customer set varies by split), which serves as a representative middle-of-window time point.

### 2.5 Stratified profile-coherent LightGCN for RQ4

RQ4 is a method-axis intervention. Two design decisions distinguish it from the global RQ3 baseline. First, **stratification**: rather than training one global LightGCN over the full customer-asset bipartite graph, the architecture maintains four independent sub-models, one per declared MiFID risk band (Conservative, Income, Balanced, Aggressive). Each sub-model is trained on the sub-graph induced by customers in its band and the full asset universe, so the recommender can specialise the embedding geometry to the subpopulation. Customers without a usable risk band are routed to the Balanced sub-model at inference time, matching the centre-of-distribution prior used in regulator-aligned imputation.

Second, a **profile-coherent margin loss** is added to the BPR objective. For each training step on a sub-model targeting band `b`, two assets are sampled per customer: an asset whose risk band lies within tolerance of `b` (`coherent`, drawn from the asset-band-conditional pool with `|b_i - b| <= 1`), and an asset whose risk band lies at or beyond two ordinal steps (`discordant`, `|b_i - b| >= 2`). The loss is

```
L = L_BPR + L_L2 + lambda * mean( softplus(score_discordant - score_coherent) )
```

where `score_x` is the dot product of the customer embedding with the asset embedding under the standard LightGCN propagation, and `lambda` is the profile-coherent loss weight. `lambda = 0` recovers a pure stratified LightGCN with no explicit coherence pressure (the **ablation** trial); `lambda = 1` is the **treatment** trial used as the headline configuration. Both trials share a fixed LightGCN backbone configuration (`embedding_dimension = 64, number_of_layers = 3, learning_rate = 1e-3, weight_decay = 1e-5, keep_probability = 0.6, number_of_epochs = 50, batch_size = 1024`) inherited from the RQ3 grid-search configuration, so any difference between `lambda = 0` and `lambda = 1` is attributable to the coherence pressure rather than to a hyperparameter accident, and the difference between `lambda = 0` and the RQ3 LightGCN baseline is attributable to stratification alone.

Both trials are evaluated on the same 69 monthly temporal splits used in RQ1-RQ3, with the same metrics (nDCG@10, ROI@10, Recall@10, PC@10, PC-lift@10), so the comparison against the global LightGCN baseline is paired at the per-split level. Three contrasts are reported with the findings: (i) `lambda = 0` versus the RQ3 LightGCN baseline, isolating the stratification effect; (ii) `lambda = 1` versus the RQ3 LightGCN baseline, the headline practitioner-relevant comparison; (iii) `lambda = 1` versus `lambda = 0`, isolating the profile-coherent loss effect under matched architecture and hyperparameters. The per-band decomposition mirrors Section 2.4's panel layout: PC@10 is averaged across customer-split rows for each declared band, then divided by `pi(b)` to give a band-conditional lift.
