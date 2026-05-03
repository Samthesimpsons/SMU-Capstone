# Thesis: Profile Coherence as a Diagnostic Lens for Financial Asset Recommendation

> **Note.** All numerics in this document come from a single end-to-end pipeline run. The dataset audit was produced by `uv run poe eda` (`outputs/eda/summary.json`). The baseline grid sweep used by RQ3 is timestamped `20260427_122215` (`outputs/results/evaluation/{random_forest,light_gcn}/20260427_122215/` and `outputs/analysis/baseline_decomposition/20260427_122215/`). The transaction-level RQ2 regression is in `outputs/analysis/transaction_return_regression/20260504_004849/` and the model-level RQ3 panel regression in `outputs/analysis/panel_regression/20260504_003145/`. Project context, code architecture, and reproduction instructions live in [README.md](README.md).

## Table of Contents

1. [Abstract](#abstract)
2. [Introduction](#1-introduction)
3. [Methodology](#2-methodology)
4. [Findings](#3-findings)
5. [Discussion](#4-discussion)
6. [Conclusion](#5-conclusion)

## Abstract

Financial Asset Recommendation (FAR) literature optimises ranking quality (nDCG, Recall) and realised return (ROI) without exposing whether recommendations are profile-aligned in the regulatory sense required by MiFID II suitability. This thesis introduces **Profile Coherence at k (PC@k)**, an evaluation metric that scores the share of a top-k list whose asset risk class lies within ordinal tolerance of the customer's declared MiFID II risk band, together with a band-conditional random baseline `pi(b)` that makes the metric interpretable across customer segments. Three questions are asked of FAR-Trans (Sanz-Cruzado et al., 2024) under this lens. **RQ1** audits how prevalent profile-discordance is in the observed Buy record: 18.6% of 228,241 scoreable Buys are discordant (`d > 1`), the per-customer distribution is sharply bimodal (64.4% fully coherent, 17.2% fully discordant), and the two extreme bands account for almost all of the discordance (Conservative coherent share 45.1%, Aggressive 56.2% versus 90% on Income and Balanced). **RQ2** asks whether discordant transactions are also worse trades: a transaction-level OLS over 208,029 priced Buys, controlling for asset volatility, customer segment, and year and clustered on `customerID`, finds that profile-coherent transactions earn **+2.94 percentage points** more 6-month forward return than profile-discordant ones (95% CI [+2.15, +3.72] pp, p < 1e-12). **RQ3** locates the FAR-Trans baselines on the new axis: at their primary-metric optima LightGCN attains aggregate PC@10 = 0.794 (PC-lift 1.17) and Random Forest attains PC@10 = 0.667 (PC-lift 1.06), but the model-level panel regression shows that LightGCN is below the random baseline for the Conservative band and Random Forest is below random for Conservative and Income, which together cover 60.9% of the eligible customer population. The headline finding for the field is that profile coherence is not a regulatory tax: on FAR-Trans it co-moves positively with realised return at the transaction level, and the existing baselines under-serve precisely the customer bands the regulation was designed to protect.

## 1. Introduction

Financial asset recommendation sits at the intersection of personalised ranking and regulatory obligation. The European Markets in Financial Instruments Directive (MiFID II) requires investment firms to assess suitability before recommending a financial product. Suitability means alignment between the product's risk profile and the customer's declared risk tolerance, investment horizon, and capacity to absorb losses. A recommender that maximises predicted return without conditioning on the customer's risk band can be both an excellent ranker on standard benchmarks and a regulatory liability in deployment.

The FAR-Trans dataset (Sanz-Cruzado et al., 2024) is the first open benchmark for FAR that includes both customer profile metadata and a multi-year transaction history. It records 228,913 Buy transactions across 806 assets and 29,090 retail customers, alongside MiFID II declared risk bands for each customer and asset metadata sufficient to derive each asset's risk class through a hierarchical mapping. The accompanying paper benchmarks LightGCN, Random Forest, and several other recommenders on nDCG@10, ROI@10, and Recall@10. None of those metrics measures profile alignment, and none of the published baselines explicitly conditions on the declared risk band.

This thesis introduces a profile-alignment lens for FAR at three levels: a metric (PC@k), an audit of the FAR-Trans transaction record through that metric, and a comparative test of how the existing baselines behave on the new axis. The research questions answered here are:

- **RQ1** *(Diagnostic)*: How prevalent is profile-discordance in observed FAR-Trans Buy transactions, and is it customer-level trait or transaction-level noise?
- **RQ2** *(Quasi-causal)*: Do profile-discordant transactions earn lower realised 6-month return than profile-coherent ones, after adjusting for asset volatility, customer segment, and year?
- **RQ3** *(Audit)*: Where do the FAR-Trans baselines (Random Forest, LightGCN) sit on the PC@10 axis, and does any baseline match or exceed the band-conditional random baseline for every declared band?

A method-axis question (RQ4) is being redefined with the supervisor and is omitted from this draft.

The contributions of the thesis are: (i) a regulator-aligned coherence metric with a band-conditional normalisation; (ii) a customer-level audit of FAR-Trans that decomposes discordance by year, segment, and declared band; (iii) a transaction-level regression that quantifies the conditional return gap between coherent and discordant Buys; (iv) a model-level panel regression that locates each baseline on the PC@10 axis, by declared band, with cluster-robust standard errors clustered on `customer_id`.

## 2. Methodology

### 2.1 Profile Coherence at k

Let `b_u in {0, 1, 2, 3}` be customer `u`'s declared MiFID II risk band, ordinally encoded as Conservative (0), Income (1), Balanced (2), Aggressive (3). Each asset `i` is mapped to a risk band `b_i` via a hierarchical scheme that prefers explicit subcategory metadata (mutual-fund subcategory or bond subcategory) and falls back to a 252-day annualised volatility quartile for stocks and any remaining assets without subcategory information. Pairwise discordance is the absolute ordinal distance:

```
d(u, i) = |b_u - b_i| in {0, 1, 2, 3}
```

A recommendation is **profile-coherent** under the default tolerance iff `d(u, i) <= 1`. PC@k for one user is the share of the top-k that is coherent:

```
PC@k(u) = (1/k) * |{i in top_k(u) : d(u, i) <= 1}|
```

Recommendations to assets whose risk band cannot be determined are treated as discordant. Customers without a usable MiFID risk band (declared or regression-imputed) are excluded by setting their per-customer PC@k to zero, so the aggregate metric draws only on customers with a profile signal.

A uniformly-random recommender that samples assets ignoring customer profile achieves a band-conditional baseline that depends only on the asset-universe distribution:

```
pi(b) = |{i in A : |b - b_i| <= 1}| / |A|
```

On FAR-Trans, where the asset bands are distributed 190 / 333 / 105 / 178 across Conservative / Income / Balanced / Aggressive (from the EDA's `asset_band_distribution_by_category`), this gives `pi = (0.649, 0.779, 0.764, 0.351)`. The Aggressive baseline is the lowest because its tolerated set `{Balanced, Aggressive}` covers only 35.1% of the 806-asset universe; the Income baseline at 0.779 includes the three central bands. PC-lift@k normalises PC@k by the customer's band baseline:

```
PC-lift@k(u) = PC@k(u) / pi(b_u)
```

A lift of 1.0 means the recommender is no better than uniform sampling for that customer's band; values above 1.0 indicate above-random skill. The thesis reports PC@k and PC-lift@k jointly because aggregate PC@k can be inflated by a recommender that simply tilts toward the largest asset band (Income, in our case), while PC-lift@k makes that inflation visible.

### 2.2 Evaluation protocol for RQ3

Following FAR-Trans, the model-level analysis uses 69 monthly temporal splits between August 2019 and April 2025. For each split, models are trained on interactions whose timestamp falls before the split's `time_point` and evaluated on test interactions in `[time_point, test_end]`. Per-split metrics are nDCG@10, ROI@10 (geometric 30-day-rescaled forward return averaged over the top-10), Recall@10, PC@10, and PC-lift@10. A "trial" is one fully specified hyperparameter configuration evaluated as 69 fresh train-evaluate cycles. Random Forest is tuned on a 12-trial grid over `(number_of_estimators, max_depth)` with `prediction_horizon_months = 6` and primary metric ROI@10; LightGCN is tuned on an 8-trial grid over `(embedding_dimension, number_of_layers, learning_rate)` with primary metric nDCG@10. The asset risk classification used by PC@10 is computed once on the full FAR-Trans price history; we treat this as a methodological caveat in Section 4.4. Note also the validation/evaluation window overlap caveat documented in [README.md](README.md#validationevaluation-window-overlap-known-caveat); it affects all models symmetrically and therefore does not bias relative model comparisons.

### 2.3 Transaction-level regression for RQ2

For RQ2, the unit of observation is one Buy transaction. Each Buy is joined to the closest available close price at its transaction timestamp (a `merge_asof` with backward direction and 7-day tolerance) and to the closest available close price exactly six calendar months later under the same convention. The realised return is

```
realised_return = (end_price - start_price) / start_price
```

Each transaction is annotated with `is_coherent = 1[d(u, i) <= 1]`, asset volatility (the asset's trailing 252-day annualised volatility, computed once over the full history), customer segment (`customer_type`), and the calendar year of the transaction. The fit is

```
realised_return ~ is_coherent + asset_volatility + C(customer_type) + C(year)
```

with cluster-robust standard errors clustered on `customerID`. The coefficient on `is_coherent` is the headline: it gives the conditional 6-month return gap between coherent and discordant Buys for a typical customer-year-volatility cell. Out of 228,913 Buys, 208,029 (90.9%) survive the joint join (forward price within tolerance, both bands present, customer segment present, volatility computable), so the panel covers 27,873 unique customers. RQ2 is *quasi-causal* in the sense that observable confounders are adjusted for through the controls, but unobservable factors (customer attention, exogenous information shocks, channel-level effects) are not, so the coefficient should be read as the conditional association rather than as a causal effect.

### 2.4 Model panel regression for RQ3

The model-level regression takes each baseline's best-trial recommendation parquet, reduces it to one row per (customer, split) carrying the customer's coherent share for that split, and stacks the two model panels. The fit is

```
coherent_share ~ C(declared_band) * C(model) + C(split_index)
```

with cluster-robust standard errors clustered on `customer_id`. The interaction coefficients `C(declared_band)[T.b]:C(model)[T.random_forest]` quantify how Random Forest shifts each band's coherence relative to the reference cell (Conservative band, LightGCN). Predictions are reported at the median split index (36) of the unbalanced panel (the eligible-customer set varies by split), which serves as a representative middle-of-window time point.

## 3. Findings

### 3.1 RQ1: profile-discordance is prevalent and structural

Across 228,241 scoreable Buy transactions (99.7% of the 228,913 raw Buys carry both customer band and asset band), the discordance distribution is:

| Discordance `d` | Count | Share |
|---|---:|---:|
| `d = 0` (exact match) | 77,925 | 34.1% |
| `d = 1` (within tolerance) | 107,899 | 47.3% |
| `d = 2` | 35,581 | 15.6% |
| `d = 3` (extreme mismatch) | 6,836 | 3.0% |

Under the default tolerance (`d <= 1`), **81.4% of Buy transactions are profile-coherent**, so 18.6% (42,417 Buys) violate the declared MiFID band by 2 or more steps. Mean discordance is 0.87 bands. The discordance is *not* spread uniformly across the customer base: per-customer self-discordance is sharply bimodal, with **64.4% of customers fully coherent** (every Buy at `d <= 1`), **17.2% fully discordant** (every Buy at `d >= 2`), and **20.3% above 50% discordance**. The middle of the per-customer distribution is sparse, so most customers transact in the band that the questionnaire predicts (or systematically outside it) rather than alternating. This bimodality is the strongest empirical motivation for treating profile alignment as a load-bearing recommender axis.

The discordance is concentrated on the two ordinal extremes:

| Declared MiFID band | Transactions | Coherent share (`d <= 1`) |
|---|---:|---:|
| Conservative | 14,193 | **45.1%** |
| Income | 73,468 | 90.9% |
| Balanced | 99,901 | 89.9% |
| Aggressive | 40,679 | **56.2%** |

Mid-band customers (Income, Balanced) are roughly 90% coherent. Conservative and Aggressive customers are dramatically less coherent: more than half of Conservative customers' Buys land at `d >= 2` (chasing risk), and 44% of Aggressive customers' Buys land in safer assets than their declared profile would permit. The pattern is consistent with regression-toward-the-centre under loss aversion (Aggressive customers reaching for yield-bearing safer assets) and with reach-for-yield under low rates (Conservative customers buying riskier products).

Two further audit slices rule out competing explanations. Mean discordance per calendar year stays inside the 0.83-0.98 band across 2018-2022 (`outputs/eda/discordance_by_year.png`); the COVID-19 drawdown and recovery do not visibly shift the pattern, ruling out a "panic trading drove the discordance" reading. Mean discordance by `customerType` ranges from 0.81 (Professional) to 1.52 (Legal Entity), with Mass at 0.85 and Premium at 0.91, so the trait is segment-driven rather than regime-driven. A volatility-only sensitivity mapping (no subcategory metadata, all assets binned by quartile) shifts the headline `d <= 1` rate from 81.4% down to 73.2% and the mean discordance from 0.87 up to 1.03; the qualitative findings are robust to the mapping convention while absolute coherence levels are not.

The audit therefore answers RQ1: profile-discordance in FAR-Trans is prevalent (18.6% of Buys), a customer-level trait rather than transaction-level noise (bimodal self-discordance, U-shaped band decomposition), and stable across the 2018-2022 macro window. Full numerics are in `outputs/eda/summary.json`.

### 3.2 RQ2: profile-coherent transactions earn higher realised return

The transaction-level OLS over the 208,029-Buy panel (27,873 unique customers; coherent share 81.3%) has the following key coefficients, with cluster-robust standard errors clustered on `customerID`:

| Term | Estimate | Std. error | 95% CI | p-value |
|---|---:|---:|---|---:|
| `is_coherent` | **+0.0294** | 0.0040 | [+0.0215, +0.0372] | 2.3e-13 |
| `asset_volatility` | -0.1025 | 0.0165 | [-0.1349, -0.0701] | 5.5e-10 |
| `C(year)[T.2020]` | +0.2107 | 0.0061 | [+0.1986, +0.2227] | 6.2e-259 |
| `C(year)[T.2022]` | -0.0195 | 0.0045 | [-0.0283, -0.0107] | 1.4e-05 |
| `Intercept` | -0.0471 | 0.0141 | [-0.0746, -0.0195] | 8.1e-04 |

Customer-type fixed effects (`Mass`, `Premium`, `Professional`, `Legal Entity`) are not individually significant; the intercept absorbs the omitted category (`Inactive`). The full coefficient table is in `outputs/analysis/transaction_return_regression/20260504_004849/coefficients.csv`.

The headline reading: **conditional on asset volatility, customer segment, and year, profile-coherent Buy transactions earn 2.94 percentage points more 6-month realised return than profile-discordant ones.** The raw slice means are even larger (coherent +4.23%, discordant +0.92%), so the controls absorb roughly 0.4 pp of the unconditional 3.3 pp gap and the residual 2.9 pp is the conditional association. The asset-volatility coefficient is negative as expected (more volatile assets earned lower 6-month forward returns over the sample window). The 2020 fixed effect captures the post-COVID equity rally, and the negative 2022 effect captures the bond drawdown and the equity correction.

The direction is the opposite of the implicit assumption baked into FAR's "profit-versus-preference" framing. RQ2 was posed as "do discordant transactions earn lower returns?" anticipating that the regulator-aligned trades might be the worse trades and that profile coherence would be a regulatory tax. On FAR-Trans the answer is the reverse: **profile-discordant transactions earn lower returns**, the gap is statistically tight (p < 1e-12 under cluster-robust inference on customerID), and the magnitude is economically meaningful at roughly 6 pp annualised. This reframes the problem facing FAR systems: profile coherence is not an axis traded off against return, but an axis that is positively correlated with return on the customer side once observable confounders are absorbed. Whether the field's recommenders preserve this alignment is a separate question, addressed in RQ3.

### 3.3 RQ3: both FAR-Trans baselines under-serve declared-band coherence

After full grid search, the two baselines reach the following at their primary-metric optima:

| Model | Best trial | nDCG@10 | ROI@10 (%/mo) | Recall@10 | PC@10 | PC-lift@10 |
|---|---|---:|---:|---:|---:|---:|
| LightGCN | eb788_00006 | 0.330 | -0.54 | 0.495 | 0.794 | 1.171 |
| Random Forest | df0bd_00008 | 0.019 | +1.42 | 0.036 | 0.667 | 1.060 |

The LightGCN best-trial backbone is `embedding_dimension = 64, number_of_layers = 3, learning_rate = 1e-3, weight_decay = 1e-5, keep_probability = 0.6, number_of_epochs = 50, batch_size = 1024`; the Random Forest best trial is `number_of_estimators = 40, max_depth = 50, prediction_horizon_months = 6`. Both clear the unconditional random baseline `pi(b_u)` averaged over the customer mix, but aggregate PC@10 hides the per-band picture.

The model panel regression (Section 2.4) predicts the following PC at the median split (36), with cluster-robust 95% confidence intervals on `customer_id`:

| Band | `pi(b)` | LightGCN | LGCN lift | Random Forest | RF lift |
|---|---:|---:|---:|---:|---:|
| Conservative | 0.649 | 0.520 | **0.80x** | 0.374 | **0.58x** |
| Income | 0.779 | 0.919 | 1.18x | 0.579 | **0.74x** |
| Balanced | 0.764 | 0.926 | 1.21x | 0.821 | 1.07x |
| Aggressive | 0.351 | 0.508 | 1.45x | 0.697 | 1.99x |

LightGCN is below the random baseline for the Conservative band (lift 0.80x). Random Forest is below random for Conservative (0.58x) **and** Income (0.74x). These two bands together cover 17,510 of 28,770 customers with a usable risk band (5,672 Conservative + 11,838 Income, declared and predicted), or 60.9% of the eligible population once the 320 `Not_Available` customers are removed. Every cell shift is statistically significant against the reference (Conservative, LightGCN) at p < 1e-9 under cluster-robust inference (full coefficient table in `outputs/analysis/panel_regression/20260504_003145/coefficients.csv`). The two interaction coefficients that drive the per-band conclusions are `C(declared_band)[T.3]:C(model)[T.random_forest] = +0.334` (RF over-serves the Aggressive band by +33 pp relative to the Conservative-LightGCN reference) and `C(declared_band)[T.1]:C(model)[T.random_forest] = -0.195` (RF under-serves Income by -19 pp). The forest plot is in `outputs/analysis/panel_regression/20260504_003145/forest_predicted_pc.png`.

A second view comes from decomposing each model's recommendation pool into coherent and discordant slices and computing simple mean monthly return on each slice, across all 1,372,010 customer-split-slot recommendations (`outputs/analysis/baseline_decomposition/20260427_122215/decomposition.csv`):

| Model | Coherent share | Coherent ROI (%/mo) | Discordant ROI (%/mo) | Overall ROI (%/mo) |
|---|---:|---:|---:|---:|
| LightGCN | 0.794 | -0.84 | -0.54 | -0.78 |
| Random Forest | 0.665 | +1.67 | +0.80 | +1.38 |

LightGCN's coherent slice underperforms its discordant slice by 0.30 pp/month, the model-level inversion of the transaction-level finding in Section 3.2: the recommendations LightGCN is willing to label coherent are not the same population as the coherent transactions in the underlying data. Random Forest preserves the transaction-level direction (coherent slice + 0.87 pp/month over discordant) but at a low aggregate PC level (0.665) and with the per-band failure profile shown above, where the +1.42%/month ROI advantage is concentrated on the Aggressive band (+1.99x lift) where RF over-serves by 33 pp.

The findings answer RQ3 affirmatively: existing FAR baselines under-serve declared-band coherence, the failures fall on Conservative (both models) and Income (Random Forest only), together covering 60.9% of the eligible population, and neither baseline reaches `pi(b_u)` for every band simultaneously.

## 4. Discussion

### 4.1 Profile coherence is not a regulatory tax on FAR-Trans

The strongest framing inherited from the FAR-Trans paper is "preference versus profit": transaction-based recommenders win nDCG and lose ROI; price-based recommenders do the reverse; no method dominates either axis. A natural extension is "profile versus profit": adding a regulatory constraint must cost some return, because the constraint excludes the trades the recommender thinks are profitable. RQ2 contradicts this framing on FAR-Trans. Conditional on volatility, customer segment, and year, the *coherent* Buys earn 2.94 pp more 6-month return than the discordant Buys (p < 1e-12 under cluster-robust inference). On the customer side, profile coherence and realised return co-move in the same direction. A FAR recommender that restricted itself to profile-coherent recommendations would, in expectation under this dataset, be a higher-return recommender, not a lower-return one.

The model side does not preserve this alignment. LightGCN's coherent slice underperforms its discordant slice (Section 3.3 decomposition table), so LightGCN is recovering a different population in its coherent recommendations than the population of coherent transactions in the data. Random Forest preserves the direction (coherent slice +0.87 pp/month over discordant) but with a per-band coverage failure on Conservative and Income (5,672 + 11,838 = 17,510 customers, 60.9% of the eligible population). The implication is that RQ2's finding is a property of customer behaviour in FAR-Trans, not a property the existing baselines automatically inherit.

### 4.2 Where the audit places the regulatory risk

The model-level panel regression in Section 3.3 quantifies a structurally suspect deployment risk. LightGCN's Conservative-band coverage is 0.520 against the random baseline of 0.649, i.e. 0.80x random; Random Forest's coverage is even lower at 0.374 (0.58x random). Random Forest also fails Income coverage at 0.579 (0.74x random). The two failing bands together cover 60.9% of the customer population. A naive inference from aggregate PC@10 (LightGCN 0.794, RF 0.667) hides this: the aggregates are above the population-mix-weighted random baseline because the largest customer segment (Income) sits in the asset band with the highest `pi(b)` value (0.779) and because LightGCN over-serves the Income and Balanced bands at the expense of Conservative. PC-lift@k normalisation by `pi(b_u)` makes this inflation visible at the customer level; the per-band panel regression makes it visible at the model level.

The Random Forest interaction coefficient `C(declared_band)[T.3]:C(model)[T.random_forest] = +0.334` is the largest in the regression: Random Forest over-serves Aggressive by 33 pp relative to the reference cell. It is also the model with the headline ROI advantage (+1.42%/month against LightGCN's -0.54%/month). The two facts together suggest RF's ROI advantage is concentrated on the band where it over-shoots the random baseline almost 2x, which is structurally suspect for any deployment whose suitability requirement is binding. The thesis cannot decompose the magnitude of this concentration into a "regime" component and a "skill" component without an independent benchmark in a different macro window, but the interaction coefficient places the RF-versus-LightGCN ROI gap on a cautionary footing for regulator-aligned deployment.

### 4.3 Methodological contribution: PC-lift@k normalisation

The band-conditional random baseline `pi(b)` is non-uniform on FAR-Trans because the asset universe is skewed (Income is 41% of the catalogue; Balanced is 13%; the Aggressive coherent set is 35% of total assets). Reporting raw PC@k allows a centre-tilting recommender to inflate the metric on Income and Balanced customers without delivering structural alignment. PC-lift@k makes this inflation visible: the LightGCN aggregate of 1.171 sits on a per-band lift profile of (0.80, 1.18, 1.21, 1.45) where the Conservative band is sub-random; the RF aggregate of 1.060 sits on a per-band lift profile of (0.58, 0.74, 1.07, 1.99) where two bands are sub-random. The thesis recommends reporting raw PC@k and PC-lift@k jointly in any future PC@k publication, with the basis (per-trial aggregate or panel-regression-predicted) stated explicitly.

### 4.4 Limitations

The findings inherit FAR-Trans's specifics. The asset universe is small (806 assets) and biased toward European listings. The customer base is retail rather than institutional. The temporal window includes the COVID drawdown, the 2022 yield-spike regime, and the 2023-2025 recovery, which is representative but not exhaustive macro spread.

The hierarchical risk-class mapping over assets relies on subcategory metadata that is denser for mutual funds than for stocks, where a volatility-quartile fallback is used. The volatility-only sensitivity (`outputs/eda/summary.json`'s `sensitivity_volatility_only_summary`) shifts the dataset's overall coherence rate by roughly 8 pp (0.81 to 0.73), so absolute per-band PC values are sensitive to the mapping convention while the qualitative audit findings of Section 3.1 are robust under both conventions. The asset risk classification used throughout this thesis is computed once on the full FAR-Trans price history. For PC@k as a metric this is interpretable (alignment to the canonical end-of-dataset risk classification matches how a regulator-aligned suitability check would treat a "currently Aggressive" stock). The transaction-return regression in RQ2 uses the same end-of-dataset volatility for the `asset_volatility` control, which is a mild lookahead; a fully time-honest sensitivity check that recomputes asset volatility per transaction date is left to future work.

The RQ2 regression is *quasi-causal*: observable confounders (asset volatility, customer segment, year) are absorbed, but unobservable confounders are not. The regression cannot distinguish "coherent transactions are better trades because the customer's questionnaire-derived risk band is correlated with skill or attention" from "coherent transactions are better trades because the asset properties implied by the customer's tolerated band are systematically rewarded over the sample window". The 2020 fixed effect of +21 pp is consistent with the latter reading (the pandemic recovery was strongest in equities, which sit in the band the largest customer segment tolerates), but the residual `is_coherent` coefficient of +2.9 pp after year fixed effects suggests there is something beyond regime composition.

The FAR-Trans evaluation protocol has a known overlap between validation and early evaluation splits documented in the [Validation/Evaluation Window Overlap](README.md#validationevaluation-window-overlap-known-caveat) section of the project README. The substantive comparisons in Section 3.3 are within-protocol so the caveat affects both models symmetrically; absolute per-split metric levels should be read with this caveat in mind.

The thesis omits a method-axis question (RQ4) that is being redefined with the supervisor.

## 5. Conclusion

This thesis introduced Profile Coherence at k as a regulator-aligned risk-band evaluation axis for Financial Asset Recommendation, with a band-conditional normalisation (PC-lift@k) that makes the metric scale-invariant across customer segments. PC@k captures the risk-band dimension of MiFID II suitability and does not address the other suitability axes (investment horizon, capacity, knowledge and experience, sustainability preferences); a comprehensive suitability metric for FAR systems would compose multiple alignment terms of which PC@k is one.

Three findings emerge from the FAR-Trans audit. **RQ1**: profile-discordance is prevalent (18.6% of 228,241 scoreable Buys), a customer-level trait rather than transaction-level noise (64.4% of customers are fully coherent and 17.2% are fully discordant), and concentrated on the two ordinal-extreme declared bands (Conservative 45.1% coherent, Aggressive 56.2% coherent). **RQ2**: conditional on asset volatility, customer segment, and year, profile-coherent Buys earn 2.94 percentage points more 6-month realised return than profile-discordant ones, with cluster-robust 95% CI [+2.15, +3.72] pp and p < 1e-12; profile coherence is not a regulatory tax on FAR-Trans, it is positively correlated with realised return. **RQ3**: at their primary-metric optima, LightGCN's predicted PC at split 36 is below the band-conditional random baseline for the Conservative band, and Random Forest's is below random for Conservative and Income, which together cover 60.9% of the eligible population. Random Forest's headline ROI advantage of +1.42%/month is concentrated on the Aggressive band where it over-serves the random baseline nearly 2x, which places the ROI advantage on a structurally suspect footing for a regulator-aligned deployment.

The headline outcome for the field is that the profile-coherence axis is not antagonistic to the return axis on FAR-Trans at the customer level, but the existing FAR baselines do not preserve this alignment in their recommendations. The method-axis question of how to design recommenders that do preserve this alignment, while explicitly defending per-band random-baseline coverage, is the natural follow-up; that question is being defined as the redefined RQ4.
