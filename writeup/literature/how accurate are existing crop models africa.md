# How accurate are existing land cover maps for agriculture in Sub-Saharan Africa?

**Citation:** Hannah Kerner, Catherine Nakalembe, Adam Yang, Ivan Zvonkov, Ryan McWeeny, Gabriel Tseng & Inbal Becker-Reshef (2024). "How accurate are existing land cover maps for agriculture in Sub-Saharan Africa?" *Scientific Data*, 11, 486. DOI: `10.1038/s41597-024-03306-z` (verified against Crossref — title, authors, journal, volume, and article number all match).

## Objectives

Provide a statistically rigorous, country-scale quantitative evaluation and intercomparison of **11 publicly available land cover / cropland maps** to determine which most accurately identify cropland for Earth-observation-based agricultural monitoring in Sub-Saharan Africa. Specifically:

1. Assess each map's cropland-vs-non-crop accuracy using probability-sampled reference datasets across **8 countries** (Kenya, Malawi, Mali, Rwanda, **Tanzania**, Togo, Uganda, Zambia).
2. Quantify consensus and pairwise agreement among maps.
3. Test whether accuracy correlates with spatial resolution and with temporal mismatch.
4. Demonstrate how the choice of crop mask changes downstream NDVI time-series interpretation.

The 11 maps span 2009–2020, resolutions 10 m–1000 m/px, and tree-based to deep-learning methods: Digital Earth Africa Cropland Extent, Dynamic World, Esri LULC, ESA WorldCover, ESA-CCI Land Cover Africa, GFSAD Global Cropland Extent, Nabil et al., GLAD, Copernicus Land Cover, ESA GlobCover, and ASAP Crop Mask. This is the most directly on-point external benchmark for our "beat the generic agriculture map" claim — and it includes Tanzania.

## Methods

- **Reference data.** A new high-quality reference dataset of **3,386 samples from 8 countries**, collected with statistically rigorous (probability) sampling following accepted best practices (Olofsson-style design-based protocol). Released as a public benchmark, with a GEE app and code repository.
- **Task.** Binary **crop vs non-crop**. All maps were harmonized to this binary and resampled to a common 10 m/px grid for the consensus analysis.
- **Metrics.** Overall accuracy, F1, precision (user's accuracy), recall (producer's accuracy), each with **standard errors** derived under the sampling design. A majority-vote ensemble of all 11 maps was also evaluated.
- **Analyses.** Per-country and mean-over-countries metrics; consensus (% pixels where all maps agree) and pairwise agreement matrices ordered by resolution; correlation of each metric with spatial resolution and with temporal mismatch (years between map and reference); and a downstream demonstration masking MODIS MOD13A1 NDVI time series with each map.

**Evaluation protocol (load-bearing).** This is a **design-based accuracy assessment over national probability samples** — a genuinely **stronger evaluation bar than ours**. Reference points are drawn by statistically rigorous sampling across each country, so the reported F1/precision/recall estimate population-level map accuracy (with standard errors) over the whole country, not within-sample model generalization. It clears the highest practical rung for *map evaluation*: independent, spatially distributed, country-wide reference data — well beyond our field-grouped CV within one region. **Two caveats make it not apples-to-apples with our results:** (1) it is **binary crop/non-crop**, whereas we do **multi-class crop typing** (maize, cotton, rice, sorghum, millet, sunflower, cassava, plus non-crop classes); and (2) it evaluates *pre-existing published maps*, not a model we trained. So it sets the external bar our map must beat, but its F1 numbers are not directly comparable to our multi-class Cohen's Kappa / per-class F1.

## Key Findings

- **No single best map.** Across metrics and countries, **WorldCover and GLAD** performed best overall (high agreement with the majority vote, similar NDVI behavior); Digital Earth Africa most often had the single highest score; **GlobCover, Copernicus, ASAP, ESA-CCI, and Dynamic World were lowest**. GlobCover should be used cautiously.
- **Accuracy is low and highly country-dependent.** Mean F1 across maps ranged from **0.21 ± 0.22 (Mali) to 0.71 ± 0.16 (Rwanda)**; mean F1 was **below 0.7 for 7 of 8 countries**. Best individual maps reach roughly **F1 ~0.66–0.78** in the better-performing countries. Cropland classification in Sub-Saharan Africa remains hard.
- **Very low cropland consensus.** All 11 maps unanimously agree on cropland in **<0.5% of pixels** in every country (e.g., Tanzania crop consensus 0.0%, overall-class consensus 44.4%). Disagreement concentrates at crop/non-crop boundaries — likely zones of cropland expansion.
- **Accuracy weakly favors higher resolution and closer temporal match,** but the correlations are weak-to-moderate and sensitive to outliers (ASAP 1000 m, GlobCover 300 m/2009). Finer resolution mainly reduces false positives (higher precision), not false negatives (recall).
- **Regional beats global.** Maps trained/optimized for a regional sub-group (agro-ecological zone, tile, or country) tend to **outperform single global models** (Esri, Dynamic World, WorldCover are the global ones). Global models suffer high intra-class variance across diverse agro-ecologies.
- **Temporal features matter.** Esri and Dynamic World are the only maps with **no temporal input** (single-image/composite segmentation); the authors note temporal information is important for cropland identification and likely contributes to better-performing maps having it.
- **Ensembles are not automatically best.** The majority-vote ensemble beat most but not all individual maps; ensembling fails when base maps are correlated (e.g., Nabil et al. is itself a blend of GFSAD/ESA-CCI/Copernicus).
- **Map choice changes downstream science.** Different crop masks produce materially different MODIS-NDVI time series (GlobCover overestimates crop conditions in Malawi, Tanzania, Rwanda, Zambia), affecting end-user interpretation of vegetation conditions.

## Relevance to Our Crop-Classification Study

This is the **single most important external benchmark for the manuscript's "beat the generic agriculture map" claim**, and it explicitly includes Tanzania.

- **Sets the bar to beat — with Tanzania-specific numbers.** It gives design-based, Tanzania-specific cropland accuracy for 11 widely used maps; the best maps reach only roughly F1 ~0.66–0.78 (and mean across maps is below 0.7 for almost every country). Our crop-type map should be positioned as outperforming these generic agriculture/land-cover products in the same region. Cite the Tanzania column specifically.
- **"Regional beats global" supports our local-model thesis.** Their finding that region-optimized maps beat global models directly backs our decision to train a Tanzania-specific classifier rather than rely on global land-cover products — the same argument extended from cropland masking to crop typing.
- **"Temporal features matter" supports our feature design.** Their observation that maps without temporal input (Esri, Dynamic World) underperform reinforces our multi-temporal `xr_fresh` time-series feature approach over single-composite classification.
- **The crowdsourced-label and field-grouping motivation.** Their <0.5% cropland consensus and persistently low accuracy show how poorly existing maps capture smallholder African cropland — the gap our crowdsourced (YouthMappers) ground truth and crop-type model aim to fill.
- **Per-class / per-region honesty.** Their per-country F1 spread (0.21–0.71) and use of precision/recall (not just accuracy — Esri got 0.98 accuracy in Mali by predicting all non-crop) reinforce our choice of balanced metrics (Cohen's Kappa, per-class F1) over overall accuracy, especially for rare classes.

## Evaluation Caveats

- **Stronger protocol than ours — not apples-to-apples.** Their national probability-sampled, design-based accuracy assessment is a higher evaluation rung than our field-grouped CV within one region. When comparing, we must be explicit that their F1 estimates *population-level map accuracy* over a whole country, whereas our Kappa/F1 estimate *model generalization within the labelled sample*. Do not present our grouped-CV scores as if measured on the same footing.
- **Binary vs multi-class.** They evaluate crop-vs-non-crop only; we do multi-class crop typing. Their F1 ~0.66–0.78 is a cropland-masking number, not a crop-type-discrimination number, so it bounds a *different and easier* task. Our multi-class results should be benchmarked against the cropland-detection ceiling carefully, not equated with it.
- **Evaluates existing maps, not a trained model.** This is an intercomparison of published products, so its numbers are a target/context, not a competing model we re-ran under identical conditions.
- **Temporal/resolution correlations are weak and outlier-sensitive.** The resolution and temporal-mismatch relationships are weak-to-moderate (R values often |R|<0.45) and flip sign when ASAP/GlobCover are excluded — treat as suggestive, not definitive.
- **Reference-set size per country is modest.** 3,386 samples across 8 countries (a few hundred per country) yields sizable standard errors (e.g., Mali F1 0.21 ± 0.22) — country-level rankings carry real uncertainty.
