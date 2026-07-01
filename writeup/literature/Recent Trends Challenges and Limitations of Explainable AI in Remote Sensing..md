# Recent Trends, Challenges, and Limitations of Explainable AI in Remote Sensing

**Citation:** Adrian Höhl, Ivica Obadic (joint first authors), Miguel-Ángel Fernández-Torres, Dario Oliveira & Xiao Xiang Zhu (2024). "Recent Trends, Challenges, and Limitations of Explainable AI in Remote Sensing." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops* — XAI4CV Workshop, pp. 8199–8205. **DOI: not found in PDF** → tagged `unverified`. This is the CVPR Open Access (Computer Vision Foundation) workshop version; one Crossref title search returned **no matching record** (CVPRW Open Access papers are frequently not Crossref-indexed). **Do NOT conflate** with the same group's separate, longer IEEE Geoscience and Remote Sensing Magazine review (Höhl et al., 2024, DOI `10.1109/mgrs.2024.3467001`) — that is a different paper; this briefer covers the 7-page CVPRW scoping paper (pp. 8199–8205).

## Objectives

A **scoping review** of explainable AI (xAI) applied to remote sensing (RS) / Earth observation (EO) deep learning. The authors aim to:

1. Systematically map the current literature on xAI in RS via database searches (Scopus, Springer, IEEE).
2. Identify key trends in *which EO tasks* use xAI and *which xAI method families* are used over time.
3. Surface the specific challenges that RS data properties (scale, spectral richness, topology/geographic relations, temporal dependencies) pose for off-the-shelf xAI methods.
4. Point to promising directions: self-interpretable deep models, and rigorous (quantitative / user-study) explanation evaluation.

This is a review/conceptual paper, so the protocol-rung question does not apply; relevance is framed around interpretability methodology — directly bearing on our SHAP usage.

## Methods

- **Scoping review.** A Boolean query combining xAI keywords (interpret*/explain*, deep learning/ML/AI/model) with EO keywords (earth observation, remote sensing, satellite/aerial/airborne/spaceborne, radar, LiDAR, SAR, UAV). Initial 1,075 papers filtered in three stages — dedup/abstract/full-text screening → **964 → 357 → 147** papers, plus 60 from the authors' libraries.
- **Categorization.** Each work coded by EO task, xAI family, and evaluation procedure. Trends plotted over 2017–2023 for EO tasks (Fig. 1a) and xAI categories (Fig. 1b), and for RS-adapted xAI methods (Fig. 2).
- **No new model or dataset** — this is a meta-analysis of the literature, not an empirical RS classification study.

**Evaluation protocol.** Not applicable in the train/test-leakage sense (review paper — skip per the review-paper guidance). The paper's relevant *methodological* point about evaluation is meta: it documents that **xAI explanation evaluation in RS is itself weak** — most studies rely on anecdotal, qualitative, cherry-picked visual evidence (Fig. 3), with no standardized, objective, or quantitative explanation-quality benchmark, and almost no user studies. This is a caution about how we should (and should not) report our SHAP analysis.

## Key Findings

- **SHAP is the single most-used xAI method in RS — 38% of publications**, with usage spiking in the last two years (it had been rare before). Because SHAP and similar local-approximation/perturbation methods are **model-agnostic**, they offer a simple framework for first-pass interpretation of black-box EO models. Backpropagation methods (notably CAM / Grad-CAM) are the other common family, especially for land-cover and target mapping.
- **Load-bearing caveat: SHAP (and perturbation/local-approximation methods) assume feature independence.** The paper states plainly that "local approximation and perturbation approaches typically assume feature independence" and that these assumptions "are often violated in RS data due to the presence of scale, geographical relationships, and temporal dependencies." This is the key methodological warning for our SHAP-based feature importance.
- **RS data break standard-xAI assumptions** along four axes: (1) **scale** — spatial/spectral resolution and granularity; (2) **spectral richness** beyond RGB (most xAI ignores per-band spectral attribution); (3) **topology** — geographic/spatial relations act as hidden confounders; (4) **temporal dependencies** — most xAI methods do not account for time-series structure (only attention-based and a few bespoke methods do).
- **xAI for land-cover mapping and agricultural monitoring has stagnated** recently, while xAI for natural-hazard/atmosphere/vegetation monitoring is rising; land cover is treated as a "mature, established" task.
- **Explanation evaluation is immature.** Most RS xAI studies provide only anecdotal/qualitative evidence; no standardized, objective evaluation exists; user studies are almost entirely absent — a promising but neglected direction.
- **Self-interpretable ("by-design") models** (attention/transformers, ProtoPNet-style prototype networks, BagNets, DNN+GAM/linear hybrids) are highlighted as the forward direction over post-hoc attribution.

## Relevance to Our Crop-Classification Study

This review directly informs **how we justify and how we qualify our SHAP-based feature-importance analysis**.

- **Backs our SHAP choice.** SHAP being the most-used RS xAI method (38% of publications) and the default model-agnostic interpreter for tree/black-box models is external support for our decision to explain the LightGBM/RandomForest classifier with SHAP. It is the community-standard tool, not an idiosyncratic choice.
- **Load-bearing caveat for our SHAP results.** Our `xr_fresh` engineered time-series features (mean, max, min, slope, skewness, number of peaks, complexity over `EVI`, `B2`, `B6`, `B11`/`B12`, `hue`) are **strongly correlated** — multiple statistics computed from the same band's temporal profile, plus correlated bands. SHAP's feature-independence assumption is therefore **violated in exactly the way this paper warns about** (temporal dependencies and spectral relations). We should explicitly flag that our SHAP importances can mis-attribute or split credit among correlated features, and consider correlation-aware alternatives (e.g., grouped/clustered SHAP, permutation importance on feature groups, or reporting feature-cluster importance) or at least caveat the rankings. This is the single most actionable interpretability finding for our manuscript.
- **Tempers over-interpretation of per-feature rankings.** The paper's documentation that RS xAI evaluation is mostly anecdotal/qualitative is a reminder to present our top-SHAP-feature story as *suggestive and internally consistent across folds* (we already compute mean and max SHAP over CV folds) rather than as a validated causal account of crop phenology.
- **Our setup partly sidesteps the deep-learning xAI pain points.** Several challenges the paper raises — saliency on raw pixels, patch-level spatial attribution, black-box temporal encoders — are specific to deep nets. Because we classify on *named, engineered* features with a tree model, our explanations are over interpretable quantities (e.g., "EVI slope," "B11 maximum"), which is closer to the "interpretable-by-design / low-level-feature" direction the authors advocate — a qualitative advantage of the lite-learning route worth noting.
- **Temporal dependence is intrinsic to our features.** The paper's emphasis that most xAI ignores temporal structure is relevant because our features *are* temporal summaries; SHAP over them collapses time, so we should be clear that importance is attributed to the summary statistic, not to a specific date.

## Evaluation Caveats

- **DOI unverified.** CVPRW Open Access paper, not Crossref-indexed in the title search; cite as CVPR 2024 Workshops (XAI4CV), pp. 8199–8205, tagged `unverified`. Do not invent a DOI, and do not borrow the companion IEEE-GRSM review's DOI (`10.1109/mgrs.2024.3467001`) for this paper.
- **Review, not a benchmark.** No accuracy/Kappa numbers, no model, no comparator — it informs interpretability *methodology*, not classification performance. Do not treat the 38% SHAP figure as a performance metric; it is a usage frequency from the literature survey.
- **Scope is deep-learning-centric.** The challenges catalogued (patch saliency, spectral CAM, transformer attention) target DNNs; the transferable point to our classical-ML pipeline is the **feature-independence violation under correlated temporal/spectral features**, which fully applies to SHAP regardless of model class.
- **No quantitative xAI evaluation standard.** The paper itself stresses that RS xAI lacks objective evaluation — so its qualitative trend findings (which methods/tasks rose or stagnated) are descriptive, and our own SHAP interpretation inherits the same "no standardized validation" limitation.
- **Counts/percentages are survey artifacts.** Trends depend on the 2017–2023 search window and three databases; the "stagnation of land-cover/agriculture xAI" is relative to a rising baseline of other tasks, not an absolute decline.
