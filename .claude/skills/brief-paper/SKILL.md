---
name: brief-paper
description: Create a per-paper briefer markdown file for a crop-classification / remote-sensing PDF in writeup/literature/. Use when asked to brief a paper, summarize a PDF, add a paper to the literature folder, or create a literature briefer.
---

Generate one markdown briefer per PDF in `writeup/literature/`. This SKILL.md is the canonical, self-contained source — follow the template and rules below exactly.

This corpus supports the manuscript `writeup/writeup.md` — the IEEE JSTARS paper "Lite Learning: Efficient Crop Classification in Tanzania Using Feature Extraction with Machine Learning & Crowd Sourcing" (Mann et al.). The study classifies smallholder crop types across northern Tanzania from multi-temporal Sentinel-2 imagery using classical machine learning (LightGBM, Random Forest) trained on **engineered time-series features** — tsfresh-style statistics (mean, max, min, slope, skewness, number of peaks, complexity) computed per pixel by the custom `xr_fresh` library on `EVI`, `B2`, `B6` (red edge), `B11`/`B12` (SWIR), and `hue` — rather than on raw temporal sequences fed to deep RNN/CNN/patch networks. Its thesis ("lite learning"): cheap, interpretable, GPU-free feature extraction plus tree-based ML, validated with field-disjoint grouped cross-validation (`StratifiedGroupKFold` on `field_id`, `Field_size`-weighted) and explained with SHAP, matches or beats heavier deep-learning models and generic "agriculture"-only land-cover maps in data-scarce sub-Saharan settings (Cohen's Kappa 0.82, F1-micro 0.85). Ground truth is crowdsourced via the YouthMappers network. Target crops: maize (Masika), cotton, rice, sorghum, millet, sunflower, cassava (the latter five are the hard minority classes); plus urban, forest, shrub, tidal, water. Briefers should be written through that lens — how does each paper inform the deep-learning-vs-engineered-feature trade-off, interpretability, minority-crop performance, SAR/optical and band/index choice, smallholder data scarcity, and evaluation rigor against spatial leakage? Note that this paper's own evaluation sits at the **field-wise grouped-CV-within-one-region** rung — it does not use a spatially disjoint holdout — so calibrate comparators against that bar and flag any that do something stronger (true spatial or cross-domain/cross-sensor transfer).

## When to invoke

The user names one or more PDFs (paths or filenames matching `writeup/literature/*.pdf`) and asks for a briefer / summary / synopsis. If asked to brief many papers at once (N ≥ 5), parallelize via sub-agents, one agent per group of ~5 PDFs, each agent following this same template.

## What you produce

For each PDF, one markdown file at `writeup/literature/<same-base-name>.md` — same base filename as the PDF, `.pdf` replaced by `.md`, all other characters preserved (spaces, punctuation, em-dashes). Each briefer uses this exact section template:

- `# {Paper Title}`
- `**Citation:**` (authors, year, venue, DOI)
- `## Objectives`
- `## Methods` (including the load-bearing **Evaluation protocol** line — see rule 1)
- `## Key Findings`
- `## Relevance to Our Crop-Classification Study`
- `## Evaluation Caveats`

## Authoring rules

1. **Evaluation protocol is the load-bearing section.** State precisely how the paper measured generalization, and where it sits on the spectrum that this manuscript is built around: pooled/random k-fold over all pixels → field-wise (FID) k-fold within one scene → a spatially disjoint holdout (a different tile, region, or agroecological zone) → cross-year or cross-sensor transfer. Most crop-classification papers report the first two and call them "test-set" or "out-of-sample" accuracy; flag in-region-k-fold-disguised-as-OOS explicitly, because it is exactly the conflation our field-disjoint grouped CV is designed to avoid. Calibrate every comparator against our bar — field-wise grouped CV within one region (not a spatial holdout) — and flag any paper that clears a higher bar (true spatial or cross-domain/cross-sensor transfer).
2. **Flag spatial leakage.** The cardinal sin is pixels from the same field (or adjacent, spatially autocorrelated pixels) appearing in both train and test, which inflates accuracy without measuring transfer. Note whether splitting was done field-wise (FID-disjoint) and whether train/test are spatially separated at all.
3. **Flag class-imbalance artifacts.** Crop datasets are dominated by a few majority classes (here maize and the non-crop classes; the hard minority crops are cassava, millet, sunflower, sorghum, and cotton). Note overall-accuracy figures that mask poor minority-crop recall, and whether the paper reports a balanced metric (macro-F1, per-class F1, Cohen's Kappa) rather than only accuracy or weighted F1.
4. **Cite the protocol, not the headline F1.** e.g. "Their 0.92 overall accuracy comes from random k-fold over pooled pixels within a single scene and is not comparable to our field-grouped-CV Cohen's Kappa / per-class F1."
5. **Note what the paper does NOT measure.** Spatial transfer, cross-year robustness, per-class minority performance, and computational cost are frequently absent — silences are findings.
6. **Note feature and sensor design** relevant to us: optical-only vs SAR fusion, band/index choice, raw temporal sequences vs engineered time-series features (e.g. xr_fresh-style statistics), pixel vs field vs patch granularity.
7. **For reviews/theory papers**, skip the protocol question and focus Relevance on methodological/conceptual implications (e.g. inductive bias, domain transfer, interpretability).
8. **Markdown only. No emojis. Identifiers and band names in backticks.** Roughly 250–400 lines per briefer.
9. **Unparseable PDF**: write a stub with `## Status: Could not extract` rather than skipping.
10. **DOI extraction and verification (required).** Extract the DOI from the PDF (first page, header/footer, or abstract block — formats `10.xxxx/...`, `https://doi.org/...`, `DOI:`). Verify it by calling `WebFetch` on `https://api.crossref.org/works/<doi>` and confirming the returned `title` matches (case-insensitive substring is fine). If Crossref 404s or the title diverges, tag the DOI `unverified` and add a one-line note. If no DOI is in the PDF, try one Crossref title search (`https://api.crossref.org/works?query.title=<title>&rows=3`); if a top hit matches, use it tagged `(from Crossref title match)`. Otherwise write `DOI: not found in PDF`. Never fabricate a DOI. The first place to look for an existing DOI is `writeup/refs.bib` (the pandoc bibliography for the manuscript) — but treat any DOI found there as a lead, not ground truth: still verify it against Crossref before trusting it. This project's bibliography has already contained fabricated, duplicate-key, and wrong-metadata entries (e.g. a `xun2021novel` entry carrying an unrelated DOI, and entries with truncated or mis-attributed authors), so a DOI being present in the `.bib` is not evidence that it is correct.

## After writing the briefer(s)

If a new briefer materially changes the set of directly comparable papers (a new true-spatial-holdout comparator, or a feature/architecture finding that bears on the paper's argument), follow up with [brief-synthesize-crop-findings](../brief-synthesize-crop-findings/SKILL.md) to refresh `writeup/literature/0_findings_summary.md`. Otherwise the briefer alone suffices.

## Reference

- Project context to inject: [CLAUDE.md](../../../CLAUDE.md) and the manuscript [writeup/writeup.md](../../../writeup/writeup.md).
- Cite keys / DOIs for the corpus: [writeup/refs.bib](../../../writeup/refs.bib) (verify every DOI against Crossref — see rule 10; this `.bib` has carried wrong entries).
- Especially relevant comparators in the corpus: "Deep Architectures Fail to Generalize: A Lightweight Alternative for Agricultural Domain Transfer in Hyperspectral Images" (Pankajakshan et al., 2025, `10.3390/s26010174`) — a true cross-scene/cross-sensor transfer study, the strongest on-point comparator; "Smallholder maize area and yield mapping at national scales with Google Earth Engine" (Jin et al., 2019, `10.1016/j.rse.2019.04.016`) — same-country (Tanzania/Kenya), smallholder, group-CV evaluation; "How accurate are existing land cover maps for agriculture in Sub-Saharan Africa?" (Kerner et al., 2024, `10.1038/s41597-024-03306-z`) — Tanzania-specific benchmark for the "beat the generic agriculture map" claim; "Satellite Imagery Analysis for Crop Type Segmentation Using U-Net Architecture" (Kenya CV4A) — Global-South deep-learning comparator; "CropHarvest" — a data-scarce crop-typing benchmark that includes a Tanzania rice set.
- Sibling skills: [brief-literature-review](../brief-literature-review/SKILL.md), [brief-synthesize-crop-findings](../brief-synthesize-crop-findings/SKILL.md), [briefer-bibliography](../briefer-bibliography/SKILL.md).
