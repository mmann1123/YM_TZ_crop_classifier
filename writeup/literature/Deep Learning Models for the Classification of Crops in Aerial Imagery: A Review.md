# Deep Learning Models for the Classification of Crops in Aerial Imagery: A Review

**Citation:** Teixeira, I., Morais, R., Sousa, J. J., & Cunha, A. (2023). "Deep Learning Models for the Classification of Crops in Aerial Imagery: A Review." *Agriculture*, 13(5), 965. DOI: `10.3390/agriculture13050965` (verified via Crossref; title, authors, journal match exactly).

> Review paper (PRISMA-style systematic review of 36 studies) — per skill rule 7, focus is on the methodological/quantitative implications rather than a single evaluation protocol. This is the **best quantitative anchor in the corpus for "classical ML wins specifically in small-data regimes."**

## Objectives

Systematically review (PRISMA protocol) deep-learning models for crop classification from aerial/satellite/UAV imagery published 2020-2022, to answer four questions: (1) which DL architectures are commonly used; (2) **how DL performance compares to classical machine learning (ML)**; (3) what aerial imagery/data sources are used; and (4) how the number of classes affects performance. 36 studies passed full-text screening (from 262 retrieved).

## Methods

- **Selection:** Harzing's Publish-or-Perish over Google Scholar + Scopus, keywords "image" / "crop classification" / "deep learning"; 2020-2022, peer-reviewed, English; excluded leaf/trunk/disease-focused work. PRISMA flow: 262 -> 166 (dedup) -> 55 -> 44 (full text) -> **36 included**.
- **Synthesis:** studies grouped by data source — satellite (Table 1), UAV (Table 2), and multisource satellite+UAV/aircraft (Table 3) — with model architectures and class counts tabulated. Narrative comparison of OA / AA / Kappa / F1 across studies.
- **Architectures surveyed:** CNNs (2D/3D), LSTM/Bi-LSTM, transformers (ViT), hybrid CNN-RNN and CNN-transformer, object-based CNNs (OCNN/SS-OCNN/TS-OCNN), plus fusion (optical+SAR) and transfer learning.
- **Evaluation-protocol framing (for a review):** The review reports each study's headline metrics as published; it does **not** harmonize protocols or check for spatial leakage / field-grouped splits. Many of the cited studies use random pixel splits within a scene (e.g. the recurring "3% train / 0.1% val / 96.9% test by random sampling" pattern, or stratified 10-fold CV over pooled pixels). So the comparison is between self-reported, mostly in-scene accuracies. The analytically load-bearing output is not any single number but the **win/loss tally of DL vs ML across the 36 studies** and the stated reason for the lone ML win.

## Key Findings

- **THE anchor result:** "**Of the 36 articles analysed, the machine learning approach only outperformed the deep learning methods in one study. The authors suggested that it was due to the small size of the dataset.**" So DL beat ML in **35 of 36** studies, and the **single ML win is explicitly attributed to small dataset size.** This is the cleanest citable evidence that the ML-beats-DL outcome is a *small-data-regime* phenomenon — directly supporting our thesis that classical ML wins precisely where labels are scarce (our sub-Saharan smallholder setting), while conceding DL's general dominance when data are plentiful.
- The single ML-win study (the review's reference [31], an India crop-classification study on Sentinel-2 NDVI time series) found an **SVM gave the strongest agreement with ground-surveyed crop areas (95.9% agreement, F1 = 0.994), beating CNN/RNN under stratified 10-fold CV**, with the authors attributing the traditional-model advantage to the limited training-set size. This is essentially a miniature of our argument.
- Class-count and non-crop classes matter: model performance degrades when classes share phenology, and **including non-crop classes improves overall accuracy** — relevant to our inclusion of urban/forest/shrub/tidal/water alongside crops.
- Data-source patterns: Sentinel-2 is the most common satellite source; **optical+SAR fusion (Sentinel-1+2) repeatedly improves accuracy** over optical-alone (e.g. LSTM 93.7% optical -> 97.5% fused; sugarcane NDVI+VH fusion preventing algae/water confusion); higher spatial resolution and balanced train/val ratios raise accuracy.
- Architecture takeaways: CNNs dominate; multitemporal/time-series modeling (PSE+LTAE, CNN-transformer for phenology, two-stream temporal-attention) and transfer learning are recurring strategies to cope with limited labels; ensembling/multisource fusion generally helps.
- Reported headline accuracies are frequently very high (many >95-99% OA), but typically on random in-scene splits with tiny training fractions — see caveats.

## Relevance to Our Crop-Classification Study

- **Primary citation for our central claim.** "DL beat ML in 35/36 studies; the only ML win was attributed to small dataset size" is the best quantitative external support for framing our classical-ML approach as the right tool *for the data-scarce regime specifically*, while honestly acknowledging DL's general edge when data are abundant. It lets us argue "we are deliberately operating in the regime where ML competes," not "ML is generally better."
- **The one ML-win study mirrors us.** That study used **SVM on Sentinel-2 NDVI time-series features**, validated against ground-surveyed crop areas, and beat CNN/RNN because of limited data — a close analogue to our LightGBM/RF-on-engineered-Sentinel-2-features design and crowdsourced ground truth. Strong supporting precedent.
- **Supports our design choices:** non-crop classes improving OA (we include urban/forest/shrub/tidal/water); optical+SAR fusion gains (a lever we forgo for cost — useful to acknowledge what we leave on the table); multitemporal modeling being a standard accuracy driver (our `xr_fresh` temporal features).
- **Calibration / honesty:** because the review's 35/36 tally is over uncontrolled, mostly in-scene protocols, we should cite it for the *direction and the small-data caveat*, not as proof DL is "91% accurate" or similar — the headline numbers are not leakage-clean (see caveats).

## Evaluation Caveats

- **Headline accuracies are protocol-uncontrolled and often leakage-prone.** Many surveyed studies use **random pixel splits within a single scene** (the "3%/0.1%/96.9% random sampling" and "stratified 10-fold over pooled pixels" patterns recur), which our field-grouped CV is specifically designed to avoid. The review does not flag or correct for spatial autocorrelation / field-disjointness, so its many >95-99% OA figures are not comparable to our field-grouped Cohen's Kappa / per-class F1 and should not be cited as if they were.
- **No spatial-transfer or cross-year evaluation is synthesized.** The review reports in-scene/benchmark accuracies; only isolated studies (e.g. a Landsat-8 model "tested on new data from Fayette and Pickaway County, OA 81%") touch cross-region transfer. As a body, the evidence sits at the pooled/in-scene end of our spectrum — none of it clears a higher bar than ours.
- **Class imbalance largely hidden by OA.** Most cited results are reported as OA (sometimes AA/Kappa/F1); the review notes phenologically similar classes hurt accuracy but does not systematically extract per-class minority-crop recall, so minority-crop performance across studies is not characterizable from it.
- **The 35/36 tally is a vote count over heterogeneous studies, not a controlled benchmark.** It is strong directional evidence, but the comparisons were made under each study's own (varying) conditions; the value is the *consistency* of the direction and the explicit small-data attribution of the lone exception, not a rigorous effect size.
- **Scope window 2020-2022, peer-reviewed only**; predates the most recent transfer-focused work (e.g. the Pankajakshan 2025 cross-sensor study in this corpus), so it does not address true cross-domain generalization.
