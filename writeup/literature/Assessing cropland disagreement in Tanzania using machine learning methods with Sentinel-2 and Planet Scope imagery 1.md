# Assessing cropland disagreement in Tanzania using machine learning methods with Sentinel-2 and Planet Scope imagery

**Citation:** Ogweno, V. W., Moore, N., & Wanyama, D. (2023). Assessing cropland disagreement in Tanzania using machine learning methods with Sentinel-2 and Planet Scope imagery. *International Journal of Remote Sensing*, 44(21), 6716–6735. DOI: `10.1080/01431161.2023.2274320` (verified via Crossref — title, authors, year, journal, volume, issue, and pages all match).

> Source PDF is a scanned/image-only document (no extractable text layer); this briefer was built from rendered page images, so figures and table cells are read from raster scans.

## Objectives

Quantify and locate **disagreement among existing land-cover/cropland products** in two Tanzanian districts, and test whether locally trained machine-learning classifiers on high-resolution imagery can produce more reliable cropland maps than coarse global/regional products (USGS cropland mask, Copernicus, MODIS, GlobCover-type maps), which disagree badly in heterogeneous smallholder landscapes. The framing is a cropland-vs-non-cropland reliability assessment to guide where future ground validation should be focused — not multi-crop typing.

## Methods

- **Study area.** Kilombero and Ulanga districts, **Morogoro region, southwestern/central Tanzania** (Kilombero River floodplain) — NOT the northern Arusha/Dodoma/Mwanza area of our manuscript. Smallholder, rainfed, with rice as a critical regional crop.
- **Imagery.** PlanetScope (4 m, NICFI bi-annual median composites via GEE) and Sentinel-2 (10 m, Level-2A SR), 2017–2018 two-year median composites. Features: spectral bands 2/3/4/8, TCI, AOT, WVP; five vegetation indices (NDVI, EVI, BSI, NDWI, NDBI); plus SRTM elevation/slope and GLCM textural features.
- **Classes.** Imagery is classified into **seven classes** then reclassified to the **binary target of cropland vs non-cropland** (non-cropland aggregating forest, water, woodland, wetland, built-up, bare soil). The core map product is binary cropland — there is no per-crop typing.
- **Classifiers.** CART, Random Forest (RF), and Support Vector Machine (SVM), tuned on GEE; CART used 10-fold cross-validation for pruning.
- **Evaluation protocol (load-bearing — and the key foil for our manuscript).** 5,504 samples were drawn over the entire study area by a stratified random sampling criterion, then **split 70% training / 30% independent validation by random pixel split**. There is **no field-level grouping**: pixels are the sampling unit and the split is random across the scene, so spatially autocorrelated and same-field pixels can fall on both sides of the 70/30 line. This sits at the **pooled/random k-fold-over-pixels-within-one-scene** rung — the weakest position on the generalization spectrum, and exactly the conflation our `StratifiedGroupKFold`-on-`field_id` design exists to avoid. The "independent validation" label means held-out-pixels, not held-out-fields or held-out-region. No spatial holdout, no cross-year, no cross-sensor transfer (PlanetScope vs Sentinel-2 are compared as inputs, not as a transfer test). Reported metrics are overall accuracy from confusion matrices (and the cross-product disagreement maps).

## Key Findings

- **Binary cropland accuracy, from Classification-and-Regression results:** PlanetScope (4 m) 93% / 89% / 83% and Sentinel-2 (10 m) 91% / 86% / 76% across the three classifiers — higher-resolution PlanetScope consistently outperformed Sentinel-2 for cropland detection.
- **SVM was the most reliable classifier on the high-resolution imagery**, improving crop forecasts relative to CART/RF.
- **Product disagreement is large:** on average **73% of pixels were consistently classified across the existing products while 27% were misclassified/inconsistent**, with disagreement concentrated in transition zones (small fields, grassland/wetland/cropland margins) where reflectance differences are subtle.
- **Conclusion drawn:** existing global/regional cropland products are inadequate for this heterogeneous floodplain; locally trained high-resolution classifiers and targeted ground validation are needed.

## Relevance to Our Crop-Classification Study

This is the **same-country foil** — a Tanzanian cropland-mapping paper that makes precisely the methodological choices our manuscript is designed to improve on, which makes it a useful contrast rather than a comparator. Three contrasts to draw explicitly:

1. **Evaluation rigor.** It uses a **random 70/30 pixel split with no field grouping**; we use field-disjoint `StratifiedGroupKFold`. Its overall accuracies (e.g., 91–93%) are therefore inflated by within-field/spatial-autocorrelation leakage and are **not comparable** to our field-grouped Cohen's Kappa / per-class F1 — a clean illustration of skill rule 4.
2. **Task difficulty.** It is **binary cropland vs non-cropland**; we do multi-class crop typing including the hard minority crops. Its high numbers come from the much easier problem, reinforcing our manuscript's point that beating a generic "agriculture" class is an easier task than per-crop classification.
3. **Geography.** It is **central/southwestern Tanzania (Kilombero/Ulanga, Morogoro)**, not our northern study area, so even its cropland conclusions do not transfer regionally — and it makes no claim to spatial transfer.

It does, however, supply independent same-country evidence that (a) existing global cropland products disagree substantially (motivating our "beat the generic agriculture map" framing alongside Kerner et al. 2024) and (b) higher spatial resolution (PlanetScope 4 m) helps in smallholder mosaics — relevant to our discussion of cassava/maize confusion at 10 m.

## Evaluation Caveats

- **Random pixel split = spatial leakage.** No field-disjoint or spatially disjoint holdout; same-field and neighboring pixels can straddle train/test, inflating the 76–93% figures. This is the cardinal sin (skill rule 2) and the headline reason its accuracies overstate transferable performance.
- **No balanced or per-class minority metric.** Reports overall accuracy on a binary task; no Cohen's Kappa across a multi-class confusion matrix, no per-crop recall — so class-imbalance artifacts within "cropland" are invisible. The binary framing sidesteps the minority-crop problem entirely.
- **Reference data quality.** Samples are drawn over the study area by stratified random criterion and verified against high-resolution imagery, not systematic field ground truth; cropland labels at transition zones (where 27% disagreement lives) are the least certain.
- **Single region, single epoch.** 2017–2018 two-year composites in one floodplain; no cross-year or cross-region test, so even within Tanzania the result is local.
- **Optical-only.** No SAR; persistent cloud in the floodplain is a stated reason for using two-year median composites, which smear phenology.
