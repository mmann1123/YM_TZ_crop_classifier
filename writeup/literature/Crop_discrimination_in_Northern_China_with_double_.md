# Crop discrimination in Northern China with double cropping systems using Fourier analysis of time-series MODIS data

**Citation:** Zhang Mingwei, Zhou Qingbo, Chen Zhongxin, Liu Jia, Zhou Yong, Cai Chongfa (2008). "Crop discrimination in Northern China with double cropping systems using Fourier analysis of time-series MODIS data." *International Journal of Applied Earth Observation and Geoinformation*, 10(4), 476-485. DOI: `10.1016/j.jag.2007.11.002` (verified via Crossref; title and authors match exactly).

## Objectives

Map the area and distribution of maize and cotton across the North China Plain (Beijing, Tianjin, Hebei, Shandong, Henan; ~539,508 km2), a region with mixed single- and double-cropping systems. The methodological objective is to estimate regional crop acreage from coarse-resolution (500 m, 8-day) MODIS NDVI time series by applying the Fast Fourier Transform (FFT) to extract phenological signatures — amplitude and phase of the dominant harmonic terms — and use those engineered features to discriminate cropping systems and then crop types.

## Methods

- **Sensor / data:** MODIS 8-day composite Surface Reflectance (`MOD09A1`), 500 m, seven bands, forty-six 8-day composites for the year 2004. NDVI computed per composite from red and NIR.
- **Preprocessing:** Cloud masking from MODIS QC flags plus a blue-reflectance threshold; gap-filled and denoised with a Savitzky-Golay filter (two parameters: smoothing-window half-width `m` and polynomial degree `d`). Reprojected from sinusoidal to UTM, resampled to a uniform 500 m.
- **Feature engineering (the part relevant to us):** Per-pixel FFT of the NDVI time series. The real and imaginary components are recast as **amplitude** (in NDVI units) and **phase** (timing) of harmonic terms 0-3 (Table 1). These FFT amplitude/phase coefficients are the classification features. This is an explicit precedent for representing a temporal NDVI curve by a small set of engineered descriptors rather than feeding the raw sequence to a learner — conceptually adjacent to our `xr_fresh` tsfresh-style statistics on `EVI`, `B2`, `B6`, `B11`, `B12`, `hue`, though FFT harmonics are a different (frequency-domain, fixed-basis) feature family than slope/skewness/number-of-peaks/complexity statistics.
- **Classifier:** Two-stage, non-ML by modern standards. (1) Unsupervised ISODATA clustering on amplitude terms 0-3 (30 clusters) to separate single- vs double-cropping systems; clusters merged/labeled against reference polygons. (2) Supervised maximum-likelihood classification on selected amplitude/phase terms to split spring maize vs cotton (single-crop areas) and winter-wheat-maize vs winter-wheat-cotton (double-crop areas). Software: ERDAS 8.5; IDL for the FFT.
- **Reference data:** 69 GPS-surveyed field polygons (~1 km2 each, single crop per field) collected in 2005; only ~4 polygons used to derive representative NDVI patterns / spectral signatures.
- **Evaluation protocol (load-bearing):** There is **no pixel-level or field-level classification accuracy assessment at all** — no confusion matrix, no holdout, no per-class precision/recall, no k-fold. Validation is **area-aggregate agreement against agricultural census statistics**: province-level percent agreement (Table 2) and county-level regression of mapped area vs statistical area (cotton R2 = 0.84, RMSE = 48.11 km2; maize R2 = 0.71, RMSE = 82.00 km2). This sits *below* even the lowest rung of the spectrum our manuscript is built around (pooled random k-fold over pixels): it never measures per-pixel labeling correctness, only how well aggregated mapped acreage correlates with an independent census tally for the same year and region. It is therefore not comparable to our field-grouped-CV Cohen's Kappa or per-class F1.

## Key Findings

- Single-cropping NDVI profiles are unimodal; double-cropping profiles are tri-modal, so the number/distribution of harmonic amplitude is diagnostic of cropping system.
- Amplitude of term 1 dominates in single-crop areas; terms 1-3 are comparable in double-crop areas. Cotton's longer growing season and earlier, higher NDVI peak relative to spring maize is captured in term-0 amplitude and term-1/2 phase, enabling crop separation.
- Mapped cotton and maize acreage agrees well with census data at county and (mostly) province level, except Beijing (underestimated) and Tianjin (cotton overestimated), attributed to small fragmented fields and sub-pixel mixing at 500 m.
- The authors conclude FFT-of-NDVI is "promising" for regional double-cropping crop mapping and note the method requires a full annual data cycle.

## Relevance to Our Crop-Classification Study

- **Engineered-temporal-feature precedent.** This is an early, citable instance of the core "lite learning" idea: collapse a noisy multi-temporal vegetation-index series into a handful of interpretable engineered descriptors (here FFT amplitude/phase) and classify on those, instead of using the raw sequence with a heavy learner. It supports the lineage argument that temporal-feature engineering is a long-standing, effective representation for crop phenology. It also makes our feature design look richer by contrast: the paper relies on NDVI alone, no SWIR/red-edge, no SAR.
- **Smallholder/fragmentation caution.** Its main failure mode — sub-pixel mixing of small fragmented fields at 500 m — is precisely the smallholder problem our 10 m Sentinel-2 pipeline targets. It is evidence for why coarse-resolution area estimation is inadequate for the fragmented sub-Saharan smallholder landscape, motivating finer resolution and field-level labels.
- **Target-crop overlap is partial.** It maps maize and cotton (two of our priority crops) but in a temperate double-cropping system unlike northern Tanzania's Masika regime; rice, sorghum, millet, sunflower, cassava are absent.
- **Use as a contrast, not a comparator.** Because it never reports classification accuracy, it cannot anchor any accuracy claim. Its value is conceptual (frequency-domain temporal feature engineering) and as a cautionary tale on coarse-resolution area-only validation.

## Evaluation Caveats

- **No classification accuracy whatsoever.** No confusion matrix, no holdout, no per-class or balanced metric (no macro-F1, no Kappa, no per-class recall). Only area-vs-census agreement — an aggregate measure that can hide compensating per-pixel errors (commission and omission canceling within an aggregation unit). This places it well below our evaluation bar.
- **No spatial-transfer or cross-year test.** Single year (2004), single region; the census "validation" is for the same year and area, so it measures area-tallying agreement, not generalization to a new tile, region, agroecological zone, or season.
- **Class-imbalance / minority-crop performance unmeasured.** Only two crop classes (plus their wheat rotations) are mapped; no minority-crop recall analysis. The hard minority crops central to our study (cassava, millet, sunflower, sorghum) are not addressed.
- **Spatial leakage not assessable.** No train/test split is reported for the supervised step beyond cluster labeling on a handful of reference polygons, so there is no leakage-controlled accuracy figure to cite either way. The reference set is tiny (69 polygons, ~4 used as signatures).
- **Coarse resolution (500 m)** is structurally unsuited to smallholder fields; the authors themselves call for finer resolution.
- **Optical-only, fixed-basis features.** NDVI only; no SAR, no SWIR/red-edge; FFT is a fixed frequency-domain basis rather than the broader statistical feature library used by `xr_fresh`.
