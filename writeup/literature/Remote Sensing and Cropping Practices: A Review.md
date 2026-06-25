# Remote Sensing and Cropping Practices: A Review

**Citation:** Bégué, A.; Arvor, D.; Bellon, B.; Betbeder, J.; de Abelleyra, D.; Ferraz, R.P.D.; Lebourgeois, V.; Lelong, C.; Simões, M.; Verón, S.R. (2018). "Remote Sensing and Cropping Practices: A Review." *Remote Sensing*, 10(1), 99. DOI: `10.3390/rs10010099` (verified against Crossref — title "Remote Sensing and Cropping Practices: A Review", first author Bégué, *Remote Sensing*, 2018, all match). This is the `begue2018remote` entry in `refs.bib`.

> This is a review paper. Per authoring rule 7, the per-study evaluation-protocol question is skipped; the Relevance section focuses on methodological and conceptual implications — in particular, this review is the canonical precedent for *engineered phenological time-series features*, the lineage from which our `xr_fresh` approach descends.

## Objectives

- Survey the remote-sensing literature on mapping *cropping practices* (as distinct from crop-type mapping, which the authors explicitly exclude as already reviewed elsewhere).
- Organize the field into a clean agronomic typology with three categories: **crop succession** (rotation, fallowing), **cropping pattern** (single-tree planting pattern, sequential cropping, intercropping/agroforestry), and **cropping techniques** (irrigation, soil tillage, harvest/post-harvest practices, crop varieties, agro-ecological infrastructures).
- For each practice, synthesize the agronomic/environmental/socio-economic stakes and the RS methods used to detect it, and recommend future research directions for the new (Sentinel-era) generation of Earth-observation systems.

## Methods

- A structured narrative review (not a meta-analysis). The authors group representative studies by their typology and, for each practice, discuss sensors, indices, and analytical methods.
- Heavy emphasis on **time-series phenology** as the core analytical engine for cropping-practice detection, since most practices (rotation, fallow, sequential cropping, irrigation timing) express themselves temporally rather than in a single-date spectral signature.
- Survey of the engineered-feature and time-series-matching methods that recur across the literature:
  - **Engineered phenological metrics from VI time series.** The review highlights Morton et al., who computed **36 metrics** from MODIS vegetation-index time series — including `NDVI` and `EVI` minimum, maximum, mean, and related descriptors — as inputs to classification. This is the direct intellectual ancestor of statistic-based time-series feature extraction.
  - **Frequency / shape transforms.** Local Fourier analysis (Delenne et al.) and wavelet analysis of image texture and temporal profiles (Aksoy; Lefebvre et al.) to characterize periodicity, orientation, and scale of cropping patterns and to discriminate single vs. double cropping.
  - **Temporal shape matching.** Dynamic Time Warping (DTW) with temporal weights for classifying land-cover and single/double cropping systems by matching the shape of MODIS `EVI` time series.
  - **Bag-of-temporal features.** Bailly et al.'s Dense Bag-of-Temporal-SIFT-Words approach — encoding salient temporal events/peaks in the VI sequence as a feature vocabulary.
- Sensor discussion spans coarse high-temporal sensors (MODIS, SPOT-VGT) for phenology, high-spatial sensors (Ikonos, QuickBird, WorldView, Pléiades) for intra-field structure, and the then-new Sentinel constellation (including Sentinel-1 SAR's dense, weather-independent time series) as the expected enabler going forward.

## Key Findings

- Most cropping-practice RS studies are exploratory, local-scale, single-sensor, and heavily dependent on ground data and local agronomic knowledge — they do not generalize off the shelf. The review calls for land stratification, multi-sensor fusion, and expert-knowledge-driven methods to scale up.
- **Engineered time-series features are the established route** to turning multi-date imagery into discriminative inputs: VI-derived statistics (min/max/mean/amplitude/slope-like descriptors), Fourier/wavelet transforms, DTW shape matching, and temporal bag-of-features all recur as ways to compress a phenological sequence into classifier-ready features.
- **Intercropping is flagged as the least tractable optical-RS problem.** "Few remote sensing studies address intercropping" because the intra-field variability of mixed crops sits at an infra-metric scale below most pixels; discriminating intercrops "based only on their spectral signature is almost impossible," and the few successes rely on sub-1 m imagery (Ikonos/QuickBird/WorldView), Haralick texture, and object-based methods that detect individual tree crowns/rows. Agroforestry is treated as the only partially tractable subcase.
- The new spatial/temporal density of Sentinel-1/-2 is identified as the key opportunity for operational cropping-practice monitoring in the food-security context.

## Relevance to Our Crop-Classification Study

- **Direct lineage for `xr_fresh`.** Our pipeline computes tsfresh-style statistics — mean, max, min, slope, skewness, number of peaks, complexity — per pixel on `EVI`, `B2`, `B6`, `B11`/`B12`, and `hue` time series. This review documents the precedent: Morton's 36-metric MODIS-VI feature set, plus Fourier/wavelet/DTW/bag-of-temporal-SIFT, are the literature's canonical "engineer the phenology into features, then classify" recipe. We can cite this paper to ground the *idea* that compressing a temporal sequence into engineered descriptors (rather than feeding raw sequences to a deep RNN/CNN) is a long-established, well-motivated strategy — supporting the "lite learning" thesis with a high-citation review rather than only with our own results.
- **`number of peaks` and `complexity` map onto known methods.** Our peak-counting and complexity statistics are direct relatives of the bag-of-temporal-SIFT "salient temporal event" encoding and the wavelet/Fourier periodicity descriptors the review catalogs — useful provenance when justifying our specific feature choices.
- **Sets realistic expectations for our hard classes.** The review's verdict that intercropping is near-intractable with optical RS at coarse-to-medium resolution is a strong prior for why some of our minority crops (often grown intercropped in smallholder Tanzania) are the difficult ones. It frames our minority-crop difficulty as a known structural limit of medium-resolution optical RS, not merely a modeling shortfall — and motivates field-size-aware sampling (our `Field_size`-based buffering and weighting) as a partial mitigation.
- **Endorses multi-sensor and stratification, which we partly forgo.** The review's recommended directions (multi-sensor fusion incl. SAR, land stratification, expert knowledge) are exactly the levers our optical-only, single-region "lite" design deliberately does not pull. Citing it lets us position our work honestly: we trade some of those levers for cheapness/interpretability and test whether engineered optical features alone suffice.

## Evaluation Caveats

- **Review, not a benchmark.** It reports no single accuracy figure of its own and provides no comparator on our field-grouped-CV / Cohen's-Kappa bar. The numbers it cites belong to heterogeneous primary studies with their own (mostly local, often leakage-prone) protocols and must not be quoted as benchmarks against us.
- **Scope is cropping *practices*, not crop *type*.** The authors explicitly exclude crop-type mapping. Its relevance to us is methodological (feature-engineering lineage, intercropping difficulty), not a direct head-to-head on crop-type classification accuracy.
- **Pre-deep-learning vintage (2018).** Written before the deep-temporal-network wave crested, so it does not adjudicate the engineered-feature-vs-deep-learning trade-off our manuscript is built around — it simply establishes that the engineered-feature branch is mature and well-founded.
- **Coarse-resolution emphasis.** Much of the cited phenology work is MODIS-scale (250 m–1 km); its conclusions about feature methods transfer to our Sentinel-2 10 m setting, but its specific intercropping-intractability thresholds were set at coarser resolution and may be somewhat looser at 10 m.
