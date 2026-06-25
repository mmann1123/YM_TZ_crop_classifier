# Exploring the impact of socioeconomic factors on land use and cover changes in Dar es Salaam, Tanzania: a remote sensing and GIS approach

**Citation:** Simon, O., Lyimo, J., & Yamungu, N. (2024). Exploring the impact of socioeconomic factors on land use and cover changes in Dar es Salaam, Tanzania: a remote sensing and GIS approach. *Arabian Journal of Geosciences*, 17(99). DOI: `10.1007/s12517-024-11908-5` (verified against Crossref: title, journal, authors, and year all match).

## Paper type

This is primarily a **socioeconomic-driver / land-change study with an embedded land-use/land-cover (LULC) classifier** — not a crop-type classification study. Its scientific contribution is a geographically weighted regression (GWR) linking LULC change in Dar es Salaam (1995–2022) to socioeconomic variables. However, it contains a genuine supervised **Random Forest LULC classification** that produces the change maps, and that classifier *is* evaluated with overall accuracy and Cohen's Kappa, so the evaluation-protocol lens applies to that component. Per rule 7, the Relevance section centers on Tanzanian study-area context, land-use drivers, and the smallholder/urban-frontier landscape, while flagging the classifier's leakage and imbalance risks.

## Objectives

- Quantify LULC change in the Dar es Salaam metropolitan area across three epochs (1995, 2009, 2022) using Landsat imagery.
- Use **geographically weighted regression (GWR)** to model *how* socioeconomic factors (population density, distance to city center, GDP, distance to roads, poverty) spatially drive different *types* of LULC change, contrasting GWR against a global ordinary-least-squares (OLS) model.
- Identify the dominant drivers of urbanization and agricultural/natural-cover conversion to inform urban-planning policy.

## Methods

- **Study area:** Dar es Salaam, Tanzania's largest city — a rapidly urbanizing coastal megacity (urban land cover expanding ~6%/yr over three decades).
- **Imagery:** Landsat Thematic Mapper for 1995 and 2009, Landsat 8 OLI for 2022 — one date (epoch) per year, **single-date single-image classification**, pre-processed in the Google Earth Engine code editor.
- **Classifier:** **Random Forest** machine learning implemented in **R**, trained on manually digitized training sites delineated in QGIS for seven NAFORMA-aligned classes: agriculture, bare soil, built-up area, bushland, forest, grassland, water.
- **Reference/training data:** 265 fieldwork training points, distributed **highly unevenly** across classes — 90 agriculture, 59 built-up, 41 forest, 35 bushland, 18 grassland, 14 bare soil, and only 8 water.
- **Driver analysis:** post-classification change detection, then GWR (and OLS baseline) relating change types to socioeconomic covariates assembled on a 1×1 km grid (population, World Bank geocoded data, poverty indicators, TANROADS road network), with zonal statistics in ArcGIS.

**Evaluation protocol:** The RF classifier was assessed by an accuracy assessment comparing the classified maps against the collected fieldwork samples, yielding overall accuracies of 81.40% / 88.42% / 81.51% and **Cohen's Kappa of 0.7686 / 0.8585 / 0.7688** for 1995 / 2009 / 2022. This is a **pooled point-based accuracy assessment, not a spatially disjoint holdout.** There is no field-disjoint or FID-grouped split, no spatial separation between training and evaluation samples, and no statement that the accuracy points are independent of the training sites — so this sits at (or below) the lowest rung of our spectrum and is **weaker than our field-grouped cross-validation**. It does not measure spatial transfer, cross-year robustness of the *classifier* (each epoch is trained/assessed independently), or per-class minority recall in a balanced way. Calibrating against our bar: their Kappa figures are headline pooled-accuracy numbers and are **not comparable** to our field-grouped-CV Cohen's Kappa / per-class F1.

## Key Findings

- Between 1995 and 2022, **built-up area rose 14.9%** while **bushland fell 14.6%**; 65.8% of the landscape experienced gains/losses and 34.2% was stable.
- **The single largest land-change transition was bushland → agriculture** (274 km², 25.7% of all change), with agriculture concentrated along the Msimbazi, Mzinga, Kizinga, and Mbezi river plains and along major trunk roads. Agricultural land increased overall while bushland, forest, and bare soil decreased — documenting the conversion of semi-natural cover to cultivation at the urban frontier.
- **GWR outperformed OLS** (R² = 0.73 vs lower), explaining 73% of the spatial variation in LULC change. **Population density and proximity to the city center** were the dominant drivers; GDP and distance to roads were weaker; **poverty was not a significant driver.**
- Driver influence varied spatially and by change type, supporting a spatially explicit (rather than global) modeling approach.

## Relevance to Our Crop-Classification Study

- **Documents the dominant Tanzanian land-change driver we map against.** The headline transition — bushland/semi-natural cover → agriculture — is precisely the dynamic that makes accurate, current crop maps valuable in this setting. It is a strong citation for our motivation that agricultural expansion is reshaping Tanzanian landscapes and that distinguishing cultivated land (and crop types within it) from bushland/grassland is non-trivial. (Note: the study is Dar es Salaam, an *urban* frontier, distinct from our northern smallholder cropping zone, so cite it for the driver narrative, not as a study-area analog.)
- **Single-date imagery is the clear methodological contrast our within-season `xr_fresh` features improve on.** Each epoch is classified from one Landsat scene on spectral signatures alone, with no temporal/phenological information. Agriculture and bushland/grassland are spectrally confusable on a single date — which is exactly the confusion our multi-temporal engineered features (mean, max, slope, number of peaks, etc. over `EVI`, `B2`, `B6`, `B11`, `B12`, `hue`) are designed to resolve via crop phenology. This paper is a concrete illustration of the limitation we advance past.
- **RF on tabular spectral features is the same family as our classical-ML approach** (RF is one of our baselines alongside LightGBM), so it corroborates the broad finding that tree-based classifiers handle Tanzanian LULC well — while our contribution is the engineered-temporal-feature design and the leakage-aware evaluation it lacks.
- **A cautionary example on evaluation rigor.** Its accuracy assessment illustrates exactly the in-region, non-field-disjoint validation our manuscript argues inflates apparent performance; we can cite it (and similar regional LULC studies) as the prevailing-but-flawed evaluation practice that our field-grouped CV corrects.

## Evaluation Caveats

- **Spatial leakage risk (unaddressed).** No FID/field-disjoint splitting and no stated spatial separation between training sites and accuracy-assessment points; spatially autocorrelated pixels can appear on both sides, inflating the reported Kappa. This is the cardinal sin our grouped CV is designed to avoid.
- **Severe class imbalance.** Training points range from 90 (agriculture) down to 8 (water) and 14 (bare soil). Overall accuracy and pooled Kappa can look strong while minority-class recall is poor; the paper reports only overall accuracy + Kappa, not per-class F1 or a balanced metric, so minority performance is invisible. The authors themselves note agriculture and water classes show lower accuracy in some maps.
- **No temporal/phenological features.** Single-date, single-image classification per epoch; no use of within-season time series, so spectrally similar vegetated classes (agriculture vs bushland) are hard to separate — a structural limitation, not just a tuning issue.
- **Classifier transfer not measured.** Each epoch's classifier is trained and evaluated on its own scene; the paper does not test whether a model transfers across years, regions, or sensors. Cross-year robustness of the *classifier* (as opposed to the land-change product) is a silence.
- **Aggregation mismatch in the driver analysis.** The GWR is run on a coarse 1×1 km grid against per-pixel-derived change, a smoothing that limits resolution of fine smallholder-scale change (a caveat for the driver conclusions, separate from the classifier).
