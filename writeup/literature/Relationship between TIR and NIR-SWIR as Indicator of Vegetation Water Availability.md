# Relationship between TIR and NIR-SWIR as Indicator of Vegetation Water Availability

**Citation:** Holzman, M.E.; Rivas, R.E.; Bayala, M.I. (2021). "Relationship between TIR and NIR-SWIR as Indicator of Vegetation Water Availability." *Remote Sensing*, 13(17), 3371. DOI: `10.3390/rs13173371` (verified against Crossref — title "Relationship between TIR and NIR-SWIR as Indicator of Vegetation Water Availability", first author Holzman, *Remote Sensing*, 2021, all match).

> This is a biophysical / spectral-index methods paper, not a crop-classification study. There is no classifier, no train/test split, and no accuracy metric. Per authoring rule 7, the "Evaluation protocol" question does not apply; the Relevance section below focuses on what the paper tells us about *why* our `B11`/`B12` (`SWIR`) and NIR/red-edge features carry crop-water signal, and on the conceptual boundary of an optical-only (Sentinel-2) feature set.

## Objectives

- Establish the physical relationship between near-infrared (`NIR`), short-wave-infrared (`SWIR`) reflectance, and land surface temperature (`LST`, the thermal-infrared / `TIR` channel) as joint indicators of *water availability* for vegetation — going beyond vegetation water *content* alone toward the soil water actually available for evapotranspiration.
- Build a model, parameterizable purely from remotely sensed `NIR`-`SWIR`/`LST` scatterplots, that estimates root-zone water status, not just canopy water content.
- Validate the index against field- and lab-measured vegetation water content and against in-situ soil moisture at multiple depths, then test regional coherence over the Argentine Pampas.

## Methods

- **Targets and ground data:** Vegetation water content, `LST`, and spectral reflectance were measured in the field and laboratory over three crops — soybean, corn (maize), and barley. Leaf fresh weight (`Fw`) and dry weight (`Dw`) were measured through a controlled dehydration process to derive relative water content (`RWC`).
- **Satellite data:** MODIS/Aqua reflectance and `LST` products scale the field relationships to the regional level; consistency between MODIS and the field/lab measurements anchors the proposed model. Landsat 8 `NDVI` is used illustratively for spatial context.
- **Spectral physics exploited:**
  - `SWIR` (1.0–2.5 µm) reflectance is driven by liquid-water absorption — incident radiation in `SWIR` is absorbed by leaf water "with no influence of foliar pigment," making it an early, direct proxy for vegetation water content and stomatal status.
  - `NIR` reflectance responds chiefly to internal leaf/cellular structure and dry-matter content rather than to water per se; it changes only after a prolonged stress process rearranges cellular space.
  - `LST`/`TIR` (8–14 µm) responds to the partition of net radiation into sensible (`H`) and latent (`LE`) heat: as stomata close under deficit, transpirational cooling drops and canopy temperature rises, so `LST` is a fast indicator of evapotranspiration and root-zone soil moisture.
- **Model form:** A `NIR`-`SWIR`/`LST` feature space (analogous to, but distinct from, the classic `NDVI`/`LST` "triangle/trapezoid" method). The authors note prior work substituting a linear `NDVI`/`SWIR`-reflectance relationship for the `NDVI`/`LST` model when relating reflectance to root-zone soil moisture.
- **Validation:** Comparison of the index against soil moisture at different depths yielded `R2 > 0.7`, indicating sensitivity to root-zone (not just surface) water availability; regional maps showed coherence with known surface hydrological processes in the Pampas.

**Evaluation protocol:** Not applicable. This is a physically grounded regression/index validation against in-situ soil moisture and lab `RWC`, not a land-cover classifier with a train/test protocol. There is no spatial-leakage axis on which to place it and no accuracy/Kappa to calibrate against our bar.

## Key Findings

- `SWIR` and `NIR` carry *different* biophysical information: `SWIR` tracks liquid-water content directly and early; `NIR` is governed by leaf dry-matter/structure and lags water changes. Treating them as one "infrared" block discards signal.
- Greenness indices (`NDVI`, and by extension `EVI`) respond to photosynthetic pigments and canopy structure and therefore **lag** water-status changes — they shift only after a sustained stress process alters chlorophyll and cellular arrangement, whereas `SWIR` and `LST` register the deficit much sooner.
- Adding the thermal (`LST`) dimension materially improves estimation of *available* water for evapotranspiration, because `LST` is tied to the latent/sensible-heat partition and stomatal resistance; the combined `NIR`-`SWIR`/`LST` space resolves root-zone water (`R2 > 0.7` vs. soil moisture) that reflectance alone cannot.
- The index is parameterizable from satellite data alone, supporting regional water-stress and hydrological monitoring without dense field instrumentation.

## Relevance to Our Crop-Classification Study

- **Justifies our `SWIR` (`B11`, `B12`) features.** Our `xr_fresh` pipeline extracts time-series statistics from `B11`/`B12` alongside `EVI`, `B2`, `B6` (red edge), and `hue`. This paper supplies the mechanistic reason those `SWIR` features are not redundant with greenness: they encode a *water-driven* signal that pigment/structure indices miss. Crops with distinct water-use strategies or senescence timing (rice's standing water, sorghum/millet drought tolerance, cotton's long water-demanding season) should separate in `SWIR` space in ways `EVI` alone cannot capture.
- **Explains the value of phenological time-series statistics on `SWIR`.** Because `SWIR` responds *earlier* and more directly to water dynamics than `EVI`, the `xr_fresh` temporal descriptors (slope, number of peaks, complexity, min/max) computed on `B11`/`B12` plausibly capture crop-specific drydown and water-stress phenology that distinguishes our hard minority crops. This is a concrete, physically motivated argument for retaining multiple `SWIR`-derived features rather than collapsing to a single index — relevant to the SHAP-based feature-selection stage.
- **`NIR`/red-edge complementarity.** The `NIR`-is-dry-matter / `SWIR`-is-water distinction reinforces using `B6` (red edge) and `SWIR` together: they probe orthogonal biophysical axes (structure/chlorophyll vs. water), exactly the kind of feature diversity tree-based models (LightGBM, RF) exploit.
- **Names the gap in our sensor design.** The thermal/`LST` axis that this paper shows adds the most to *water-availability* estimation is **absent** from our Sentinel-2-only feature set (Sentinel-2 has no thermal band). This is a silence worth acknowledging: any crop-discrimination signal that lives primarily in canopy temperature / evapotranspiration contrast (e.g., irrigated vs. rain-fed separation, fine water-stress timing) is not directly available to us and would require Landsat/MODIS `LST` or ECOSTRESS fusion to recover. It bounds how far an optical-only "lite" feature set can go on water-driven class separability.

## Evaluation Caveats

- **Not a comparator on our evaluation bar.** No classifier, no `field_id`-grouped CV, no per-class F1 / Cohen's Kappa, no spatial holdout — nothing to calibrate against our field-disjoint grouped-CV standard. Cite this paper only for the biophysics of band choice, never as a classification benchmark.
- **Different crops, region, and sensor.** Validation was on soybean/corn/barley in the temperate Argentine Pampas using MODIS/Aqua and field/lab data — not Tanzanian smallholder maize/cotton/rice/sorghum/millet/sunflower/cassava at Sentinel-2 10 m. The *mechanism* (SWIR = water, NIR = dry matter, NDVI/EVI lag) transfers; the specific scatterplot parameterization and `R2` values do not.
- **Scale mismatch.** The MODIS pixel scale is far coarser than smallholder field size in northern Tanzania; the regional coherence results say nothing about pixel-level class separability at the granularity our pipeline operates on.
- **Thermal dependence is a strength there, an unavailable input here.** The headline capability (root-zone water via `LST`) relies on a band we do not have, so part of the paper's result is structurally out of reach for our Sentinel-2-only design — a caveat to state, not a benchmark to match.
