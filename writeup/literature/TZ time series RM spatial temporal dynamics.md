# Using Time-Series Remote Sensing Images in Monitoring the Spatial–Temporal Dynamics of LULC in the Msimbazi Basin, Tanzania

**Citation:** Machiwa, H., Mango, J., Sengupta, D., & Zhou, Y. (2021). Using Time-Series Remote Sensing Images in Monitoring the Spatial–Temporal Dynamics of LULC in the Msimbazi Basin, Tanzania. *Land*, 10(11), 1139. DOI: `10.3390/land10111139` (verified against Crossref: title, journal, authors, and year all match).

## Paper type

This **is a land-cover classification / change-detection study**, so the full evaluation-protocol lens applies. It classifies land use/land cover (LULC) in the Msimbazi Basin (Dar es Salaam, Tanzania) at four dates (1990, 2000, 2010, 2019) and analyzes change in relation to population growth. It is *not* crop-type classification — "agriculture" is a single coarse class — but it produces supervised classifications with confusion matrices and Kappa, and its design is a sharp methodological **contrast** to our genuine within-season temporal features. Despite "Time-Series" in the title, it is **four discrete single-date snapshots**, not a temporal-feature model.

## Objectives

- Quantify the spatial–temporal dynamics of LULC along the Msimbazi valley over 1990–2019 (a 29-year span), using satellite imagery at four decadal-ish epochs.
- Relate observed LULC change to population growth/urbanization pressure, using population census and projection data.
- Diagnose drivers of wetland and basin degradation to inform management and policy.

## Methods

- **Study area:** Msimbazi Basin, Dar es Salaam, Tanzania — a riverine/wetland-bearing urban basin under heavy population and urbanization pressure.
- **Imagery:** **Landsat 5 TM** for 1990, 2000, 2010 and **Landsat 8 OLI** for 2019 (from USGS GLOVIS). Atmospheric correction via `ENVI 5.1 FLAASH`; reprojected to UTM WGS-1984 zone 37S. Sentinel-1 SAR was considered for cloud-free coverage but available only for 2019, so it was not used in the time series. Each epoch is **one single-date image** — there is no within-epoch temporal stack and no phenological feature extraction.
- **Classifier:** **Maximum-likelihood supervised classification** in **ArcGIS 10.5**, using user-digitized training-site spectral signatures, into **seven classes**: agriculture, built-up land, forest, bushland, grassland, water, and wetland vegetation. Annual change rates were computed per decadal interval.

**Evaluation protocol:** This is the load-bearing flaw. Accuracy was assessed by sampling **1000 random points** on each classified map and comparing the classified pixel value against a reference value taken from a **single 2016 Sentinel-2 image (10 m)** — and **the same 2016 reference image was used to validate all four epochs (1990, 2000, 2010, 2019)**. This is a **non-contemporaneous reference**: a 2016 image cannot be ground truth for 1990 or 2019 land cover, since classes (especially the dynamic agriculture and water classes) changed in the intervening 20+ years. The authors themselves observe that agriculture and water show "significant mismatches of their reflectance" against the static reference, particularly for 2010 and 2019. The reported Kappa coefficients are **0.79 / 0.91 / 0.90 / 0.85** for 1990 / 2000 / 2010 / 2019. On our spectrum, this is **worse than even a pooled in-scene k-fold**: there is no field-disjoint or spatially separated split, the training and accuracy points are not stated to be independent, and the "ground truth" is a different-sensor, different-year image. It measures neither spatial transfer nor cross-year robustness. The Kappa figures are **not comparable** to our field-grouped-CV Cohen's Kappa / per-class F1, and should be cited as protocol, not headline accuracy.

## Key Findings

- **Built-up land dominated and grew steadily:** 39.3% of the basin in 1990 → 42.6% (2000) → 54.1% (2010) → 65.5% (2019), driven by population growth and urbanization concentrated along the riverine corridor.
- **Forest and agriculture both declined** throughout the period (they had been the second- and third-largest classes in 1990), squeezed by built-up expansion.
- Wetland vegetation was threatened during 1990–2000 but recovered somewhat after government interventions; bushland and grassland were minor classes with inconsistent trends.
- Reported classification Kappa of **0.79–0.91** across the four epochs — but see the protocol caveat: these are anchored to a single 2016 Sentinel-2 reference for all years, and the **agriculture and water classes specifically showed lower per-class accuracy**, consistent with the non-contemporaneous-reference problem.

## Relevance to Our Crop-Classification Study

- **A direct methodological contrast for the "time-series" claim.** This paper carries "Time-Series" in its title yet is four independent single-date classifications with no temporal features. It is an ideal foil for our manuscript's argument that *genuine within-season temporal information* matters: where they classify each year from one scene's static spectral signature, we extract phenological statistics (mean, max, min, slope, skewness, number of peaks, complexity via `xr_fresh`) across the growing season from `EVI`, `B2`, `B6`, `B11`, `B12`, and `hue`. We can cite it to show that "time series" in the regional literature often means "snapshots across decades," not the dense intra-season sequences crop typing requires.
- **Illustrates the spectral confusability our temporal features resolve.** Their explicit finding that agriculture (and water) are the hardest, lowest-accuracy classes under single-date maximum-likelihood classification is precisely the failure mode multi-temporal crop phenology is meant to fix — supporting our rationale for engineered temporal features over single-date spectral classification.
- **Tanzanian context.** Reinforces the same land-change narrative as the other TZ context papers — agriculture being squeezed by built-up expansion in coastal Tanzania — useful for motivation, though Msimbazi is an urban basin rather than our northern smallholder cropping zone.
- **A cautionary evaluation example.** Its single-2016-reference-for-all-epochs design is a vivid case of validation that does not measure what it claims (temporal land-cover accuracy), reinforcing our manuscript's broader argument that evaluation rigor — contemporaneous, independent, spatially/temporally appropriate references and field-disjoint splits — is frequently lacking in the regional literature.

## Evaluation Caveats

- **Non-contemporaneous reference (cardinal flaw).** A single 2016 Sentinel-2 image is used as ground truth for 1990–2019; it cannot validate land cover decades removed, and the authors concede agriculture/water mismatches result. The Kappa values are therefore unreliable as measures of per-epoch accuracy.
- **Spatial leakage unaddressed.** No field-disjoint or spatially separated train/test split; training-site signatures and the 1000 accuracy points are not shown to be independent or spatially separated, so autocorrelation can inflate Kappa.
- **Class imbalance / minority-class blind spots.** Overall accuracy and pooled Kappa are reported; the dynamic minority/edge classes (agriculture, water, wetland) are exactly where accuracy drops, but there is no balanced metric (macro-F1) foregrounded — minority performance is partly masked.
- **Single-date, optical-only, no temporal features.** No within-season stack, no SAR (Sentinel-1 was available only for 2019 and dropped), no engineered time-series statistics; classification rests on one scene's spectral signature per epoch.
- **Coarse class scheme.** "Agriculture" is a single bin; the study cannot distinguish crop types and so says nothing about minority-crop discrimination, the core challenge of our work.
- **No spatial or cross-sensor transfer measured.** Each epoch is classified and assessed within the same basin; there is no holdout region, no cross-year classifier transfer, and no computational-cost reporting.
