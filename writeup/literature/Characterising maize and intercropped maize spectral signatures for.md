# Characterising maize and intercropped maize spectral signatures for cropping pattern classification

**Citation:** Mahlayeye, M., Darvishzadeh, R., & Nelson, A. (2024). Characterising maize and intercropped maize spectral signatures for cropping pattern classification. *International Journal of Applied Earth Observation and Geoinformation*, 128, 103699. DOI: `10.1016/j.jag.2024.103699` (verified via Crossref — title, authors, year, journal, volume, and article number all match). Open access (CC BY).

## Objectives

Characterize how the Sentinel-2 spectral signatures of **monocropped maize vs intercropped maize ("imaize")** differ across the crop-growing season, and identify the **optimal crop-growth phases, spectral regions, and vegetation indices** that discriminate the two cropping patterns at field level. The motivating gap: intercropping is the backbone of African smallholder agriculture (41–86% of maize/rice/sorghum/millet in Africa is intercropped) yet is rarely mapped, and intercropped fields are routinely misclassified because they look spectrally similar to monocrops at most growth stages.

## Methods

- **Study area.** Busia County, **western Kenya** (bimodal rainfall, smallholder rainfed), within three agro-ecological zones. Maize grown as monocrop (commercial) or intercropped with beans/soybean/cowpea/cassava (subsistence).
- **Imagery.** Five Sentinel-2 L2A surface-reflectance scenes, March–August 2019, < 15% cloud (no cloud masking needed), dates aligned to a crop calendar spanning emergence-seedling, vegetative, flowering-yield formation, ripening, and post-harvest. Bands 2/3/4/5/6/7/8/8A/11/12; SWIR/red-edge resampled 20 m → 10 m. Eleven VIs (NDVI, NDWI, NDRE2, NDRE, GNDVI, SIPI, EVI, REcl, MTCI, REI, NDRE3) emphasizing red-edge and NIR.
- **Ground truth.** Field boundaries from PlantVillage (Penn State), updated by 240 farmer interviews and field visits; Google Earth verified. **87 fields** retained (50 maize, 37 imaize) — field-averaged reflectance per field, not per pixel.
- **Statistical separability.** Mann-Whitney U test (data non-normal) on field-averaged reflectance per band per phase to find which bands/phases differ significantly.
- **Classification.** Random Forest (scikit-learn), five feature scenarios (all bands; significant bands; VIs; all bands + VIs; significant bands + VIs). `mtry` = √features; `ntree` and `max_depth` searched (optimal `ntree`=50, `max_depth`=4).
- **Evaluation protocol (load-bearing).** A **single random 65% train / 35% validation split of the 87 fields** — one split, not cross-validated, reported per phase/scenario. Granularity is **field-level** (each field is one field-averaged sample), so this is *not* pixel-level spatial leakage in the within-field sense — but with n=87 and a single hold-out, the accuracy estimates are high-variance and not field-grouped cross-validation. It sits below our manuscript's bar: same region, single year, single split, no spatially disjoint holdout, no cross-year/cross-sensor transfer. The "100% accuracy" figures are one split on tens of fields and must be cited as *protocol*, not as a robust number.

## Key Findings

- **Separability is phase-gated.** Maize and imaize are spectrally distinguishable **only in narrow phenological windows** — the **vegetative phase** (significant differences across *all* bands) and the **flowering-yield phase** (significant in Blue, Green, Red, RE704, RE783, NIR833, NIR865; *not* in RE740, SWIR1614, SWIR2202). At emergence-seedling, ripening, and post-harvest there are **no significant spectral differences** (soil-dominated or post-harvest mixing), so single-date mapping at the wrong time fails.
- **Vegetative-phase classification:** user, producer, and overall accuracy = **100%** with F1 and Kappa = 1.0 across all feature scenarios — but this is a single 65/35 split on n=87 (50/37); cite the protocol and the narrow window, not the number.
- **Flowering-yield phase is harder:** best overall accuracy 86% (all bands), 79% (significant bands), 66% (VIs); maize producer accuracy 80%, imaize 100% — accuracy drops once phenology blurs.
- **Original Sentinel-2 bands beat VIs**, and "all bands" beat "significant bands only" — non-significant bands (RE740, SWIR1614/2202) still carry subtle discriminative information. Red-edge and NIR are the most valuable regions.
- **Class imbalance bites:** imaize (n=37) classified worse than maize (n=50); RF is noted as sensitive to imbalance even on small samples. Weeds in monocrop maize act as a confounding secondary canopy, mimicking intercrop signal.

## Relevance to Our Crop-Classification Study

Two strong, directly transferable lessons. **(1) Phenological-window dependence.** Our manuscript's whole premise is engineering time-series features (`xr_fresh`) to capture phenology across the season — this paper is field-level evidence that crop-distinguishing signal lives in *specific* windows (vegetative, flowering-yield) and vanishes outside them. That validates extracting season-spanning statistics (slope, day-of-year-of-max, number-of-peaks) rather than relying on any single composite, and it is a caution about our **single April–May 2023 collection window**, which captured crops "primarily in late growing season" — potentially the *low-separability* end for some crops. **(2) Intercropping as a confusion source.** Our manuscript explicitly attributes cassava/maize confusion to intercropping at 10 m; this paper shows quantitatively that intercropped maize is spectrally near-indistinguishable from monocrop maize at most phases, corroborating that mechanism and suggesting that our cassava/maize errors are partly irreducible without phase-targeted features or higher resolution. Band-wise, its red-edge/NIR emphasis aligns with our use of `B6` (red edge).

## Evaluation Caveats

- **n = 87 fields, single 65/35 split.** The 100% vegetative-phase accuracy is a single hold-out on a tiny, imbalanced sample (50 maize / 37 imaize) — high variance, not cross-validated, and almost certainly optimistic. **Cite the protocol and the phenological-window finding, not the accuracy value** (skill rule 4).
- **Binary task (maize vs imaize), single region (Busia, Kenya), single year (2019).** Not multi-crop, not multi-region; no spatial or temporal transfer test. Conclusions are local.
- **Field-averaged reflectance, not pixel-level.** Avoids within-field pixel leakage by design, but also discards within-field heterogeneity that our pixel-buffer approach deliberately models; the two granularities are not directly comparable.
- **Weed confound acknowledged.** Maize-as-imaize misclassification is partly driven by weeds acting as a secondary canopy, which the spectral analysis cannot separate from genuine intercrop signal.
- **No SAR, no balanced large-sample validation.** Optical-only; class-imbalance effect on imaize is noted but not corrected (no resampling/weighting), so the imaize numbers are the floor.
