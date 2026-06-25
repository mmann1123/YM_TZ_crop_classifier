# Assessment of Land Degradation in Semiarid Tanzania—Using Multiscale Remote Sensing Datasets to Support Sustainable Development Goal 15.3

**Citation:** Reith, J., Ghazaryan, G., Muthoni, F., & Dubovyk, O. (2021). Assessment of Land Degradation in Semiarid Tanzania—Using Multiscale Remote Sensing Datasets to Support Sustainable Development Goal 15.3. *Remote Sensing*, 13(9), 1754. DOI: `10.3390/rs13091754` (verified against Crossref: title, journal, authors, and year all match).

## Paper type

This is a **land-degradation monitoring / land-change study, not a crop-type classification study.** It quantifies the UN Sustainable Development Goal (SDG) 15.3.1 indicator ("proportion of land that is degraded over total land area") for the Kiteto and Kongwa (KK) districts of central semiarid Tanzania, 2000–2019. Per rule 7, the relevance lens below focuses on Tanzanian study-area context, land-use drivers, the smallholder landscape, data scarcity, and methodological adjacency to our `xr_fresh` temporal-feature pipeline — not on a train/test generalization protocol. There is no crop classifier, no per-crop accuracy, and (author-flagged) no independent field validation of the degradation maps.

## Objectives

- Produce the first sub-national SDG 15.3.1 land-degradation assessment for Tanzania using **higher-resolution (30 m)** Landsat time series and a customized 30 m land-cover map, rather than the coarse global default datasets (250–300 m).
- Compare the adapted high-resolution method (AM) against the United Nations Convention to Combat Desertification (UNCCD) default method (DM, run in `Trends.Earth`) to test whether finer resolution changes the estimated extent and location of land degradation.
- Answer three questions: how much land is degraded and where are the hotspots; how each of the three SDG sub-indicators contributes; and whether 30 m data improves delineation versus 250 m data.

## Methods

- **Study area:** Kiteto and Kongwa districts (Dodoma and Manyara regions), central Tanzania — a hot arid steppe, ~600 mm/yr rainfall, growing season November–June, where 75% of the labor force depends on agriculture and croplands are expanding under population pressure.
- **SDG 15.3.1 framework:** three complementary sub-indicators aggregated by the "one-out, all-out" rule (if any one signals degradation, the land is deemed degraded):
  1. **Land cover (LC) change** — transitions among six IPCC classes (forestland, grassland, cropland, wetland, urban, otherland). DM uses the 300 m ESA-CCI map; AM uses 30 m RCMRD maps (2000–2018). Conversions *to* cropland were deliberately *not* counted as degradation in the AM, to avoid an ecosystem-services tradeoff.
  2. **Land productivity (LP)** — decomposed into three components: **trend** (the trajectory of NDVI/productivity over time), **state** (recent productivity vs a historical baseline), and **performance** (local productivity vs the maximum for similar land units). DM uses 250 m MODIS NDVI; AM uses 30 m NDVI from harmonized Landsat 5/7/8 surface reflectance, processed in Google Earth Engine with `fmask` cloud/shadow masking and restricted to the growing season.
  3. **Soil organic carbon (SOC)** — from the `SoilGrids250m` product (0–30 cm) for both methods, since no national SOC database exists for Tanzania.
- **Precipitation:** CHIRPS (0.05°) integrated into the productivity analysis.
- **Baseline (t0) = 2015**, computed as the 2000–2015 average, and remeasured to monitor progress toward land-degradation neutrality by 2030. A baseline period (BP, 2000–2015) and a first monitoring period (MP, 2015–2019) are reported separately.

**Evaluation protocol:** Not a classification study; no train/test generalization protocol applies, so the paper sits nowhere on the pooled-kfold → field-grouped-CV → spatial-holdout → cross-year/cross-sensor spectrum our manuscript is calibrated against. Validation of the degradation product is **indirect and weak**: the authors compare AM against the UNCCD DM and against prior national estimates, but perform **no independent ground-truth field validation** of the degradation maps (author-acknowledged limitation). The 30 m RCMRD land-cover maps and the Landsat NDVI inputs carry their own unreported classification error that propagates into the degradation indicator. No confusion matrix, no per-class accuracy, and no Kappa are reported for the underlying land-cover labels.

## Key Findings

- **Method choice dominates the headline number.** Under DM, 70% of KK was reported degraded during 2000–2015; under the higher-resolution AM, only ~16% — a roughly four-fold difference driven almost entirely by resolution and method, demonstrating how sensitive degradation accounting is to the input data.
- Over the full 2000–2019 window the AM found **27.7% of KK degraded and 2.8% improved.**
- **Croplands are central to the degradation story.** Croplands were the *most affected* land-cover class by land-productivity decline (48.4% under DM in 2000–2015; ~38–42% of the degraded area under AM), and anthropogenic covers — cropland and urban — *expanded* while natural covers (forest −3 to −9%, grassland −6.6%) declined. Forest loss roughly doubled from ~3000 ha/yr in the baseline period to ~6000 ha/yr in the monitoring period, with conversion largely to cropland. This makes cropland both the fastest-expanding and one of the most-degraded classes in the landscape we are mapping.
- The **land-productivity (LP) sub-indicator dominated** the combined degradation signal; LC and SOC contributed comparatively little.
- High-resolution data shifted not just the magnitude but the **spatial pattern** of detected hotspots, supporting the argument that coarse global products miss fragmented smallholder-scale change.

## Relevance to Our Crop-Classification Study

- **Tanzanian context and motivation.** This paper documents, with quantitative SDG accounting, exactly the land-use dynamic that motivates accurate crop mapping in central/northern Tanzania: croplands are the fastest-expanding land class, encroaching on forest and grassland under population pressure, while per-hectare productivity stagnates. It is a strong source for our introduction's framing of *why* fine-grained, crop-specific maps matter in this smallholder, data-scarce setting.
- **The "coarse global maps are inadequate" argument.** Its central empirical result — that the 300 m ESA-CCI / 250 m MODIS default products give a wildly different (70% vs 16%) and spatially mislocated picture than 30 m data — directly reinforces our manuscript's thesis that generic, coarse-resolution "agriculture"-only land-cover products are insufficient for sub-Saharan smallholder landscapes, and that finer, locally adapted mapping is needed.
- **Methodological adjacency to `xr_fresh`.** The LP sub-indicator's trend/state/performance decomposition of a Landsat NDVI time series is conceptually a cousin of our engineered time-series features: both reduce a temporal stack to interpretable per-pixel statistics (a slope/trajectory term, a state/level term) rather than feeding raw sequences to a deep network. This is a useful comparator for the "cheap, interpretable temporal summaries" half of our lite-learning argument — though their statistics are hand-specified for one index (NDVI), whereas `xr_fresh` computes a broader tsfresh-style battery (mean, max, min, slope, skewness, number of peaks, complexity) across `EVI`, `B2`, `B6`, `B11`, `B12`, and `hue`.
- **Shared tooling.** Like our pipeline, they use Google Earth Engine, surface-reflectance harmonization, `fmask`-style cloud/shadow masking, and growing-season restriction — corroborating these as standard, defensible preprocessing choices for the region.
- **Not a comparator on the evaluation axis.** Because there is no crop classifier and no field validation, this paper does not sit anywhere on the evaluation spectrum our manuscript is calibrated against; cite it for context and motivation, never as an accuracy benchmark.

## Evaluation Caveats

- **No independent validation.** The degradation product is never checked against field observations (author-flagged). Its accuracy is asserted by internal consistency and cross-method comparison only.
- **Error propagation from unvalidated land-cover inputs.** The 30 m RCMRD land-cover maps and the Landsat NDVI series have unreported classification/measurement error that flows into the SDG indicator; no confusion matrix or per-class accuracy is given for the cropland label specifically.
- **Method sensitivity, not transfer.** The 70%-vs-16% gap shows the result is highly sensitive to input resolution and methodology — a cautionary note for any study (including ours) that compares against coarse reference maps.
- **Class definitions are coarse.** "Cropland" is a single IPCC bin; the study cannot distinguish crop *types* and so cannot speak to minority-crop discrimination, the hard problem our work targets.
- **Heterogeneous input fusion.** Different sub-indicators use different sensors, resolutions, and date ranges, so the aggregated indicator blends heterogeneous data — a structural limitation inherent to the multi-source design.
