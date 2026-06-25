# CropHarvest: A global dataset for crop-type classification (benchmark dataset)

**Citation:** Tseng, G., Zvonkov, I., Nakalembe, C. L., & Kerner, H. (2021). "CropHarvest: A global dataset for crop-type classification." *Thirty-fifth Conference on Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track (Round 2).* OpenReview: `https://openreview.net/forum?id=JtjzUXPEaCu`. DOI: **not found in PDF**. NeurIPS Datasets & Benchmarks Track papers are published via OpenReview and carry no Crossref DOI; the `refs.bib` lead is the key `tseng2021cropharvest` with the OpenReview URL, not a DOI. Per skill rule 10, this is the not-found path: title/authors/venue confirmed from the citation block, DOI tagged `not found`.

## Status / source caveat (read first)

> **The supplied PDF is NOT the paper.** It is a 5-page print-to-PDF of the project's **GitHub README** (`github.com/nasaharvest/cropharvest`, captured 26 Jun 2024), showing the repo file tree, install instructions, the pipeline diagram, and the BibTeX citation. The README links to "this paper" but does not contain it. **Therefore no benchmark accuracy tables, no per-class metrics, no train/test protocol, and no model-comparison numbers are present in this document and none should be attributed to "CropHarvest (Tseng et al. 2021)" from this source.** The dataset-composition figures below come from the README's own summary text; everything about the actual benchmark task design and baseline results would need to be read from the NeurIPS paper itself before being cited. Statements in this briefer about the benchmark/baseline are flagged as README-level / general knowledge, not extracted-from-paper.

## Objectives

CropHarvest is an open-source, global remote-sensing dataset and benchmark for agricultural land-use / crop-type classification, built to give the data-scarce crop-typing problem a common, reproducible testbed. Its purpose is to aggregate many heterogeneous agricultural label sets and pair them with harmonized satellite + climatology time series so that models (including simple baselines) can be trained and compared across geographies.

## Methods (dataset construction, from the README)

- **Scale and labels:** 95,186 datapoints; 33,205 (35%) carry multiclass crop-type labels, the remainder binary crop/non-crop. 70,213 (74%) are paired with remote-sensing + climatology data.
- **Sensors / features:** Sentinel-2 (optical), Sentinel-1 (SAR), the SRTM DEM, and ERA5 climatology — i.e. optical + SAR + topography + weather, as per-pixel time series. This is a richer multi-sensor stack than our optical-only `EVI`/`B2`/`B6`/`B11`/`B12`/`hue` design and is relevant to any SAR-vs-optical discussion.
- **Aggregation:** 21 constituent agricultural datasets merged into a single GeoJSON, satellite data exported from Earth Engine, then combined into `(X, y)` tuples exposed via a Python `Dataset` API (Zenodo-hosted).
- **Tanzania content (relevant to us):** the repo's commit history explicitly references a "tanzania ecaas rice dataset" — CropHarvest includes a **Tanzania rice** label set, making it a Global-South, in-country-adjacent reference point for the rice class.
- **Baseline (README-level only):** the README's getting-started text says the demo notebook trains a **random forest** against the data, and points to a `benchmarks/` folder for "more examples of models." Any specific baseline scores live in the paper/benchmarks, not in this PDF.
- **Evaluation protocol (load-bearing):** **Not determinable from the supplied document.** The README does not state the benchmark's train/test split design, whether splits are spatially disjoint, field-grouped, or pooled-random, nor what metrics are reported. The CropHarvest benchmark is known (from the literature generally) to provide held-out per-region test tasks, but that cannot be verified here. Treat the protocol as **unstated in this source** and do not assert where it sits on our pooled-kfold to cross-region spectrum without reading the actual paper.

## Key Findings (what this source actually supports)

- A single, openly licensed (CC-BY-SA-4.0), Earth-Engine-reproducible benchmark consolidating 21 crop/land-use datasets with harmonized Sentinel-1 + Sentinel-2 + SRTM + ERA5 time series exists and is pip-installable (`pip install cropharvest`).
- The dataset is global in coverage (the README's world map shows dense coverage across all inhabited continents, including sub-Saharan Africa) and is heavily binary crop/non-crop with a 35% multiclass subset — i.e. it is itself class-imbalanced toward the binary task.
- It includes a **Tanzania rice** set, directly on-theme for our region and one of our priority crops.
- A random-forest baseline is the advertised entry point — consistent with the broader finding that classical ML is the natural first baseline in data-scarce crop typing.

## Relevance to Our Crop-Classification Study

- **Canonical data-scarce crop-typing benchmark.** CropHarvest is the standard reference for the exact regime our paper occupies (few labels, smallholder, multi-region) and is worth citing as the community benchmark embodying the data-scarcity premise. Its RF baseline supports our "classical ML as the sensible baseline under scarcity" framing.
- **Tanzania rice overlap.** The included Tanzania rice set makes it the most directly geographically relevant of the data-scarce benchmarks for one of our minority/priority crops; a candidate external comparison point if we ever want an out-of-our-corpus rice reference.
- **Multi-sensor design contrast.** Its Sentinel-1 + Sentinel-2 + DEM + ERA5 stack is a foil for our optical-only, engineered-feature approach — useful when arguing about whether SAR/climatology add value vs the cost of ingesting them.
- **Caution on attribution.** Because we only hold the README, **we must source any CropHarvest benchmark/baseline number from the NeurIPS paper directly**, not from this file, to avoid mis-citing figures. This is the single most important practical takeaway.

## Evaluation Caveats

- **Primary caveat: wrong document.** This PDF is the GitHub README, not the paper; it contains no experimental results. Any benchmark accuracy, per-class F1, or protocol detail must be obtained from the OpenReview paper before citation. Do not paste numbers "from CropHarvest" using this source.
- **Protocol unknown from this source.** Whether the benchmark uses spatially disjoint / cross-region holdouts (which would put it *above* our field-grouped-within-one-region bar) or simpler splits cannot be confirmed here — flag as to-verify. (The benchmark's design is plausibly a stronger spatial-transfer setup than ours, which is exactly why the protocol must be checked before any comparison.)
- **Class imbalance is structural.** 65% of datapoints are binary-only crop/non-crop; the multiclass crop-type subset is a minority of the data, so headline "CropHarvest" performance can conflate the easy binary task with the harder multiclass one.
- **DOI absent by venue convention**, not by error — `not found in PDF`; cite via the OpenReview URL and the `tseng2021cropharvest` key. Never fabricate a DOI for it.
