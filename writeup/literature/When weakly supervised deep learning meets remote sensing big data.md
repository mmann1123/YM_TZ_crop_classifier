# Cost-efficient information extraction from massive remote sensing data: When weakly supervised deep learning meets remote sensing big data

**Citation:** Li, Y., Li, X., Zhang, Y., Peng, D., & Bruzzone, L. (2023). "Cost-efficient information extraction from massive remote sensing data: When weakly supervised deep learning meets remote sensing big data." *International Journal of Applied Earth Observation and Geoinformation*, 120, 103345. DOI: `10.1016/j.jag.2023.103345` (verified via Crossref; title and authors match exactly). Open access, CC BY-NC-ND 4.0.

> Survey / review paper with **no original experiments or new metrics** — per skill rule 7, the protocol question is skipped and Relevance focuses on conceptual/methodological implications. Tabulated numbers (e.g. its Table 4 object-detection mAPs) are quoted from cited works, not measured here.

## Objectives

Survey **weakly supervised deep learning (WSDL)** as the route to *cost-efficient* information extraction from remote-sensing big data (RSBD). The premise: deep nets need high-quality, dense (e.g. pixel-level) labels that are infeasible to produce at the scale/velocity/variety/volume of modern RS data, so the field turns to **weaker, cheaper labels** (image-level, point-level, box-level, noisy/crowdsourced) to train deep models. The review organizes WSDL achievements across four RS tasks — scene classification, object detection, semantic segmentation, change detection — and outlines future directions.

## Methods (survey structure)

- **Framing:** RSBD's "4 V" (volume, variety, velocity, veracity) make exhaustive dense annotation "impossible," so weak supervision (per Zhou 2018's taxonomy: **incomplete**, **inexact**, **inaccurate** supervision) is the cost-efficiency strategy. Cloud computing handles the compute/storage side.
- **Coverage:** for each of the four tasks, the review catalogs WSDL strategies — error-tolerant (noise-robust loss, noise correction), semi-supervised (GAN-based, self-labeling, self-supervised), domain adaptation, and low-shot (zero-/few-shot) learning — and lists available weakly labeled datasets (Tables 1, 3, 5).
- **Crowdsourcing / cheap-label angle (relevant to us):** explicitly discusses generating labels from existing land-cover products and **crowdsourcing (e.g. OpenStreetMap)**, noting these "introduce label noises into datasets," which motivates the error-tolerant branch. Mentions **TimeSen2Crop** (>1M pixel-based Sentinel-2 time-series samples over 16 crop types) among weak-segmentation datasets — a crop-time-series resource.
- **No evaluation protocol of its own:** as a survey it reports others' results; it does not run experiments, define a holdout, or measure generalization. Its tabulated mAP/accuracy numbers are inherited from primary studies under those studies' own (largely benchmark, in-scene) protocols.

## Key Findings

- **WSDL's entire "cost-efficiency" is defined as cheaper labeling for GPU-bound deep nets**, not cheaper compute. The recurring argument is that image-level labels "save at least one hundred times more time than pixel-level labels," so weak labels reduce *annotation* cost while the model remains a heavy, GPU-dependent deep network. Compute cost is treated as solved by cloud distribution, not reduced. This is the explicit **foil** for our paper's different reading of "cost-efficiency."
- A persistent "performance gap between fully and weak supervision" remains, but WSDL methods can be "competitive with fully supervised methods" on benchmark tasks (e.g. their Table 4 weakly-supervised object-detection mAPs vs Fast-RCNN).
- Crowdsourced / product-derived labels are recognized as a legitimate weak-supervision source but are framed primarily as **noisy data to be denoised** by error-tolerant learning, not as a first-class validated ground-truth pipeline.
- Domain-adaptation and low-shot WSDL are surveyed as the answers to cross-sensor/cross-region label scarcity — but as *deep-network* techniques (GAN-based adaptation, meta-learning), keeping the heavy-model assumption.
- **Near-silent on spatial / cross-year holdout discipline.** The survey organizes everything by supervision type and task; it does **not** foreground spatial-leakage, field-disjoint splitting, or cross-year robustness as evaluation concerns. Generalization is discussed as a *domain-adaptation modeling* problem (train a better transferable net), not as an *evaluation-rigor* problem (test on a spatially disjoint holdout). That gap is itself a finding for us.

## Relevance to Our Crop-Classification Study

- **The defining contrast for our cost-efficiency claim.** This survey crystallizes the prevailing definition — **cost-efficiency = cheaper labels to feed GPU-bound deep nets** — which is exactly the foil our paper rewrites as **cost-efficiency = cheaper compute** (GPU-free engineered features + tree models). Cite it to sharpen our positioning: the WSDL community lowers labeling cost but keeps the expensive deep model; we lower the model/compute cost (and pair it with crowdsourced YouthMappers labels). The two are complementary axes of "cheap."
- **Shared premise, opposite solution.** We agree with its premise (dense labels are infeasible at scale in data-scarce settings) but diverge on the remedy: WSDL = weak labels + heavy net + cloud GPUs; ours = field-level crowdsourced labels + light interpretable features + CPU. Useful for a related-work paragraph framing the design space.
- **Crowdsourcing validation.** Its treatment of OSM/crowdsourced labels as a recognized (if noisy) supervision source legitimizes our YouthMappers crowdsourcing — though it frames such labels as noise to be tolerated, whereas we treat field-disjoint grouped CV and `Field_size` weighting as the discipline that makes crowdsourced labels trustworthy.
- **Evaluation-rigor gap we can occupy.** Because the survey is near-silent on spatial/cross-year holdout discipline, it underscores that evaluation rigor (our `StratifiedGroupKFold` on `field_id`, leakage control) is a comparatively under-emphasized dimension in the WSDL big-data literature — a place our paper contributes beyond just being "lighter."
- **TimeSen2Crop pointer:** a >1M-sample Sentinel-2 crop-time-series weak-label dataset is noted — a possible external reference for crop-time-series feature work, though European, not sub-Saharan.

## Evaluation Caveats

- **Survey, no original metrics.** Reports no new experiments; any numbers (e.g. Table 4 mAPs, Table 2 zero-shot scores) are quoted from primary studies under their own protocols and must be cited to those sources, not to this review. Do not attribute measured performance to "Li et al. 2023."
- **Inherits benchmark / in-scene protocols of its sources.** The surveyed results sit mostly at the pooled/benchmark end of our spectrum; the review does not re-stratify by spatial-disjointness or field-grouping, so it offers no leakage-controlled accuracy and no transfer comparator that clears our bar.
- **Cost-efficiency is labeling-cost only.** Its scope explicitly excludes the *compute*-cost dimension (assumed handled by cloud/GPU), so it cannot speak to the GPU-free / interpretability advantages central to our paper — that omission is precisely the contrast we exploit.
- **Crowdsourced labels framed as noise, not validated ground truth.** No discipline is offered for turning crowdsourced labels into a trustworthy evaluation set beyond noise-robust training — i.e. it does not address the spatial-leakage / field-grouping issues that govern whether crowdsourced crop labels yield honest accuracy.
- **Minority-class / per-class crop performance not addressed** (it is task-organized, not crop-organized), so it says nothing about cassava/millet/sunflower/sorghum-type minority-crop recall.
