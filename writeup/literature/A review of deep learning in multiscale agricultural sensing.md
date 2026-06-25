# A Review of Deep Learning in Multiscale Agricultural Sensing

**Citation:** Wang, D., Cao, W., Zhang, F., Li, Z., Xu, S., & Wu, X. (2022). A Review of Deep Learning in Multiscale Agricultural Sensing. *Remote Sensing*, 14(3), 559. DOI: `10.3390/rs14030559` (verified via Crossref — title, authors, year, and journal all match).

> Review/theory paper. Per skill rule 7, the evaluation-protocol question is skipped; Relevance focuses on conceptual implications for the deep-learning-vs-engineered-feature trade-off, data hunger, and transfer fragility that motivate our "lite learning" thesis.

## Objectives

Survey deep learning (DL) in multiscale agricultural remote and proximal sensing across four observation scales — land, field, canopy, and leaf — with emphasis on three paradigms: convolutional-neural-network supervised learning (CNN-SL), transfer learning (TL), and few-shot learning (FSL). The review aims to map applications (land mapping, crop classification, biotic/abiotic stress monitoring, yield prediction) and, critically for us, to document the *prerequisites and bottlenecks* of DL in agriculture: data volume, dataset quality, compute, and generalization.

## Methods

A structured literature survey (not an experiment). It organizes prior DL agricultural work by sensing platform (satellite, UAV, terrestrial robot), sensor modality (RGB, multispectral, hyperspectral, thermal, SAR), and observation scale, then reviews CNN-SL, TL, and FSL separately. It catalogs public agricultural image datasets and prior reviews (e.g., Kamilaris & Prenafeta-Boldú 2018; Lu & Young's 34-dataset survey) and synthesizes recurring limitations.

## Key Findings

- **DL is data-hungry.** The review's central, repeatedly stated conclusion: high-performing CNN-SL depends on large, high-quality, well-annotated datasets, and "the quantity, diversity, complexity, and quality of currently available datasets are not sufficient to support the widespread promotion of deep learning" in precision agriculture. This is the single most important point for our argument.
- **Transfer learning and few-shot learning exist precisely because labels are scarce.** TL (reusing ImageNet/natural-image weights) and FSL are surveyed as the field's coping mechanisms for limited agricultural labels — but the review notes these are workarounds, not cures, and carry domain-gap risk when natural-RGB pretraining meets multispectral/temporal farmland data.
- **Multiscale, multimodal sensing is proliferating** (satellites, UAVs, field robots), generating large image volumes whose bottleneck is rapid, accurate information extraction rather than acquisition.
- **Supervised DL dominates** the surveyed work and achieves high accuracy, but typically under controlled or data-rich setups; the review flags annotation cost and dataset insufficiency as the limiting factors for deployment in real, heterogeneous farmland.

## Relevance to Our Crop-Classification Study

This is arguably the **best single motivation anchor for "lite learning"** in the corpus. It is an authoritative, well-cited review that documents — from inside the DL community — exactly the three weaknesses our manuscript turns into its thesis: (1) **data hunger** (DL needs large labeled datasets that do not exist for smallholder Tanzanian crops), (2) **generalization fragility** (TL/FSL are needed because models do not transfer cleanly, and natural-image pretraining mismatches remote-sensing data), and (3) **compute cost** (high-performance sensing pipelines presume substantial infrastructure). Our engineered-feature + tree-based approach is positioned as the complementary answer: cheap, CPU-only, interpretable, and viable on the ~1,400 crowdsourced observations we have. Use this paper to ground the introduction's claim that DL's requirements are ill-matched to data-scarce sub-Saharan settings. It pairs naturally with the stronger empirical comparators in the corpus (Pankajakshan et al. 2025 on deep architectures failing to generalize cross-scene/cross-sensor) — this review supplies the conceptual case, that paper supplies the experiment.

## Evaluation Caveats

- **Conceptual, not comparative.** A review reports no protocol of its own and runs no head-to-head DL-vs-classical benchmark on a shared dataset, so it cannot quantify the trade-off — it only documents the qualitative consensus. Cite it for the *existence and shape* of DL's limitations, not for any accuracy number.
- **Precision-agriculture and proximal-sensing breadth.** Much of the review covers leaf/canopy-scale phenotyping, UAV imagery, and stress detection rather than satellite land-scale crop-type mapping, so only the land/field-scale and the dataset-scarcity sections map cleanly onto our smallholder Sentinel-2 problem.
- **Pro-DL framing.** The review is written to promote DL's evolution in agriculture; its data-scarcity caveats are honest but secondary to its optimism, so it should be read as a sympathetic-insider admission of DL's costs rather than an argument against DL.
- **2022 vintage.** Predates the most recent lightweight-transformer and foundation-model work, so its survey of FSL/TL maturity is now somewhat dated.
