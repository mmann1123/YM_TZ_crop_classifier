# Enhanced crop classification through integrated optical and SAR data: a deep learning approach for multi-source image fusion

**Citation:** Niantang Liu, Qunshan Zhao, Richard Williams & Brian Barrett (2024). "Enhanced crop classification through integrated optical and SAR data: a deep learning approach for multi-source image fusion." *International Journal of Remote Sensing*, 45(19–20), 7605–7633. DOI: `10.1080/01431161.2023.2232552` (verified against Crossref — title, authors, journal, and volume all match; received 2023, published in the 2024 volume).

## Objectives

Develop a deep-learning architecture for annual county-level crop mapping in Bei'an County, Heilongjiang, Northeast China, that:

1. Fuses multi-temporal Sentinel-1 SAR polarimetric features with a small number of Sentinel-2 optical surface-reflectance acquisitions to overcome cloud-driven gaps in the optical record.
2. Combines a 3D-CNN (spatio-temporal feature extraction), a stackable convolutional recurrent cell (`ConvSTAR`), and a shallow 2D-CNN into a hybrid `3D-ConvSTAR` network that captures spatial, spectral, and temporal dependencies simultaneously.
3. Addresses imbalanced class distribution (maize and soybean dominate; wheat is a 2.9%-of-area minority crop) by comparing the joint deep-learning structure against oversampling, inverse-frequency (balanced-loss) weighting, and mix-up data augmentation.
4. Produces an interpretable annual crop map and assesses model reliability through soft-output visualization and gradient saliency maps.

Target crops: maize, soybean, wheat, and an "other crops" residual class. This is a same-region, single-year (2017) deep-learning comparator — the flagship heavy-DL contrast to our "lite learning" thesis.

## Methods

- **Sensors / features.** 23 Sentinel-1B SLC (IW) acquisitions over the May–September 2017 growing season; 3 Sentinel-2A/B Level-1C scenes (<8% cloud) atmospherically corrected to L2A with Sen2Cor. From Sentinel-2 only `B4` (Red, 10 m), `B8A` (red edge, 20 m), and `B11` (SWIR, 20 m) were used — chosen for soybean/maize separability — resampled to 10 m. From Sentinel-1, instead of raw backscatter they computed **m-chi compact-polarimetric decomposition** (Raney et al. 2012) yielding single-bounce (`Ps`), double-bounce (`Pd`), and volume (`Pv`) scattering parameters via SNAP, plus the VH−VV cross-ratio. All features min/max normalized.
- **Architecture.** `3D-ConvSTAR`: three 3D-CNN layers (3×3×3 kernels, 32/32/64 filters, zero-padded), then three bidirectional `ConvSTAR` recurrent layers (3×3 kernels, 64 units), a single 2D-CNN layer (64 3×3 kernels) for spatial refinement and dimensionality reduction, and three fully connected layers (256, 128 units, dropout 0.4). ReLU throughout; no pooling. Input patch shape **11×11×26×3** (width × height × time steps × channel). Adam, lr 0.001, weight decay 1e-4, batch 128, early stopping.
- **Compute.** Two NVIDIA Quadro P4000 GPUs (8 GB each) plus two Intel Xeon Silver 4114 CPUs — a multi-GPU rig. TensorFlow 2.5 / Keras.
- **Ground truth.** A 2017 agricultural household survey by the Chinese Academy of Agricultural Sciences digitized **21,257 fields** from 5 m RapidEye and Sentinel-2 imagery (mean parcel 1.39 ha). Only 10% of total ground samples were used for the train/val/test split (60/20/20 stratified).
- **Comparators.** `TCNN` (1D-CNN; Pelletier et al. 2019), standalone `3D-CNN` (Ji et al. 2018), hybrid `3D-2D CNN` (Roy et al. 2019), and 3-layer `ConvSTAR` (Turkoglu et al. 2021), each reproduced and run under four feature scenarios (backscatter; m-chi; optical+backscatter; optical+m-chi).
- **Interpretation.** Vanilla-backpropagation gradient saliency maps and softmax soft-output visualization; the authors caution that on 11×11 patches spatial saliency is "more of a performance check" than rigorous attribution.
- **Metrics.** Overall accuracy (OA), Cohen's Kappa, per-class and mean F1.

**Evaluation protocol (load-bearing).** Field-disjoint but **in-region** partitioning: every crop polygon is treated as an indivisible entity and parcels are further grouped on a 10 km grid so that all pixels of a parcel (and of a 10 km grid cell) land in only one of train/val/test. This is explicitly designed to prevent same-field pixel leakage — the same concern our `StratifiedGroupKFold` on `field_id` addresses — and the authors articulate the leakage problem clearly. **However, all data come from a single county and a single year (Bei'an, 2017); there is no spatially disjoint holdout (different tile/region/agro-ecological zone) and no cross-year or cross-sensor transfer test.** The "generalizability" claim rests on applying the trained model to the *remaining 90%* of the same-county, same-year fields to produce the annual map. This places the study on **exactly the same rung as ours** — field-grouped CV within one region — not above it. The three-site comparison (Sites A/B/C) is within Bei'an, so it tests spatial robustness only within the same county.

## Key Findings

- **Best configuration:** `3D-ConvSTAR` with **optical + m-chi** features achieved OA 91.7%, **Kappa 85.7%**, and per-class F1 of 93.7% (maize), 92.2% (soybean), **90.9% (wheat)**, 74.0% (other crops) — **mean F1 87.7%**, the best of all model × feature combinations.
- **Features beat depth.** The hand-engineered **m-chi polarimetric decomposition drives the largest single gains**: for the 1D `TCNN`, switching backscatter → m-chi lifted OA by >20 points and Kappa by >30 points. Adding even three optical scenes to the SAR sequence improved every model. The choice of input representation mattered more than the architecture differences among the deep models — directly reinforcing our "features > depth" argument.
- **Minority crop is the weak point.** Despite balancing efforts, the residual **"other crops" F1 collapses to 0.74** even in the best model (and far lower — single digits to ~50% — for weaker models/features), and under backscatter-only several models post single-digit wheat F1. This mirrors the minority-crop difficulty (cassava, millet, sunflower, sorghum, cotton) central to our study.
- **Augmentation is not a free lunch.** Oversampling generally helped weaker models (TCNN mean F1 64.8 → 83.5) but balanced loss and mix-up *reduced* performance for `3D-2D CNN` and the proposed `3D-ConvSTAR`; gains for majority crops came at the cost of minority recall.
- **Reported per-class F1 and Kappa**, not just OA — appropriately, since OA is dominated by the maize+soybean ~91% of sown area.
- **Cost.** Requires dual GPUs, 26-step patch sequences, 21,257 labelled fields, and reproduction of four competing deep nets — a heavy pipeline relative to our GPU-free feature-extraction + tree-model approach.

## Relevance to Our Crop-Classification Study

This is the **flagship deep-learning comparator** for the manuscript and an unusually fair one to cite.

- **Headline parity, very different cost.** Their best Kappa (0.857) and mean F1 (87.7%) are close to our Cohen's Kappa 0.82 / F1-micro 0.85 — but achieved with dual GPUs, 26-step spatio-temporal patches, 21,257 labelled fields, and a custom hybrid 3D-CNN/ConvRNN. Our "lite" pipeline reaches comparable agreement with engineered `xr_fresh` time-series statistics and tree models on crowdsourced labels and no GPU. This is the central "matches heavier DL at a fraction of the cost" data point.
- **Features > depth, from their own ablation.** Their largest accuracy jumps come from the **hand-engineered m-chi polarimetric features**, not from architectural depth — the most direct external support for our thesis that interpretable, engineered features carry most of the signal. Cite as: even in a deep-learning paper, the dominant lever was feature engineering.
- **Same evaluation rung.** Their field-and-grid-grouped partitioning is methodologically sound and matches our anti-leakage stance, but it stays within one county and one year. Neither they nor we demonstrate true spatial/cross-year/cross-sensor transfer, so the comparison is genuinely apples-to-apples on the evaluation bar — worth stating explicitly so we neither over- nor under-claim.
- **Band/index choice convergence.** They selected red edge (`B8A`), SWIR (`B11`), and red — overlapping our `B6` (red edge) and `B11`/`B12` (SWIR) choices — reinforcing that red-edge + SWIR carry crop-type signal. Their case for SAR fusion (cloud-gap filling) is the strongest argument against our optical-only design; worth acknowledging as a limitation we accept in exchange for simplicity and GPU-free inference.
- **Minority-crop honesty.** Their "other crops" F1 of 0.74 shows that even a heavy, augmentation-tuned DL model leaves a hard residual minority class — context for our per-class minority F1 results, and a reason to report per-class metrics rather than only OA/micro-F1.
- **Interpretability contrast.** Their interpretability is post-hoc gradient saliency on tiny 11×11 patches, which they themselves downgrade to a "performance check." Our SHAP analysis over named, engineered features is more directly actionable for understanding *which phenological statistics* drive each crop — a qualitative advantage of the lite-learning route.

## Evaluation Caveats

- **In-region only — not a transfer study.** Despite the "generalizability" framing, evaluation never leaves Bei'an County or 2017. Their 91.7% OA / 0.857 Kappa is from a field/grid-grouped split within one county-year and must NOT be read as cross-region or cross-year generalization; it sits on the same rung as our field-grouped CV, not above it. Do not present it as evidence of spatial transfer.
- **Only 10% of labels used.** Train/val/test draw from just 10% of the surveyed fields; the "annual map" is inference over the remaining same-county fields, so the map-vs-ground-truth confusion matrix is partly in-distribution.
- **Class imbalance only partly tamed.** Augmentation helped weak models but hurt the best model's minority recall; the residual "other crops" F1 (0.74) and backscatter-only single-digit wheat F1 show OA/Kappa still mask minority weakness — read per-class F1, not the headline.
- **Compute and data cost not comparable to ours.** Dual-GPU training, 21,257 digitized fields, 26-step sequences, and reproduction of four deep baselines represent a resource tier far above our GPU-free, crowdsourced-label pipeline — the cost asymmetry is itself a finding.
- **Optical contribution is thin.** Only three cloud-free Sentinel-2 scenes were available; the temporal richness comes from SAR. Our optical-only multi-temporal design trades SAR's cloud robustness for pipeline simplicity — a deliberate, citable trade-off.
- **SAR preprocessing burden.** m-chi decomposition via SNAP, speckle filtering, and geocoding add substantial preprocessing not present in our optical feature-extraction pipeline.
