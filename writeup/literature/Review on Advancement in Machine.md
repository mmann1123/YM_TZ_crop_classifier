# Review on Advancement in Machine Learning and Deep Learning Techniques for Crop Classification

**Citation:** Yadav, S.; Sharma, S.; Chaudhary, P. (2024). "Review on Advancement in Machine Learning and Deep Learning Techniques for Crop Classification." In *Advancements in Communication and Systems* (eds. A.K. Tripathi & V. Shrivastava), Computing and Intelligent Systems, Soft Computing Research Society (SCRS), India, pp. 593–610. DOI: `10.56155/978-81-955020-7-3-53` (verified against Crossref — title "Review on Advancement in Machine Learning and Deep Learning Techniques for Crop Classification", authors Yadav/Sharma/Chaudhary, publisher Soft Computing Research Society, all match). Crossref registers the publication year as **2024**; the book chapter itself is dated **2023** — cite with care. Affiliation: The NorthCap University, India.

> This is a survey / bibliometric review. Per authoring rule 7, the per-study evaluation-protocol question is skipped; the Relevance section focuses on conceptual implications — most importantly, that this independent survey's own conclusion calls for lightweight / edge models, echoing our "lite learning" thesis.

## Objectives

- Provide a comprehensive review of remote-sensing technologies (UAVs, satellites — Landsat, Sentinel — and image-fusion approaches) and AI algorithms (classical ML: SVM, Random Forest, Decision Tree; deep learning: CNN, RNN, neural networks) used for crop classification.
- Perform a bibliometric-style survey of how AI + image processing is applied to crop identification over large agricultural areas.
- Identify key challenges (data availability, scalability, model generalization, multi-source integration) and propose future research directions.

## Methods

- A narrative literature survey organized into: (1) RS technologies for image capture and fusion; (2) ML vs. DL models and their evaluation metrics; (3) discussion of challenges and research gaps.
- Catalogs studies in two summary tables — Table 1 (crop classification using UAV imagery) and Table 2 (crop classification using satellite imagery) — each listing dataset, model, and headline performance.
- Reviews evaluation metrics in the abstract (confusion matrix, accuracy, precision, recall, F1, Cohen's Kappa), noting F1 is appropriate for imbalanced datasets — but does **not** audit or standardize the protocols behind the numbers it tabulates.

**Evaluation protocol (of the review itself):** Not applicable — it is a survey with no experiment of its own. Critically, the studies it tabulates are reported with **unstated and heterogeneous evaluation protocols**: the table cells give a parade of headline numbers (e.g. OA 93.95%/Kappa 0.929; OA 92.80%/Kappa 0.9206; AA 95.3%/Kappa 0.89; accuracy 97.43%; "highest accuracy 99.8%") drawn from different sensors, regions, class sets, and (unspecified) splitting schemes. None can be placed on the pooled-pixel → field-wise-CV → spatial-holdout → cross-domain spectrum, because the review does not record whether any of them used field-disjoint splitting or a spatial holdout. They are therefore not comparable to one another, let alone to our field-grouped-CV Cohen's Kappa.

## Key Findings

- CNN-based models using hyperspectral and multitemporal fused imagery are the prevailing approach in the recent crop-classification literature; LSTM-based models are preferred when the input is temporal or fused.
- **Random Forest is repeatedly noted as competitive** — the conclusion states the RF machine-learning model "has also demonstrated high classification accuracy compared to other models," a useful independent acknowledgment that classical ML remains a strong baseline.
- Multimodal fusion (multispectral + hyperspectral + SAR + UAV), data augmentation, transformer models, and transfer learning are flagged as the main routes to overcome scarce labeled data.
- Persistent open problems: **multiple-crop / minority-crop classification remains under-explored**; "some crops have similar visual characteristics that make them difficult to distinguish"; scalability to larger areas and model generalization / domain adaptation across regions are open gaps; label scarcity limits robustness.
- **The conclusion independently calls for lightweight, edge-deployable models.** The authors write that real-world, vast-area monitoring "necessitates the creation of a lightweight model, a distributed computation system, and integration with edge AI," and note (citing the surveyed work) the importance of training/testing efficiency and that some compute-bound DL models "remain underfit and need integration with more high-performance computational platforms."

## Relevance to Our Crop-Classification Study

- **Independent, third-party statement of our thesis.** Our manuscript argues for cheap, GPU-free, interpretable feature-extraction + tree-based ML over compute-bound deep networks. This survey — written by authors with no stake in our approach — reaches the same destination from the deep-learning side: its own forward-looking conclusion is that the field *needs* lightweight, edge-deployable, computationally efficient models. That is a strong citation to support the "lite learning" framing, precisely because it is not advocacy from within our camp.
- **Backs the Random-Forest baseline.** Its note that RF "demonstrated high classification accuracy compared to other models" supports our use of tree-based learners (LightGBM, RF) as more than a fallback — they are a recognized strong baseline.
- **Names the minority-crop and generalization gaps we target.** The review flags minority/multiple-crop classification, label scarcity, and cross-region generalization as the open frontiers. Our manuscript speaks to exactly those: hard minority crops (cassava, millet, sunflower, sorghum, cotton), crowdsourced YouthMappers labels for the scarcity problem, and field-disjoint grouped CV for honest generalization measurement.
- **Use it for landscape framing, not numbers.** It is well-suited to support introductory statements about where the field is heading (toward fusion, transformers, and — by the authors' own conclusion — lightweight/edge deployment), and about RF's competitiveness.

## Evaluation Caveats

- **CAVEAT — do not cite any tabulated number as a benchmark.** The roughly two dozen headline OA / Kappa / F1 figures in Tables 1–2 come with **unknown and unaudited evaluation protocols**, heterogeneous datasets, sensors, and class sets. There is no indication of field-disjoint splitting or spatial holdouts for any of them, so any single figure is, by default, suspect for in-region/pooled-pixel leakage. None is comparable to our field-grouped-CV Cohen's Kappa / per-class F1, and none should be quoted in our manuscript as a performance bar. Treat the tables as a map of *which methods exist*, never as a leaderboard.
- **No protocol scrutiny.** The review does not discuss spatial autocorrelation, field-wise splitting, or train/test spatial separation — exactly the rigor axis our work emphasizes — so it offers no support on the evaluation-rigor question and silently propagates whatever leakage the primary studies contain.
- **Class-imbalance silence in the cited numbers.** Although the abstract correctly notes F1 for imbalanced data, the tabulated results are dominated by overall-accuracy figures that can mask poor minority-crop recall; the review does not report per-class minority performance for the studies it cites.
- **Non-archival venue and year ambiguity.** It is a conference/edited-book chapter (SCRS) with a 2023 chapter date vs. a 2024 Crossref registration — minor, but worth getting right in the bibliography. It is a comparator-of-comparators (a survey), not a primary result, so it never clears our evaluation bar itself.
