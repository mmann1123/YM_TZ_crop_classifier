---
title: "Response to Reviewers: JSTARS-2025-00807 (Minor Revision)"
subtitle: "Lite Learning: Efficient Crop Classification in Tanzania Using Feature Extraction with Machine Learning & Crowd Sourcing"
author:
- name: Michael L. Mann et al.
date: 2025
documentclass: article
classoption: [11pt]
geometry: margin=1in
---

<!-- compile with:
cd "writeup/Revisions 2"
pandoc reviewer_response_round2.md --template=../mytemplate.tex -o reviewer_response_round2.pdf --bibliography=../refs.bib --pdf-engine=xelatex
-->

# Editorial Decision

Your manuscript JSTARS-2025-00807 has been reviewed and recommended for publication subject to satisfactory response to minor revisions. We appreciate the opportunity to address the remaining reviewer comments.

## Submission Requirements

Per Editor-in-Chief instructions, we are submitting:

1. A revised manuscript with revised portions highlighted
2. This point-to-point response letter to reviewer comments

\newpage

# Reviewer Comments and Responses

## Reviewer 1

### Comment 1: Computational Results

> "In the Introduction, the author compares the use of traditional machine learning with deep learning. The advantage of traditional machine learning is that it requires less computational effort. The author needs to elaborate on the computational results, including training time and prediction time."

**Response:** We have added a new subsection titled "Computational Efficiency" in the Methods section that provides formal complexity analysis and practical timing benchmarks. The section now includes:

1. **LightGBM complexity analysis:**
$$T_{LGBM} = O(n \cdot m) + O(b \cdot k \cdot d)$$
where the first term captures one-time histogram binning across $n$ samples and $m$ features, while the second represents tree construction across $b$ bins, $k$ trees, and maximum depth $d$.

2. **Practical timing:** Training with Optuna hyperparameter optimization (100 trials, 3-fold CV) required approximately 10-15 minutes on a standard laptop (Intel i7, 16GB RAM) without GPU acceleration.

3. **Comparison with deep learning architectures:**
   - LSTM complexity: $T_{LSTM} = O(e \cdot n \cdot t \cdot (h^2 + h \cdot i))$ with quadratic dependence on hidden size
   - 3D U-Net complexity: $T_{3DUNet} = O(e \cdot n \cdot \sum_l k_l^3 \cdot c_{l-1} \cdot c_l \cdot s_l)$ with cubic kernel operations

4. **Inference efficiency:** LightGBM inference scales at $O(n \cdot k \cdot d)$, enabling rapid pixel-wise mapping suitable for operational deployment.

**Location:** Section III (Methods), new subsection "Computational Efficiency"

---

### Comment 2: Deep Learning Comparison

> "The author needs to compare the use of deep learning with proposed methods (traditional machine learning), as lightweight transfer learning models such as the MobileNetV2 model are currently available."

**Response:** We have substantially expanded the Introduction to include:

1. **Specific deep learning results from literature:**
   - Buttar et al. applied U-Net to a crop type dataset in Kenya, achieving F1 = 73.6%
   - Liu et al. achieved F1 = 87.7% using 3D-CNN with ConvRNN on Sentinel-1/2 data
   - Adrian et al. reported 94.1% overall accuracy for 3D U-Net in controlled agricultural settings (Bradford Research Center, Missouri)

2. **Lightweight transfer learning discussion:**
> "Recent advances in lightweight transfer learning architectures, such as MobileNetV2 and EfficientNet, have shown promising results in agricultural applications, particularly for RGB image classification on mobile devices and smart farming systems. These architectures offer reduced computational requirements compared to larger deep learning models while maintaining competitive performance. However, transfer learning approaches face domain adaptation challenges when applied to multispectral satellite imagery, as pre-trained weights are typically derived from natural RGB images rather than remote sensing data."

3. **Complementary benefits of our approach:**
> "Our feature-based approach offers complementary benefits for satellite-based crop classification: improved interpretability through SHAP values that provide agronomically meaningful explanations, accessibility on standard hardware without GPU requirements, and transparency in the feature engineering process that allows domain experts to incorporate their knowledge directly."


**Location:** Section I (Introduction), expanded deep learning discussion

---

### Comment 3: Highlight Method Weaknesses

> "The author also needs to highlight the weaknesses of the method used, as future researchers can still improve its classification performance."

**Response:** We have substantially expanded the limitations section in the Conclusion to include method-specific weaknesses:

> "Several methodological limitations warrant consideration for future applications. Our use of monthly composites, while effective for reducing cloud contamination, may miss rapid phenological changes such as double cropping events or short-duration growth stages that occur within a single month. The linear interpolation applied to fill cloud gaps assumes gradual transitions between observations, which may not accurately capture abrupt changes in crop condition due to management interventions or stress events. Additionally, persistent cloud cover during critical phenological periods could introduce systematic biases in the time-series features. These issues might be addressed by including SAR time series data.  Class boundary decisions, particularly for distinguishing between visually similar land covers such as shrub versus forest, remain challenging and depend heavily on the quality and consistency of training data. Moreover xr_fresh feature extraction, while comprehensive, may not capture all relevant phenological dynamics, suggesting that additional or alternative feature engineering approaches could further enhance model performance."

**Location:** Section VI (Conclusion), new paragraph on methodological limitations

---

## Reviewer 2

### Comment 1: Feature Selection and Model Parameters

> "Appropriately increase the explanation of feature selection and the explanation of model parameters."

**Response:** We have expanded the "Interpretation and Feature Selection" section with additional details:

1. **Variance threshold filtering:**
> "Prior to SHAP analysis, we applied a variance threshold filter to remove features with low discriminative power. Features with variance below 0.5 were excluded, as low-variance features provide minimal information for distinguishing between classes. This preprocessing step reduced the initial feature set substantially while preserving features with meaningful variation across the training samples."

2. **Feature selection rationale:**
> "The choice of ~30 features represents a balance between model parsimony and classification performance; preliminary experiments indicated diminishing returns beyond this threshold while smaller feature sets showed reduced accuracy."

3. **LightGBM hyperparameters:**
> "For the final LightGBM model, Optuna optimized hyperparameters including the number of boosting rounds (up to 10,000 with early stopping after 200 rounds without improvement), learning rate, maximum tree depth, and regularization parameters. The early stopping mechanism prevents overfitting by halting training when validation performance plateaus, while the stratified group k-fold cross-validation ensures robust parameter selection across different data partitions (fields)."

**Location:** Section III (Methods), subsection "Interpretation and Feature Selection"

---

### Comment 2: Comparative Analysis with On-Site Follow-Up Visits

> "Increase the comparative analysis of the results of machine learning models with on-site follow-up visits."

**Response:** We have renamed the section to "Field Data Cleaning and Validation" and expanded it to provide more detail on our photo-based validation approach:

> "The photo-based validation process served as a critical quality control mechanism. Each field photograph was reviewed by researchers familiar with regional crop characteristics to confirm or correct the crop type labels recorded during field visits. This remote verification allowed identification of misclassified observations where field conditions or crop similarity led to labeling errors. Approximately 320 observations were removed during the cleaning process due to factors including unclear photographs, label inconsistencies, coordinate errors, or fields that could not be confidently verified. While this photo-based approach provided valuable validation, it represents a limitation of our methodology: independent post-classification field visits to verify model predictions were not conducted. Future studies would benefit from systematic ground truthing campaigns following model application to assess real-world classification accuracy and identify systematic errors in specific crop types or geographic areas."

**Location:** Section III (Methods), subsection "Field Data Cleaning and Validation"

---
 
We thank both reviewers for their constructive feedback, which has improved the clarity and completeness of the manuscript.
