---
title: "Response to Reviewers: JSTARS-2025-00807"
subtitle: "Lite Learning: Efficient Crop Classification in Tanzania Using Feature Extraction with Machine Learning & Crowd Sourcing"
author:
- name: Michael L. Mann et al.
date: 2025
documentclass: article
classoption: [11pt]
geometry: margin=1in
---

<!-- compile working with:
cd writeup
pandoc "./Revisions 1/reviewer_response.md" --template=mytemplate.tex -o reviewer_response_JSTARS-2025-00807.pdf --bibliography=refs.bib --pdf-engine=xelatex
-->

# Editorial Decision

Your manuscript JSTARS-2025-00807 *Lite Learning: Efficient Crop Classification in Tanzania Using Feature Extraction with Machine Learning & Crowd Sourcing* has been reviewed by the J-STARS Editorial Review Board and recommended for publication subject to satisfactory response to major revisions suggested. It is recommended that you resubmit your manuscript as revised in accordance with the Editorial Review Board comments given below.

## Submission Requirements

Along with the revised manuscript, please provide an item-by-item response to reviewers' comments, including:

- Which suggested changes were accepted and made
- Which were ignored (these should be indicated and justified)
- Where the changes were made in the manuscript (this should include all changes with detailed information)

## Editor-in-Chief Comments

For revision, two files should be submitted:

1. A revised manuscript, in which the revised part should be highlighted in different color, submitted as the main file
2. A point-to-point response letter to the comments submitted as supporting file

**Note:** A "Discussion Section" is suggested for publications in JSTARS. Please improve the manuscript from this aspect, which will make the manuscript more likely to get accepted.

## Associate Editor Comments

Your paper reviewed by experts has been considered suitable for publication on JSTARS after major revision. Please proceed in reviewing the paper by closely following the reviewers' suggestions.

\newpage

# Reviewer Comments and Responses

## Reviewer 1

### Summary

This paper addresses several important challenges in developing efficient crop-type classification models using the traditional machine-learning approach in data-scarce environments, including the scarcity of training datasets collected from crop fields, especially in developing countries. Combining the dataset collected from crop fields with the imagery datasets extracted using filter-based methods is crucial for achieving high classification accuracy. Crowdsourced data collected by volunteers can serve as a potential alternative, but these data are typically lack proper validation. To address this issue, the author proposes a novel methodology leveraging KoboToolbox, YouthMappers participants, and Sentinel-2 satellite imagery, among others, to extract and validate crowdsourced data and time-series imagery features. The authors also propose a traditional machine learning-based classification model that utilizes the integrated dataset of time-series features with crowdsourced, field-validated crop-type labels to validate their approach. The proposed methods show promise by achieving high classification accuracy, evidenced by a Cohen's Kappa score of 0.82 and an F1-micro score of 0.85.

### Strengths

- The paper is well-structured and follows a clear and logical progression
- The authors provide a significant introduction and background, giving insight into the challenges that limit the implementation of an efficient crop-type classification
- The authors contribute to proposing novel methods for extracting and validating crowdsourced data

### Minor Corrections

1. Ensure that reference 25 fits within the column width
2. The sentence refers to "as seen in the crop calendar in Figure 2 below" Fig 2; However, the figure is displayed above the sentence, not below

### Response to Reviewer 1

We thank Reviewer 1 for their positive assessment of our work. We have addressed both minor corrections:

- **Reference 25**: Has been reformatted to fit within the column width
- **Figure 2 reference**: The text has been updated to correctly reflect the figure placement

\newpage

## Reviewer 2

### Summary

This study introduces a novel approach to traditional machine learning methodology for crop type classification in Tanzania, by integrating crowdsourced data with time-series features extracted from Sentinel-2 satellite imagery. Overall, the content of the paper is presented too succinctly and requires further refinement in expression, particularly in the description of data and methodology, the presentation of results and figures, as well as the discussion section, which is currently lacking.

### Major Comments

1. It is recommended to divide Section II (DATA & METHODS) into two separate sections. A schematic diagram should be provided to illustrate the data, especially the crowdsourced data. In the methodology section, a flowchart should be added, along with an accuracy evaluation component
2. The conclusion section is also overly concise. For instance, in Part B (Land Cover and Crop Type), a more detailed description of Figure 3 is suggested
3. It is advisable to present the Discussion as an independent section and enrich its content
4. Whether to use a colon after figure titles (e.g., "Fig. 6:") should strictly follow the journal's formatting guidelines
5. The font size in the figures is too small and lacks clarity, as seen in Figure 7
6. It is recommended to include comparative experiments with other methods
7. It is suggested to add visualizations and descriptions of the classification results

### Response to Reviewer 2

We thank Reviewer 2 for their thorough and constructive feedback. We have addressed each comment as follows:

#### Comment 1: Separate data and methods sections; add flowcharts

**Response:** We have made substantial improvements to the structure and clarity of our data and methods sections:

- Separated the DATA and METHODS sections into distinct parts for improved organization
- Added two comprehensive flowcharts:
  - Figure showing data collection and cleaning methodology
  - Figure showing analytical methods workflow
- Although constrained by the 2-column format, we have increased font sizes in all figures for better readability

#### Comment 2 & 3: Enhance discussion and conclusion sections

**Response:** We have significantly expanded the data, methods, results, and discussion content throughout the manuscript:

**1. Data Source Clarification**

We now explicitly state that Google imagery was only used for non-agricultural land cover types:

> "Additional training data for non-agricultural sites was collected utilizing high resolution imagery from Google Earth."

**2. Enhanced Field Data Collection Description**

We heavily revised the description of field data collection training objectives. The new text includes detailed information on:

- Field size requirements: YouthMappers were instructed to identify only fields measuring 30 meters or greater across, ensuring adequate pixel coverage in Sentinel-2 imagery
- Field composition considerations: Teams were trained to prioritize clear, open fields with single crop types to minimize spectral mixing from trees, buildings, or other obstructions
- Spatial distribution requirements: A minimum separation distance of 1 kilometer between sampling locations was established to reduce spatial autocorrelation while balancing logistical constraints
- Timing considerations: Data collection aligned with mid to late reproductive crop stages for maximum canopy cover, though 2023 drought conditions affected some fields

**3. Comprehensive Data Cleaning Section**

We added a new "Field Data Cleaning" subsection describing the thorough quality control process:

- Duplicate entry removal to prevent redundancy and bias
- Missing data handling through auxiliary sources or exclusion
- Visual inspection using in-situ photos taken by students
- Crop type label standardization (correcting typos, unifying naming conventions)
- Geographic coordinate validation
- Final dataset: 1,400 observations (reduced from initial 1,720)

**4. Clarified Model Performance Metrics**

We reorganized content into separate "Model Selection" and "Performance Evaluation" subsections that more clearly explain:

- Optuna-based hyperparameter optimization across LightGBM, SVC, and RandomForest
- Feature preprocessing (standard scaling, variance threshold filtering)
- Cross-validation strategy (stratified group k-fold with 3 splits to prevent field-level data leakage)
- Primary evaluation metric (kappa statistic) and its rationale for imbalanced datasets

**5. Enhanced SHAP Analysis Discussion**

We substantially expanded the interpretation of feature importance with detailed agronomic explanations:

- B11.mean (SWIR): Distinguishes dryland-adapted crops (sunflower, sorghum, millet) from water-demanding crops by capturing persistent differences in canopy water status. The SWIR bands are sensitive to plant water content. Sunflower's deep taproot system, sorghum and millet's drought tolerance and waxy leaf coatings create characteristically different SWIR reflectance patterns than maize and rice.

- B12.absolute.sum.of.changes (SWIR): Differentiates cotton from natural vegetation (shrubs, forests) by capturing temporal volatility. Cotton exhibits pronounced variability driven by distinct phenological transitions (establishment, vegetative growth, flowering, boll development, defoliation) and management interventions, creating sharp changes in canopy water content that B12 captures.

**6. Enhanced Figure 3 Discussion**

We added detailed discussion of the land cover distribution:

> "Peanuts, soybeans and okra are among the least represented land cover types in the dataset, highlighting the challenges associated with collecting sufficient training data for these categories. This figure also includes land cover types such as water, forest, shrub, and tidal areas, which are essential for providing context to the landscape but were not the focus of this study. The varied distribution of land cover types underscores the complexity of the classification task and the need for robust modeling techniques to accurately capture this diversity."

#### Comment 4: Figure formatting consistency

**Response:** We have reviewed all figure titles to ensure consistency with journal formatting guidelines.

#### Comment 5: Figure clarity

**Response:** We have increased font sizes in all figures to the maximum extent possible given the 2-column format constraint, improving readability substantially.

#### Comment 6: Comparative experiments

**Response:** We have a comparison with other land cover models for Tanzania and Africa. A new figure compares our model's performance metrics (Kappa, F1, accuracy) against multiple published models from the literature. Our model achieves superior performance (Kappa=0.82, F1=0.85) despite tackling the significantly more challenging task of multi-class crop type classification versus simple undifferentiated "agriculture" classification performed by most comparison models.

#### Comment 7: Classification result visualizations

**Response:** We have added a new "Land Cover Product" section with two detailed visualizations showing:

- RGB composite imagery for geographic context
- Final crop classification results
- Three selected time-series features that contribute to the classification at each site

These visualizations demonstrate the model's efficacy and provide insight into how different time-series features (e.g., B11_mean, hue_quantile_q_0.05, EVI_standard_deviation) identify critical signals for distinguishing crop types. The visualizations reveal both the model's strengths and areas needing improvement (e.g., urban/forest/shrub classification would benefit from additional training data).

#### Enhanced Conclusion with Comprehensive Limitations Section

**Response:** We have substantially revised the conclusion to better summarize our objectives and findings, and added a comprehensive limitations section addressing:

**Data Collection Constraints:**

- 2023 drought affected crop health and planting schedules, resulting in fields at varying phenological stages
- Concentrated data collection window (April-May 2023) captured late growing season, potentially missing spectral signatures from earlier phenological stages
- Crop type imbalance led to dropping underrepresented crops (peanuts, soybeans, okra)
- Confusion between similar crops (cassava/maize) indicates need for additional discriminative features

**Geographic Generalizability:**

- Model trained exclusively on three northern Tanzania districts (Arusha, Dodoma, Mwanza)
- Transferability to other regions with different agro-ecological conditions, farming practices, or crop varieties remains untested
- Spectral/temporal signatures vary with climate, soil, and management practices

**Interpretability Trade-offs:**

- Despite SHAP values, 33 final features (from hundreds of candidates) presents interpretation challenges
- Future work should explore more parsimonious feature sets balancing accuracy with interpretability

**Future Research Directions:**

- Expanding geographic coverage to assess model transferability
- Incorporating multi-temporal data collection throughout the growing season
- Developing methods to handle class imbalance
- Exploring more interpretable feature sets

#### Note on Results/Discussion Structure

Although we appreciate the suggestion of fully separating results from discussion, we have chosen to maintain a combined "Results & Discussion" structure to improve accessibility for non-technical readers. However, we have significantly enriched the discussion content throughout this section as described above, providing agronomic interpretation of SHAP values, model performance context, and methodological insights that substantially enhance the scientific contribution of the manuscript.
