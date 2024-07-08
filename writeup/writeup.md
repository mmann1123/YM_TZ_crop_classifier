---
title: "The document title"
author:
- name: Michael L. Mann
  affiliation: The George Washington University, Washington DC 20052
  thanks: Corresponding author. Email mmann1123@gmail.com
- name: Lisa Colson
  affiliation: USDA Foreign Agricultural Service, Washington DC 20250
- name: Rory Nealon
  affiliation: USAID GeoCenter, Washington DC 20523
header-includes:
   - \usepackage{geometry}
   - |
    ```{=latex}
    \usepackage{fancyhdr}
    \pagestyle{fancy}
    \fancyhf{}
    \rfoot{\thepage}
    \renewcommand{\headrulewidth}{0pt}
    \renewcommand{\footrulewidth}{0pt}
    \fancypagestyle{plain}{
      \fancyhf{}
      \rfoot{\thepage}
      \renewcommand{\headrulewidth}{0pt}
      \renewcommand{\footrulewidth}{0pt}
    }
    ```
abstract: |
  This is the abstract of the document. It contains a brief summary of the content and objectives of the document.
  It consists of two paragraphs.

  This is the second paragraph of the abstract.
---

\newgeometry{margin=1in}  

# Introduction
## Background and Context

<!-- |
├── Background and Context
|    ├── Overview of Remote Sensing Technology
|    └── Applications in [specific field or topic] -->
The free access to remotely sensed data, such as imagery from satellites (e.g. Sentinel-2, LandSat) has revolutionized the field of crop type classification in developing countries. By leveraging the power of advanced imaging technologies combined with machine learning algorithms, researchers and practitioners can now identify and map different crop types over large geographic areas at low cost. This has the potential to improve food security, land use planning, and agricultural policy in regions where ground-based data collection is limited or non-existent.

In recent years, machine learning approaches have emerged as powerful tools for crop type classification using remotely sensed data. Specifically, methods based on machine learning algorithms have gained recognition for their effectiveness in matching valuable spectral information from satellite imagery to observations of crop type for particular locations. Machine learning algorithms, including decision trees, random forests, support vector machines (SVM), and k-nearest neighbors (KNN), have been successfully used to  classify imagery into unique agricultural types. These algorithms leverage the rich spectral information captured by satellite sensors, allowing them to identify distinctive patterns associated with different crop types. By training on large labeled datasets where ground-validation information on crop types is linked to corresponding image patches, these models can effectively learn the relationships between the spectral characteristics of crops and their respective classes.

The strength of traditional machine learning approaches lies in their ability to exploit both the spectral patterns within the remotely sensed data. For instance, decision tree-based algorithms partition the feature space based on the spectral bands, enabling the identification of different crop types based on their unique spectral signatures. Random forests extend this concept by combining multiple decision trees to improve classification accuracy and handle more complex scenarios.

These traditional machine learning approaches offer advantages in terms of interpretability and computational efficiency compared to deep learning architectures. They provide insight into the decision-making process and can be more readily understood and explained by domain experts. Additionally, these methods are generally less computationally demanding and require less training data, making them suitable for applications with limited computational resources.

<!-- ├── Problem Statement
|    ├── Current Challenges
|    └── Research Need -->

The development of salient features on a pixel-by-pixel basis from remotely sensed images remains a challenge. Traditional machine learning algorithms require the extraction of relevant features from the raw data to effectively classify crop types. These features are typically derived from the spectral bands of the satellite imagery, such as the enhanced vegetation index (EVI) and basic time series stastics (e.g. mean, max, minimum, slope) for the growing season. Meanwhile a broader set of time series statistics may be more relevant for a number of applications. For instance the skewedness of EVI might help distinguish crops that greenup early vs later in the season, measures of the numbers of peaks in EVI might help differentiate intercropping or multiple plantings in a season. However, the selection and extraction of these features can be time-consuming and labor-intensive, requiring domain expertise and manual intervention.

Field-collected data provides the necessary validation and calibration for remote sensing-based models. It serves as the benchmark against which the model's predictions are evaluated and refined. Ground truth data collected through field visits, observation, and interactions with local farmers offer essential insights into the specific crop types present in the study area.  Validating and training models with accurate ground reference information ensures that the spectral patterns captured by remote sensing data are correctly associated with the corresponding crop classes. By combining the spectral information from satellite imagery with ground truth data, researchers can develop robust models that effectively differentiate between different crop types based on their unique spectral signatures. 

The collection of field observations and ground truth data is a critical input to the development of machine learning models for crop type classification. However, obtaining accurate and timely ground truth data can be challenging in developing countries due to limited resources, infrastructure, and capacity. In many cases, researchers rely on crowdsourced data from volunteers or citizen scientists to supplement or validate ground truth data collected through traditional methods. Projects like [@tseng2021cropharvest] point to the near complete lack of multi-class crop type datasets globally. This is a significant gap in the field of crop type classification, as the availability of high-quality training data is essential for the development of accurate and reliable machine learning models.


In this study we aim to address two critical challenges in the field of crop type classification: the lack of multi-class crop type datasets and the need for automated methods of developing salient time-series features for agricultural applications.

 We propose a novel approach that combines crowdsourced data with time series features extracted from satellite imagery to classify crop types in Tanzania. By leveraging the power of crowdsourcing and remote sensing technologies, we aim to develop a robust and scalable solution for crop type classification that can be applied in other regions and contexts.

<!-- |
├── Objective of the Study
|    ├── Research Goals
|    └── Scope -->

<!-- ├── Significance and Innovations
|    ├── Contributions to the Field
|    └── Innovations in Methodology or Technology -->



<!-- ├── Literature Review
|    ├── Previous Work
|    └── Distinction from Prior Work -->

<!-- └── Outline of the Paper
    └── Structure of Subsequent Sections -->



## Citation

 
## Mathematics

Here is an example of a mathematical formula in LaTeX:

$$
e^{i\pi} + 1 = 0
$$

## Conclusion

This document is a simple demonstration of MyST Markdown capabilities.

\newpage
# References

<!-- compile working with:
andoc writeup.md --template=mytemplate.tex -o output.pdf --bibliography=refs.bib --pdf-engine=xelatex --citeproc -->