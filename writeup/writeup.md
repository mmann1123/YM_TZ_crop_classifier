---
title: "Time-Series Analysis of Crop Types in Tanzania Using Crowdsourced Data and Sentinel-2 Imagery"
author:
- name: Michael L. Mann
  affiliation: The George Washington University, Washington DC 20052
  thanks: Corresponding author. Email mmann1123@gmail.com
- name: Lisa Colson
  affiliation: USDA Foreign Agricultural Service, Washington DC 20250
- name: Rory Nealon
  affiliation: USAID GeoCenter, Washington DC 20523
- name: Stellamaris Wavamunno Nakacwa
  affliation: 
header-includes:
   - \usepackage{geometry}
   - \usepackage{pdflscape}
   - \usepackage{longtable}
   - \usepackage{fancyhdr}
   - \usepackage{float}
   - \usepackage{graphicx}
   - \usepackage{amsmath}
   - \usepackage{amsfonts}
   - \usepackage{lineno} 
   - \usepackage{booktabs}
   - |
    ```{=latex}
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
  This study introduces a robust methodology for crop type classification in Tanzania, utilizing a novel integration of time-series feature from Sentinel-2 satellite data and crowdsourced ground truth data collected by the YouthMappers network. By combining advanced remote sensing techniques with extensive local knowledge, the research addresses significant gaps in agricultural monitoring within resource-limited settings. The application of machine learning algorithms to analyze temporal and spectral data enables the precise identification of crop types, showcasing the enhanced accuracy and utility of combining technological and human resources. We achieve 0.80 kappa accuracy scores, across a diverse multi-class dataset including challenging crops including cassava, millet, sorghum, and cotton amongst other. This methodological innovation not only improves crop classification accuracy but also contributes to sustainable agricultural practices and policy-making in developing countries, making a significant impact on food security and land management.
---

\linenumbers  
\modulolinenumbers[1]  
\pagewiselinenumbers  
\newgeometry{margin=1in}
  
@stella add you details above
<!-- 
Look at https://mpastell.com/pweave/docs.html -->

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
<!-- |
├── Objective of the Study
|    ├── Research Goals
|    └── Scope -->

In this study we aim to address two critical challenges in the field of crop type classification: the lack of multi-class crop type datasets and the need for automated methods of developing salient time-series features for agricultural applications.

 We propose a novel approach that combines crowdsourced data with time series features extracted from satellite imagery to classify crop types in Tanzania. By leveraging the power of crowdsourcing and remote sensing technologies, we aim to develop a robust and scalable solution for crop type classification that can be applied in other regions and contexts.


<!-- ├── Significance and Innovations
|    ├── Contributions to the Field
|    └── Innovations in Methodology or Technology -->



<!-- ├── Literature Review
|    ├── Previous Work
|    └── Distinction from Prior Work -->

<!-- └── Outline of the Paper
    └── Structure of Subsequent Sections -->




# Data & Methods

Data for this study were collected from multiple sources, including satellite imagery, and crowdsourced ground truth observations. 

## Study Area

@anyone _____________ revise this _________________ 
The study area is located in Tanzania, a country in East Africa known for its diverse agricultural landscape. We focus on the 15 northern most ?states? including Arusha, Dodoma, Geita, Kagera, Katavi, Kigoma, Kilimanjaro, Manyara, Mara, Mwanza, Shinyanga, Simiyu, Singida, Tabora and Tanga.  The region is characterized by a mix of smallholder farms, commercial plantations, and natural vegetation, making it an ideal location for studying crop type classification.  ??? 

@rory or stella - Can you create a *pretty* map of the study area? the bounds file is northern_TZ_states.geojson

### Crowd Sourced Crop Data

@anyone __________describe YM data collection__________

(keep this ->) The land cover classes included in the analysis were rice, maize, cassava, sunflower, sorghum, urban, forest, shrub, tidal, cotton, water, and millet. Classes excluded due to irrelevance or insufficient data included 'Don't know', 'Other (specify later)', 'water body', 'large building', and several others which were either ambiguous or represented a negligible fraction of the data.

Additional training data was collected utilizing high resolution imagery from Google Earth. This data was used to supplement the crowdsourced data and improve the model's ability to distinguish between crops and more common land cover types like forests, urban areas, and water.  

#### Data Collection Methods

__________ @StellaWava @RoryNealon @LColson____________

### Satellite Imagery

Satellite imagery was obtained from the Sentinel-2 satellite constellation, which provides high-resolution multispectral data at 10-meter spatial resolution. The imagery was acquired over the study area during the growing season, capturing the spectral characteristics of different crop types. The Sentinel-2 data were pre-processed to remove noise and atmospheric effects, ensuring that the spectral information was accurate and reliable for classification purposes.

In our study, cloud and cloud shadow contamination was mitigated using the 's2cloudless' machine learning model on the Google Earth Engine platform. Cloudy pixels were identified using a cloud probability mask, with pixels having a probability above 50% classified as clouds. To detect cloud shadows, we used the Near-Infrared (NIR) spectrum to flag dark pixels not identified as water as potential shadow pixels. The projected shadows from the clouds were identified using a directional distance transform based on the solar azimuth angle from the image metadata. A combined cloud and shadow mask was refined through morphological dilation, creating a buffer zone to ensure comprehensive coverage. This mask was applied to the Sentinel-2 surface reflectance data to exclude all pixels identified as clouds or shadows, enhancing the reliability of the dataset for environmental analysis.

Monthly composites were collected for January through August of 2023 for the the bands B2, B6, B8, B11, and B12. We also calcuate the Enhanced Vegetation Index (EVI) and 'hue' the color spectrum value. This computed hue value provides the basic color as perceived in the color wheel, from red, through green, blue, and back to red. Due to the high prevelence of clouds in the region, we used linear interpolation to fill in missing data in the time series using `xr_fresh` [@xr_fresh_2021]. These bands were selected based on their relevance to crop type classification and their ability to capture the unique spectral signatures of different crops. The composites were used to generate time series features for each pixel in the study area, providing valuable information on the temporal dynamics of crop growth and development.

### Time Series Features

Time series features capture the temporal dynamics of crop growth and development, providing valuable information on the phenological patterns of different crops. We leverage the time series nature of the satellite imagery to extract relevant features for crop type classification.

In this study, we utilized the `xr_fresh` toolkit to compute detailed time-series statistics for various spectral bands, facilitating comprehensive pixel-by-pixel temporal analysis [@xr_fresh_2021]. The `xr_fresh` framework is specifically designed to extract a wide array of statistical measures from time-series data, which are essential for understanding temporal dynamics in remote sensing datasets.

The metrics computed by `xr_fresh` in this study include basic statistical descriptors, changes over time, and distribution-based metrics, applied to each pixel's time series for selected spectral bands (B12, B11, hue, B6, EVI, and B2). The list of computed time-series statistics encompasses:

- **Energy Measures**: Absolute energy which provides a sum of squares of the values.
- **Change Metrics**: Absolute sum of changes to quantify overall variability, mean absolute change, and mean change.
- **Autocorrelation**: Calculated for three lags (1, 2, and 3) to assess the serial dependence at different time intervals.
- **Count Metrics**: Count above and below mean, capturing the frequency of high and low values relative to the average.
- **Extreme Values**: Day of the year for maximum and minimum values, providing insight into seasonal patterns.
- **Distribution Characteristics**: Kurtosis, skewness, and quantiles (5th and 95th percentiles) to describe the shape and spread of the distribution.
- **Variability Metrics**: Standard deviation, variance, and whether variance is larger than standard deviation to evaluate the dispersion of values.
- **Complexity and Trend Analysis**: Time series complexity and symmetry looking, adding depth to the analysis of temporal patterns.

For a full list of the time series features extracted in this study and thier descriptions, please refer to the Appendix.

Notably, certain statistics like `longest_strike_above_mean` and `longest_strike_below_mean` were excluded due to computational constraints related to GPU memory capacity on the JAX platform. Additionally, some planned metrics such as OLS slope, intercept, and R-squared calculations were not implemented.

The integration of `xr_fresh` into our analytical workflow allowed for an automated and robust analysis of temporal patterns across the study area. By leveraging this toolkit, we could efficiently process large datasets, ensuring that each pixel's temporal dynamics were comprehensively characterized, which is critical for accurate environmental monitoring and change detection.

### Data Extraction

To partially accound for variation in field size we extract pixels based on a buffer around field point locations. Small fields were buffered by only 5 meters, medium by 10m and large by 30m. This approach allowed us to capture the time series features from the surrounding area, providing a more comprehensive representation of the field's characteristics. The use of larger buffers was explored but found to decrease model performance as fields tended to be heterogenous - for instance containing patches of trees. To account for this in our modeling we treat observations from the same field as a "group" in our cross-validation scheme - as described below.

### Machine Learning Models

In our study, we utilized the extracted time-series features from satellite imagery, described above, to analyze crop classifications.  Notably, features were centered and scaled from the `scikit-learn` library to normalize the data, followed by the application of a variance threshold method to reduce dimensionality by excluding features with low variance [@scikit-learn].

In this study, we employed `Optuna`, an optimization framework, to conduct systematic model selection and hyperparameter tuning [@optuna_2019]. Our methodology involved defining a study using Optuna where each trial proposed a set of model parameters aimed at optimizing performance metrics. Specifically, we used stratified group k-fold cross-validation with the number of splits set to three, ensuring that samples from the same field were not split across training and validation sets to prevent data leakage. The scoring metric utilized was the kappa statistic, chosen for its suitability in evaluating models on imbalanced datasets.

This approach allowed us to rigorously evaluate and compare different classifiers, including LightGBM, Support Vector Classification (SVC), and RandomForest, and their configurations under a variety of conditions. The final selection of the model and its parameters was based on the ability to maximize the kappa statistic, ensuring that the chosen model provided the best possible performance for the classification of land cover types in our dataset.

### Interpretation and Feature Selection

To interpret the contributions of individual features to the model predictions, we employed SHapley Additive exPlanations (SHAP) [@shaps_2017]. This approach, based on game theory, quantifies the impact of each feature on the prediction outcome, providing insights into which features are most influential in determining land cover types.

In our feature selection process, we incorporate both the mean and maximum SHAP values to comprehensively assess the influence of features on model predictions. The mean of the absolute SHAP values across all samples, provides a measure of the average impact of each feature, highlighting its overall importance across the dataset. This approach emphasizes features that consistently affect the model’s output but might underrepresent the significance of features causing substantial impacts under specific conditions. To address this, we also consider the maximum absolute SHAP values. Sorting features by their maximum absolute SHAP values allows us to identify those that have significant, albeit possibly infrequent, effects on individual predictions. This method ensures that features crucial for particular scenarios are not overlooked, thus offering a more nuanced understanding of feature importance that balances general influence with critical, situation-specific impacts.

Feature selection then is the union of the top 30 time series features found with both the mean and maximum SHAP values, resulting in 33 total features. This approach ensures that the selected features are both consistently influential across the dataset and capable of exerting substantial impacts under specific conditions, providing a comprehensive set of features for model training and evaluation.

## Results & Discussion

### Crowd Sourced Data

In addressing the significant gap in available crop type datasets, particularly in developing regions, this study harnessed the power of crowdsourced data to enhance the robustness and applicability of our machine learning models. Crowdsourced data collection, an innovative approach in the agricultural domain, involves gathering data from a large number of volunteers or citizen scientists, who provide valuable ground truth information. This method has proven especially useful in areas where traditional data collection methods are challenging due to logistical, financial, or infrastructural constraints. 

By leveragign the YouthMappers student organization, with over 400 active chapter in _____ countries, we were able to collect a large dataset of crop type observations in Tanzania. Moreover this exercise provided an important opportunity for students to gain practical experience in data collection, analysis, and interpretation, contributing to their professional development and capacity building in the geospatial domain.

 \begin{table}[h!] \centering \begin{tabular}{|>{\bfseries}c|m{4cm}|m{4cm}|m{4cm}|} \hline \textbf{} & \textbf{Arusha} & \textbf{Mwanza} & \textbf{Dodoma} \\ \hline \begin{itemize} \item Maize \item Rice \item Sorghum \& Millet \end{itemize} & \begin{itemize} \item Maize \item Cotton \item Rice \item Peanuts or Groundnut \end{itemize} & \begin{itemize} \item Sorghum \item Maize \item Millet \item Sunflower \item Peanuts or Groundnut \item Cotton Fields \end{itemize} \\ \hline \end{tabular} \caption{Main Crops by Region in Tanzania} \end{table}
Table XX: Main Crops by Region in Tanzania


@ Someone finish ___________ 

#### Challenges and Lessons Learned

???????????? ______________?????????????

### Land Cover and Crop Type

The distribution of primary land cover types within the training dataset used for the model is represented in Figure \ref{fig:lc_percentages}. The dataset consists of a diverse range of land cover types, each contributing differently to the total number of observations. Maize is the most prevalent land cover type, accounting for the highest percentage of the observations, followed by rice and sunflower. This is indicative of the agricultural dominance in the region being studied. Lesser common land covers such as millet, sorghum, and urban areas represent intermediate percentages, suggesting a varied landscape that includes both agricultural and urbanized zones.
 
\begin{figure}[H]
    \centering
    \includegraphics[width=0.8\linewidth]{/home/mmann1123/Documents/github/YM_TZ_crop_classifier/writeup/figures/primary_land_cover.png} % Adjust the path and options
    \caption{Land Cover Percentages}
    \label{fig:lc_percentages} %can refer to in text with \ref{fig:lc_percentages}
\end{figure}



### Feature Importance

The interpretation of model behavior using SHAP values has allowed for a deeper understanding of how different spectral features impact the model's predictions, which is critical for refining the feature selection process. By analyzing both the mean and maximum SHAP values, we were able to prioritize features based on their overall impact as well as their critical contributions to specific model decisions.

In the two summary plots below, SHAP values for each feature to identify how much impact each feature has on the model output for individuals in the validation dataset. Features are sorted by the sum of the SHAP value magnitudes across all samples. The figures bar color (hue) represents the mean contributions to explaining each land class value. This visualization provides a comprehensive overview of the feature importance, highlighting the key predictors that drive the model's predictions. For example, features that are highly influential for "maize" may not be as impactful for "rice" or "sorghum", reflecting the unique spectral signatures of these crops.

#### Mean SHAP Values

In Figure \ref{fig:mean_shaps}, the mean SHAP values provide insights into the average impact of each feature across all predictions. This analysis highlights the features that consistently influence the model's output across various scenarios. For example, the mean value of B11 (B11.mean) and the 5th percentile of hue (hue.quantile.q.0.05) features were found to have substantial average impacts on model outputs, suggesting their strong relevance in distinguishing between different crop types. Reflecting on the colors of the bars we can see that 'B11.mean' is important in distinguishing sunflow, sorghum, and millet to a roughly equal degree, and has some small impact on distiguishing other classes. While 'hue.quantile.q.0.05' has the strongest effect distiguishing rice, sunflower, and to a lesser degree cotton. Looking down the list we can see that features like "EVI.standard.deviation" are most effective at isolating urban areas, and 'B12.mean.second.derivative.central' substantively differentiates shrub from other classes. Note that the mean second derivative of B12 is a measure of the rate of change of the rate of change of the B12 band, so positive values indicate increasing rate of change (increasingly upward trend), and negative values decreasing rate of change (increasingly downward trend).

\begin{figure}[H]
    \centering
    \includegraphics[width=0.8\linewidth]{/home/mmann1123/Documents/github/YM_TZ_crop_classifier/writeup/figures/mean_shaps_importance_30_LGBM_kappa_3.png} % Adjust the path and options
    \caption{Top 20 Mean SHAP Feature Importance by Land Cover Type}
    \label{fig:mean_shaps} %can refer to in text with \ref{fig:lc_percentages}
\end{figure}


### Maximum SHAP Values

On the other hand, Figure \ref{fig:max_shaps}, maximum SHAP values uncover features that, while perhaps not consistently influential, have high impacts under particular conditions. This aspect of the analysis is crucial for identifying features that can cause significant shifts in model output, potentially corresponding to specific agricultural or environmental contexts. Features such as "hue.median" and "B11.maximum" show high maximum SHAP values, indicating their pivotal roles in certain classifications. For instance,  "B11.maximum" reflects peak reflectance in the Short-Wavelength Infrared (SWIR), which could be critical in identifying crops at their maximum biomass, like sunflower at full bloom compared to other crops at different stages of growth.  

\begin{figure}[H]
    \centering
    \includegraphics[width=0.8\linewidth]{/home/mmann1123/Documents/github/YM_TZ_crop_classifier/writeup/figures/mean_shaps_importance_30_LGBM_kappa_3.png} % Adjust the path and options
    \caption{Top 20 Max SHAP Feature Importance by Land Cover Type}
    \label{fig:max_shaps} %can refer to in text with \ref{fig:lc_percentages}
\end{figure}
 
The final selection of features for model training was carefully curated to include all 30 of both the highest mean and maximum SHAP values, ensuring a comprehensive set of predictors for accurate and reliable classification of crop types in Tanzania. This strategic selection process has not only improved model accuracy but also enhanced our understanding of the spectral characteristics most relevant for distinguishing among the diverse agricultural landscape of the region.

### Model Performance

The classification model demonstrated robust performance across multiple land cover classes, as evidenced by the out-of-sample mean confusion matrix with a Cohen's Kappa score of 0.7959, indicating substantial agreement between predicted and actual classifications. The confusion matrix (Figure \ref{fig:oos_confusion}) shows high diagonal values for most classes, highlighting the model's ability to accurately identify specific land covers. For instance, 'rice' and 'urban' categories achieved classification accuracies of 90% and 94%, respectively. Other well-classified categories included 'forest' and 'millet', each with over 70% accuracy. However 'forest' is primarily confused with the category 'shrub', which is likely a result of poor training data obtained from high-res imagery.

Categories such as 'sorghum' and 'cotton' displayed moderate confusion with other classes, indicating potential areas for model improvement, especially in distinguishing features that are common between similar crop types. Notably, the 'other' category showed a broader distribution of misclassifications, likely due to its encompassing a diverse range of less frequent land covers, achieving a lower accuracy of 40%.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.8\linewidth]{/home/mmann1123/Documents/github/YM_TZ_crop_classifier/writeup/figures/final_confusion_final_model_selection_no_kbest_30_LGBM_kappa_3.png} % Adjust the path and options
    \caption{Out of Sample Confusion Matrix}
    \label{fig:oos_confusion} %can refer to in text with \ref{fig:lc_percentages}
\end{figure}

\begin{table}[h!]
\centering
\begin{tabular}{@{}ll@{}}
\toprule
Metric              & Value                 \\ \midrule
Balanced Accuracy   & 0.79    \\
Kappa Accuracy      & 0.80    \\
Accuracy            & 0.82    \\
F1 Micro Accuracy   & 0.82    \\ \bottomrule
\end{tabular}
\caption{Summary of Classification Metrics}
\label{tab:metrics}
\ref{tab:metrics}
\end{table}

The overall high performance across the majority of categories suggests that the model is effective for practical applications in land cover classification, though further refinement is recommended for categories showing lower accuracy and higher misclassification rates.



## Conclusion

In this study, we introduced a novel methodology for crop type classification in Tanzania, leveraging crowdsourced data and time series features extracted from Sentinel-2 satellite imagery. By combining advanced remote sensing techniques with local knowledge, we addressed significant gaps in agricultural monitoring within resource-limited settings. The application of machine learning algorithms to analyze temporal and spectral data enabled the precise identification of crop types, showcasing the enhanced accuracy and utility of combining technological and human resources.

Our results demonstrated the effectiveness of the proposed methodology, achieving a Cohen's Kappa score of 0.7975 across a diverse multi-class dataset. The model accurately classified a range of challenging crops, including cassava, millet, sorghum, and cotton, among others. The integration of crowdsourced data and time series features from satellite imagery provided valuable insights into the temporal dynamics of crop growth and development, enhancing the accuracy and reliability of the classification model.

The interpretation of feature importance using SHAP values allowed for a deeper understanding of the model's behavior and the key predictors driving its predictions. By identifying the most influential features across different land cover types, we refined the feature selection process, ensuring that the selected features were both consistently influential and capable of exerting substantial impacts under specific conditions.

In conclusion, this methodological innovation not only improves crop classification accuracy but also contributes to sustainable agricultural practices and policy-making in developing countries. By combining advanced remote sensing technologies with crowdsourced data, we have made a significant impact on food security and land management, paving the way for future research and applications in agricultural monitoring and environmental analysis.



<!-- compile working with:
pandoc writeup.md --template=mytemplate.tex -o output.pdf --bibliography=refs.bib --pdf-engine=xelatex --citeproc 
-->

```{=latex}
\newpage
```



## Appendix

### Time Series Features Description

The following table provides a comprehensive list of the time series features extracted from the satellite imagery using the `xr_fresh` module. These features capture the temporal dynamics of crop growth and development, providing valuable information on the phenological patterns of different crops. The computed metrics encompass a wide range of statistical measures, changes over time, and distribution-based metrics, offering a detailed analysis of the temporal patterns in the study area.

\renewcommand{\arraystretch}{1.5} % Increase the row height

\begin{longtable}{|p{4cm}|p{5cm}|p{6cm}|}
\hline
\textbf{Statistic} & \textbf{Description} & \textbf{Equation} \\
\hline
\endhead
Absolute energy &  sum over the squared values & $E = \sum_{i=1}^n x_i^2$ \\
Absolute Sum of Changes  & sum over the absolute value of consecutive changes in the series  & $ \sum_{i=1}^{n-1} \mid x_{i+1}- x_i \mid $ \\
Autocorrelation (1 \& 2 month lag) & Correlation between the time series and its lagged values & $\frac{1}{(n-l)\sigma^{2}} \sum_{t=1}^{n-l}(X_{t}-\mu )(X_{t+l}-\mu)$\\
Count Above Mean & Number of values above the mean & $N_{\text{above}} = \sum_{i=1}^n (x_i > \bar{x})$ \\
Count Below Mean & Number of values below the mean & $N_{\text{below}} = \sum_{i=1}^n (x_i < \bar{x})$ \\Day of Year of Maximum Value & Day of the year when the maximum value occurs in series & --- \\
Day of Year of Minimum Value & Day of the year when the minimum value occurs in series & --- \\
Kurtosis & Measure of the tailedness of the time series distribution & $G_2 = \frac{\mu_4}{\sigma^4} - 3$ \\
Linear Time Trend & Linear trend coefficient estimated over the entire time series & $b = \frac{\sum_{i=1}^n (x_i - \bar{x})(t_i - \bar{t})}{\sum_{i=1}^n (x_i - \bar{x})^2}$ \\
Longest Strike Above Mean & Longest consecutive sequence of values above the mean & --- \\
Longest Strike Below Mean & Longest consecutive sequence of values below the mean & --- \\
Maximum & Maximum value of the time series & $x_{\text{max}}$ \\
Mean & Mean value of the time series & $\bar{x} = \frac{1}{n}\sum_{i=1}^n x_i$ \\
Mean Absolute Change & Mean of absolute differences between consecutive values & $\frac{1}{n-1} \sum_{i=1}^{n-1} | x_{i+1} - x_{i}|$ \\
Mean Change & Mean of the differences between consecutive values & $ \frac{1}{n-1} \sum_{i=1}^{n-1}  x_{i+1} - x_{i} $ \\
Mean Second Derivative Central & measure of acceleration of changes in a time series data & $\frac{1}{2(n-2)} \sum_{i=1}^{n-1}  \frac{1}{2} (x_{i+2} - 2 \cdot x_{i+1} + x_i)
$ \\
Median & Median value of the time series & $\tilde{x}$ \\
Minimum & Minimum value of the time series & $x_{\text{min}}$ \\
Quantile (q = 0.05, 0.95) & Values representing the specified quantiles (5th and 95th percentiles) & $Q_{0.05}, Q_{0.95}$ \\
Ratio Beyond r Sigma (r=1,2,3) & Proportion of values beyond r standard deviations from the mean & $P_r = \frac{1}{n}\sum_{i=1}^{n} (|x_i - \bar{x}| > r\sigma_{x})$ \\
Skewness & Measure of the asymmetry of the time series distribution & $\frac{n}{(n-1)(n-2)} \sum \left(\frac{X_i - \overline{X}}{s}\right)^3$ \\
Standard Deviation & Standard deviation of the time series & $  \sqrt{\frac{1}{N}\sum_{i=1}^{n} (x_i - \bar{x})^2}$ \\
Sum Values & Sum of all values in the time series & $S = \sum_{i=1}^{n} x_i$ \\
Symmetry Looking & Measures the similarity of the time series when flipped horizontally & $| x_{\text{mean}}-x_{\text{median}} | < r * (x_{\text{max}} - x_{\text{min}} ) $ \\
Time Series Complexity (CID CE) & measure of number of peaks and valleys & $\sqrt{ \sum_{i=1}^{n-1} ( x_{i} - x_{i-1})^2 }$\\
Variance & Variance of the time series & $\sigma^2 = \frac{1}{N}\sum_{i=1}^{n} (x_i - \bar{x})^2$ \\
Variance Larger than Standard Deviation & check if variance is larger than standard deviation & $\sigma^2 > 1$ \\
\hline
\end{longtable}
 
 

```{=latex}
\newpage
```

# References
