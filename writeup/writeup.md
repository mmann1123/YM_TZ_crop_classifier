---
title: "Lite Learning: Efficient Crop Classification in Tanzania Using Feature Extraction with Machine Learning & Crowd Sourcing"
author:
- name: Michael L. Mann
  affiliation: The George Washington University, Washington DC 20052
  thanks: Corresponding author. Email mmann1123@gmail.com
- name: Lisa Colson
  affiliation: USDA Foreign Agricultural Service, Washington DC 20250
- name: Rory Nealon
  affiliation: USAID GeoCenter, Washington DC 20523
- name: Ryan Engstrom
  affiliation: The George Washington University, Washington DC 20052
- name: Stellamaris Nakacwa
  affiliation: YouthMappers, Texas Tech University, Lubbock TX 79409
documentclass: IEEEtran
classoption: [journal,twocolumn]
header-includes:
  - \usepackage[margin=1in]{geometry}
  - \usepackage{pdflscape}
  - \usepackage{longtable}
  - \usepackage{fancyhdr}
  - \usepackage{float}
  - \usepackage{graphicx}
  - \usepackage{amsmath}
  - \usepackage{amsfonts}
  - \usepackage{lineno}
  - \usepackage{array}
  - \usepackage{booktabs}
  - \usepackage{caption}
  - |
    \pagestyle{fancy}
    \fancyhf{}
    \rfoot{\thepage}
    \fancypagestyle{plain}{
      \fancyhf{}
      \rfoot{\thepage}
    }

abstract: |
  This study introduces a novel approach to traditional machine learning methodology for crop type classification in Tanzania, by integrating crowdsourced data with time-series features extracted from Sentinel-2 satellite imagery. Leveraging the YouthMappers network, we collected ground validation data on various crops, including challenging types such as cassava, millet, sunflower, sorghum, and cotton across a range of agricultural areas. Traditional machine learning algorithms, augmented with carefully engineered time-series features, were employed to map the different crop classes. Our approach achieved high classification accuracy, evidenced by a Cohen's κ (kappa) of 0.82 and an F1-micro score of 0.85. The model often matched or outperformed broadly used land cover models which simply classify 'agriculture' without specifying crop types. By interpreting feature importance using SHAP values, we identified key time-series features driving the model's performance, enhancing both interpretability and reliability. To address the scarcity of multi-class crop type datasets, we publicly release the crowdsourced crop-type observations and the extracted Sentinel-2 time-series features. Our findings demonstrate that traditional machine learning techniques, combined with computationally efficient feature extraction methods, offer a practical and effective “lite learning” approach for mapping crop types in data-scarce environments. This methodology facilitates accurate crop type classification using a low-cost, resource-limited approach that contributes valuable insights for sustainable agricultural practices and informed policy-making, ultimately impacting food security and land management in resource-limited contexts, such as sub-Saharan Africa.
--- 

<!-- 


cd writeup
pandoc writeup.md    --template=mytemplate.tex --from markdown+raw_tex --to latex --bibliography=refs.bib --citeproc -o final_submission/output_JSTARS-2025-00807.tex

pandoc supplement.md --template=mytemplate.tex --from markdown+raw_tex --to latex --bibliography=refs.bib --citeproc -o final_submission/supplement.tex


 
-->
<!-- 
Look at https://mpastell.com/pweave/docs.html -->

# Introduction

## Background and Context

<!-- |
├── Background and Context
|    ├── Overview of Remote Sensing Technology
|    └── Applications in [specific field or topic] -->
The free access to remotely sensed data, such as imagery from satellites (e.g., Sentinel-2, Landsat), has allowed for crop type classification in developing countries. By leveraging the power of advanced imaging technologies combined with machine learning algorithms, researchers and practitioners can now identify and map different crop types over large geographic areas at no or low cost [@hersh2021open]. This has the potential to improve food security, land use planning, and agricultural policy in regions where ground-based data collection is limited or non-existent [@begue2018remote; @li2023development; @ibrahim2021mapping].

In recent years, machine learning approaches have emerged as powerful tools for crop type classification using remotely sensed data. Specifically, methods based on machine learning algorithms have gained recognition for their effectiveness in matching valuable spectral information from satellite imagery to observations of crop type for particular locations. Machine learning algorithms, including decision trees, random forests, support vector machines (SVM), and k-nearest neighbors (KNN), have been successfully used to  classify imagery into unique agricultural types [@ibrahim2021mapping; @begue2018remote; @delince2017handbook]. These algorithms leverage the rich spectral information captured by satellite sensors, allowing them to identify distinctive patterns associated with different crop types. By training on large labeled datasets where ground-validation information on crop types is linked to corresponding image pixels, these models can effectively learn the relationships between the spectral characteristics of crops and their respective classes [@begue2018remote].

The strength of traditional machine learning approaches lies in their ability to exploit both the spectral and time-series patterns within the remotely sensed data. Traditional machine learning approaches offer advantages in terms of interpretability and computational efficiency compared to deep learning architectures. They provide insight into the decision-making process and can be more readily understood and explained by domain experts. Additionally, these methods are generally less computationally demanding and require less training data, making them suitable for applications with limited computational resources [@hohl2024recent; @rs13132591; @agriculture13050965; @LI2023103345; @MA2019166].


<!-- ├── Problem Statement
|    ├── Current Challenges
|    └── Research Need -->

Traditional machine learning algorithms require the extraction of variables (e.g., max EVI, mean blue band) that can help distinguish different plant or crop types [@begue2018remote]. The development of salient time-series features to capture phenological differences between locations from remotely sensed images remains a challenge. These features are typically derived from the spectral bands (e.g., red edge, NIR) of the satellite imagery or indexes, such as the enhanced vegetation index (EVI), and basic time series statistics (e.g., mean, max, minimum, slope) for the growing season [@morton2006cropland]. Meanwhile a broader set of time series statistics from bands or indexes may be more relevant for a number of applications. For instance the skewness of EVI might help distinguish crops that green-up earlier vs later in the season, measures of the numbers of peaks in EVI might help differentiate intercropping or multiple plantings in a season [@begue2018remote]. However, the selection and extraction of these features can be time-consuming and labor-intensive, requiring domain expertise and manual intervention.

In contrast, deep learning methods have dominated the most recent literature [@agriculture13050965; @hohl2024recent]. These methods include both recurrent neural networks (RNN) and convolutional neural networks (CNN). Recurrent Neural Networks (RNNs) are a class of neural networks that are particularly powerful for modeling sequential data such as time series, speech, text, and audio. The fundamental feature of RNNs is their ability to maintain a 'memory' of previous inputs by using their internal state (hidden layers), which allows them to exhibit dynamic temporal behavior. RNNs and their variants allow the integration of time-series imagery, substantially improving crop type classification outcomes especially in data-rich environments [@agriculture13050965; @camps2021deep]. Deep learning approaches however typically require much larger sets of training data, may be more prone to overfitting especially with small sample sizes, have substantial limitations to interpretability, and require expensive compute [@hohl2024recent; @rs13132591; @agriculture13050965; @LI2023103345; @MA2019166]. Indeed, recent work demonstrates that deep architectures achieving near-perfect in-domain accuracy can collapse when transferred across regions or sensors, while lightweight, feature-based alternatives transfer far more reliably [@pankajakshan2025; @rustowicz2019; @mai2025]. Although recent efforts have closed the gap e.g., [@tseng2021cropharvest], the lack of readily available and reliable ground truth data or benchmark datasets for training, as discussed earlier, may limit the applicability of deep learning for a variety of tasks including crop classification and make researchers more reliant of less reliable techniques like transfer learning or zero-shot or low-shot methods [@owusu2024towards; @LI2023103345; @MA2019166]. Recent papers applying deep learning methods to agriculture in a developing world context have achieved strong results. For instance, @buttar2024satellite applied U-Net (a fully convolutional encoder-decoder semantic segmentation architecture) to a crop type dataset in Kenya, achieving an F1 score of 73.6%. @liu2024enhanced achieved an F1 score of 87.7% utilizing a novel deep learning architecture that combines a three-dimensional convolutional neural network (3D-CNN) with a variant
of convolutional recurrent neural networks (ConvRNN) on Sentinel-1 synthetic aperture radar (SAR) features and Sentinel-2 surface reflectance. In controlled agricultural settings, like Bradford Research Center in Missouri, combining SAR and optical data for similar models are able to achieve much higher overall accuracy levels of 94.1% for 3D U-Net, 84.7% for 2D U-Net, and 64% for SegNet [@ADRIAN2021215]. Moreover, training data for extreme events, like crop losses, disease, and lodging are largely non-existent.  Interpretability is also a salient weakness as interpretation of models allows us to gain scientific insight and assess trustworthiness and fairness in so far as outputs affect policy decisions.

Recent advances in lightweight transfer learning architectures, such as MobileNetV2 and EfficientNet, have shown promising results in agricultural applications, particularly for RGB image classification on mobile devices and smart farming systems [@sandler2018mobilenetv2; @tan2019efficientnet]. These architectures offer reduced computational requirements compared to larger deep learning models while maintaining competitive performance. However, transfer learning approaches face domain adaptation challenges when applied to multispectral satellite imagery, as pre-trained weights are typically derived from natural RGB images rather than remote sensing data. Our feature-based approach offers complementary benefits for satellite-based crop classification: improved interpretability through SHAP values that provide agronomically meaningful explanations, accessibility on standard hardware without GPU requirements, and transparency in the feature engineering process that allows domain experts to incorporate their knowledge directly.  

An alternative approach turns back the clock on deep learning approaches. For instance CNN classifiers, through the exertion of tremendous effort of GPUs, can apply and learn from thousands of filters or convolutions that help detect distinct features like edges, textures or patterns. It is however possible to apply a more limited yet salient set of filters like Fourier Transforms, Differential Morphological Profiles [@pesaresi2001new], Line Support Regions or Structural Feature Sets [@huang2007classification] amongst others, to images and then use these as features in more traditional machine learning approaches [@graesser2012image; @owusu2024towards; @engstrom2022poverty; @chao2021evaluating; @urbansci7040116]. This approach may be particularly useful in data-scarce environments, requiring less training data and potentially offering more efficient results in low-information settings. The same approach has been taken for time series analysis, where instead of learning patterns through a RNNs memory, we can apply a more limited but potentially salient series of time series filters. Measures of trends, descriptions of distributions, or measures of change and complexity might adequately describe time series properties for regression and classification tasks [@christ2018time; @yang2021anomaly]. This time series filter approach, developed for this paper, can also be applied on a pixel-by-pixel basis to satellite image bands or index values [@xr_fresh_2021].

Field-collected data provides the necessary validation and calibration for remote sensing-based models. It serves as the benchmark against which the model's predictions are evaluated and refined. Ground validation data collected through field visits, observation, and interactions with local farmers offers essential insights into the specific crop types present in the study area. Validating and training models with accurate ground reference information allows for the spectral patterns captured by remote sensing data to be correctly associated with the corresponding crop. By combining the spectral information from satellite imagery with ground validation data, researchers can develop robust models that effectively differentiate between different crop types based on their unique spectral signatures and temporal patterns.

The collection of field observations and ground validation data is a critical input for the development of models to classify crop types [@delince2017handbook; @MA2019166]. However, obtaining accurate and timely ground validation data can be challenging in developing countries due to limited resources, infrastructure, and local capacity [@delince2017handbook; @begue2018remote]. In many cases, researchers rely on crowdsourced data from volunteers or citizen scientists to supplement or validate ground truth data collected through traditional methods. Projects like [@tseng2021cropharvest] point to the paucity of multi-class crop type datasets globally. This is a significant gap in the field of crop type classification, as the availability of high-quality training data is essential for the development of accurate and reliable machine learning models [@rs13132591].

<!-- |
├── Objective of the Study
|    ├── Research Goals
|    └── Scope -->

In this study we aim to address two critical challenges in the field of crop type classification: the lack of in-season multi-class crop type datasets, and the need for new methods to obtain high accuracy crop type predictions from limited amounts of training data.

<!-- ├── Significance and Innovations
|    ├── Contributions to the Field
|    └── Innovations in Methodology or Technology -->
We propose a novel approach that combines crowdsourced data with a new automated approach to extracting time-series features from satellite imagery. We apply this new approach to classify crop types in Northern Tanzania. By leveraging the power of crowdsourcing and remote sensing technologies, we aim to develop a robust and scalable solution for crop type classification that can be adapted to other regions and contexts with minimal or no cost. To directly address the scarcity of multi-class crop type datasets and to support reuse and benchmarking, we publicly release the resulting dataset of crowdsourced crop-type observations and the corresponding Sentinel-2 time-series features [@mann2025tanzania].

<!-- ├── Literature Review
|    ├── Previous Work
|    └── Distinction from Prior Work -->

<!-- └── Outline of the Paper
    └── Structure of Subsequent Sections -->

# Data

Data for this study were collected from multiple sources, including satellite imagery, and crowdsourced ground truth observations. The section below describes the input data and methods used throughout the paper.

## Study Area

The study was conducted in 50 wards within three major districts of Arusha, Dodoma and Mwanza in Tanzania as seen in Figure \ref{fig:study_area}. Tanzania, a country in East Africa, is known for its diverse agricultural landscape.  The region is characterized by a mix of smallholder farms, commercial plantations, and natural vegetation, making it an ideal yet challenging location for studying crop type classification. Our choice of these three districts was driven by the distinct variation in the major crop types that possibly dominated in each district, among oil seeds, grains and commercial crops such as cotton.

\begin{figure}[H]
   \centering   \includegraphics[width=0.8\linewidth]{figures/tz_ym_crop_target_provinces.png} 
   \caption{Study area map \newline Districts in Northern Tanzania where field visits were carried out (green)}
   \label{fig:study_area} %can refer to in text with \ref{fig:study_area}
\end{figure}

## Crowdsourced Data Collection

Crop type data collection was designed and executed by YouthMappers through a crowdsourced GIS approach. Data collection was designed in 3 steps where: 1) development and training of all intended student participants. 2) Data collection using KoboToolbox hosting a well developed data model. The exercise lasted 14 days with 7 days of iterative pilot testing on different farms, crops and landscapes. Finally the last step, 3) was the data review and cleaning phase to generate a sample for training as seen in Figure \ref{fig:methodology_flowchart}


\begin{figure}[H]
   \centering   \includegraphics[width=0.8\linewidth]{figures/methodology_flowchart.png} 
   \caption{Data collection methodology flowchart}
   \label{fig:methodology_flowchart} %can refer to in text with \ref{fig:methodology_flowchart}
\end{figure}

Additional training data for non-agricultural sites was collected utilizing high resolution imagery from Google Earth. These data were used to supplement the crowdsourced data and improve the model's ability to distinguish between crops and more common land cover types like forests, urban areas, and water.  The final cleaned dataset includes over 1,400 crop type observations of rice, maize, cassava, sunflower, sorghum, cotton, and millet; plus 386 other observations of land cover classes including water, tidal areas, forest, shrub and urban. This dataset, together with the extracted Sentinel-2 time-series features, is publicly available on Zenodo [@mann2025tanzania].  

## Satellite Imagery

Satellite imagery was obtained from the Sentinel-2 satellite constellation, which provides high-resolution multispectral data at 10 m spatial resolution. The imagery was acquired over the study area between January and August of 2023 during the growing season, capturing the spectral characteristics of different crop types and coinciding with field data collection. The Sentinel-2 L2 harmonized reflectance data were pre-processed to remove noise and atmospheric effects, ensuring that the spectral information was accurate and reliable for classification purposes [@begue2018remote].

In our study, cloud and cloud shadow contamination was mitigated using the 's2cloudless' machine learning model on the Google Earth Engine platform. Cloudy pixels were identified using a cloud probability mask, with pixels having a probability above 50% classified as clouds. To detect cloud shadows, we used the Near-Infrared (NIR) spectrum to flag dark pixels not identified as water as potential shadow pixels. The projected shadows from the clouds were identified using a directional distance transform based on the solar azimuth angle from the image metadata. A combined cloud and shadow mask was refined through morphological dilation, creating a buffer zone to ensure comprehensive coverage. This mask was applied to the Sentinel-2 surface reflectance data to exclude all pixels identified as clouds or shadows, enhancing the reliability of the dataset for environmental analysis.

Monthly composites were collected for January through August of 2023 for the bands B2 Blue (458-523 nm), B6 Vegetation Red Edge (733-738 nm), B8 Near Infrared (785-899 nm), B11 Short-Wave Infrared 1 (SWIR-1) (1565-1655 nm), and B12 Short-Wave Infrared 2 (SWIR-2) (2100-2280 nm). We also calculate the Enhanced Vegetation Index (EVI) and hue, the color spectrum value [@GoogleHSV]. This computed hue value provides the basic color as perceived in the color wheel, from red, through green, blue, and back to red for each pixel. Due to the high prevalence of clouds in the region, linear interpolation was used to fill in missing data in the time series using `xr_fresh` [@xr_fresh_2021]. These bands were selected based on their relevance to crop type classification and their ability to capture the unique spectral signatures of different crops. The monthly composites were used to generate time series features for each pixel in the study area, providing valuable information on the temporal dynamics of crop growth and development.

# Methods
The following section describes the methods used for data collection, feature extraction, model training, and evaluation.

## Data Collection

To ensure the success of our project, we focused heavily on the design of our data collection methods. These methods were carefully integrated, taking into account: the crop calendar, information on the different stages of crop development, the distances between crop fields, the tools used, and data quality assurance.

### Field Data

Young crops exhibit substantial differences compared to mature crops in terms of color, density, and phenological development. Variations in the crop cycle across different fields could lead to heteroscedasticity in the spectral reflectance measurements used for machine learning (ML) training, thereby affecting the precision and accuracy of the model. We targeted the period of April through May to capture crops late in the growing season but before harvest, as seen in the crop calendar in Figure \ref{fig:crop_cal} below.

\begin{figure}[H]
   \centering   \includegraphics[width=0.8\linewidth]{figures/plant_ghant.png}
   \caption{Tanzania Crop Calendar }
   \label{fig:crop_cal} %can refer to in text with \ref{fig:crop_cal}
\end{figure} Source: [@cc_tanzania]

USDA’s Foreign Agricultural Service compiles information on planting and harvest windows for grain, oilseed, and cotton crops as an important tool to support crop condition assessments with satellite imagery. Tanzania’s crop planting seasons are shaped by its bimodal and uni-modal rainfall patterns, which vary by region. In the north and northeast, bimodal areas experience the short rains (Vuli) from late-October to mid-January, during which crops like maize, beans, and vegetables are planted in October and November, and the long rains (Masika) from March to May, supporting crops like maize, rice, sorghum, and cassava, typically planted in February and March. In the central, southern, and western regions with uni-modal rainfall, there is a single rainy season from November to April, when crops such as cotton, maize, millet, rice, and sunflower are planted in November and December. This diversity in rainfall patterns allows for a wide variety of crops suited to the local climate and seasonal conditions.

Data collection took place between late April and May 2023 (Figure \ref{fig:crop_cal}) to align with the mid-season growth stage for most target crops. YouthMappers were trained to focus on crops known to be present in each region and at appropriate phenological stages for spectral discrimination. Prior to fieldwork, extensive training sessions covered the key criteria for selecting suitable field sites to ensure high-quality training data for the machine learning models.

Field size represented a critical consideration, as features below the spatial resolution of Sentinel-2 imagery would not be adequately captured. YouthMappers were therefore instructed to identify only fields measuring 30 m or greater across, ensuring that each field would encompass multiple pixels in the satellite imagery. Beyond size, field composition played an equally important role in data quality. Agriculture in the study region often includes heterogeneous land cover with tree cover, power lines, buildings, and other obstructions that can contaminate the spectral signature of crops. To minimize this spectral mixing, YouthMappers were trained to prioritize clear, open fields planted with a single crop type, using photographic examples to illustrate ideal versus problematic field characteristics.

Spatial distribution of sample sites was carefully considered to balance logistical constraints with statistical independence. A minimum separation distance of one kilometer between sampling locations was established as a compromise between the time and cost of travel and the need to reduce spatial autocorrelation in the training data. While YouthMappers were permitted to sample adjacent fields when different crop types were present, they were otherwise encouraged to maintain this spacing by traveling to more distant locations.

The timing of data collection was primarily driven by crop phenology and field condition. Mid to late reproductive stages offer maximum canopy cover and the most distinctive spectral signatures for crop type discrimination. Although most sampled fields exhibited the desired phenological characteristics, drought conditions in 2023 affected crop health across the region, with some fields showing signs of stress or early harvest. To ensure robust model training, YouthMappers were instructed to prioritize mature, healthy fields with lush green canopies whenever possible. Through this comprehensive training process, field teams developed the expertise needed to consistently identify sites well-suited for generating high-quality in-situ training data for satellite-based crop classification.

The data collection was managed through KoboCollect, hosted on the KoboToolBox infrastructure, which provided an effective platform for gathering and organizing data. This approach enabled a collection of the desired volume of data points necessary for model training and evaluation, as summarized in Table \ref{tab:data_n}.


\begin{table}[h!]
\centering
\small
\begin{tabular}{@{}lcp{4.5cm}@{}}
\toprule
\textbf{Region} & \textbf{Points} & \textbf{Primary Crops} \\
\midrule
Arusha & 300 & Maize, Rice, Sorghum, Millet \\
\midrule
Mwanza & 1000 & Maize, Cotton, Rice, Peanut \\
\midrule
Dodoma & 800 & Sorghum, Maize, Millet, Sunflower, Peanut, Cotton \\
\bottomrule
\end{tabular}
\caption{Collection Targets and Primary Crops by Region in Tanzania}
\label{tab:data_n}
\end{table}

### Field Data Cleaning and Validation

The initial dataset comprised 1,720 observations collected by YouthMappers across the three districts. A thorough data cleaning process was undertaken to ensure the quality and reliability of the dataset for model training and evaluation. This process involved several steps. First, duplicate entries were identified and removed to prevent redundancy and potential bias in the dataset. Second, observations with missing or incomplete data were addressed; depending on the extent of missing information, these entries were either corrected using auxiliary data sources or excluded from the dataset. Third, each observation was visually inspected using in-situ photos taken by students at each site, which helped verify the accuracy of the recorded crop types and field conditions. Fourth, crop type labels were standardized to ensure consistency across the dataset by correcting typographical errors and unifying different naming conventions for the same crop. Finally, the geographic coordinates of each observation were validated to ensure they fell within the expected study area and corresponded to an observable field from satellite imagery. After completing the cleaning process, the final dataset consisted of over 1,400 crop type entries, providing a robust foundation for training and evaluating the machine learning models used in this study.

The photo-based validation process served as a critical quality control mechanism. Each field photograph was reviewed by researchers familiar with regional crop characteristics to confirm or correct the crop type labels recorded during field visits. This remote verification allowed identification of misclassified observations where field conditions or crop similarity led to labeling errors. Approximately 320 observations were removed during the cleaning process due to factors including unclear photographs, label inconsistencies, coordinate errors, or fields that could not be confidently verified. While this photo-based approach provided valuable validation, it represents a limitation of our methodology: independent post-classification field visits to verify model predictions were not conducted. Future studies would benefit from systematic ground truthing campaigns following model application to assess real-world classification accuracy and identify systematic errors in specific crop types or geographic areas.

## Analytical Methods

After data collection and cleaning, we employed a series of analytical methods to extract relevant features from the satellite imagery, train machine learning models, and evaluate their performance. The overall workflow is illustrated in Figure \ref{fig:analyt_flow}.

\begin{figure}
   \centering   \includegraphics[width=0.8\linewidth]{figures/analytical_methods_flowchart.png}
   \caption{Analytical Methods Workflow }
   \label{fig:analyt_flow} %can refer to in text with \ref{fig:analyt_flow}
\end{figure} 


### Time Series Features

Time series features capture the temporal dynamics of crop growth and development, providing valuable information on the phenological patterns of different crops. We leverage the time series nature of the satellite imagery to extract relevant features for crop type classification for the 2023 growing season.

In this study, we utilized the `xr_fresh` toolkit to compute detailed time-series statistics for various spectral bands, facilitating comprehensive pixel-by-pixel temporal analysis [@xr_fresh_2021]. The `xr_fresh` framework is specifically designed to extract a wide array of statistical measures from time-series data, which are essential for understanding temporal dynamics in remote sensing datasets.

The metrics computed by `xr_fresh` in this study include basic statistical descriptors, changes over time, and distribution-based metrics, applied to each pixel's time series for selected spectral bands (B2, B6, B11, B12, EVI, and hue). The list of computed time-series statistics encompasses:

- **Energy Measures**: Absolute energy which provides a sum of squares of the values.
- **Change Metrics**: Absolute sum of changes to quantify overall variability, mean absolute change, and mean change.
- **Autocorrelation**: Calculated for three lags (1, 2, and 3) to assess the serial dependence at different time intervals.
- **Count Metrics**: Count above and below mean, capturing the frequency of high and low values relative to the average.
- **Extreme Values**: Day of the year for maximum and minimum values, providing insight into seasonal patterns.
- **Distribution Characteristics**: Kurtosis, skewness, and quantiles (5th and 95th percentiles) to describe the shape and spread of the distribution.
- **Variability Metrics**: Standard deviation, variance, and whether variance is larger than standard deviation to evaluate the dispersion of values.
- **Complexity and Trend Analysis**: Time series complexity and symmetry looking, adding depth to the analysis of temporal patterns.

For a full list of the time series features extracted in this study and their descriptions, please refer to the Supplementary Material (Table S1).

The integration of `xr_fresh` into our analytical workflow allowed for an automated and robust analysis of temporal patterns across the study area. By leveraging this toolkit, we could efficiently process large datasets, ensuring that each pixel's temporal dynamics were comprehensively characterized, which is critical for accurate environmental monitoring and change detection.

### Data Extraction

To partially account for variation in field size, we extracted pixels based on a buffer around field point locations. This allows us to account for the fact that fields likely represent groups of adjacent pixels. Small fields were buffered by only 5 m, medium fields by 10 m and large fields by 20 m. This approach allowed us to capture the time series features from the surrounding area, providing a more comprehensive representation of the field's characteristics. The use of larger buffers was explored but found to decrease model performance as fields tended to be heterogeneous - for instance containing patches of trees. To account for this in our modeling, we treat observations from the same field as a "group" in our cross-validation scheme - as described below.

### Model Selection

We employed `Optuna`, an optimization framework, to conduct systematic model selection and hyperparameter tuning [@optuna_2019]. Our methodology involved defining a study where each trial proposed a set of model parameters aimed at optimizing classification performance. We evaluated multiple classifiers, including LightGBM, Support Vector Classification (SVC), and RandomForest, testing various configurations to identify the optimal approach for crop classification.

Prior to model training, we preprocessed the extracted time-series features from satellite imagery using standard scaling (centering and scaling) from the `scikit-learn` library to normalize the data [@scikit-learn]. We then applied a variance threshold method to reduce dimensionality by excluding features with low variance, thereby improving computational efficiency and model interpretability.
The final model selection was based on maximizing Cohen's κ (kappa) across all cross-validation folds, ensuring that the chosen model and its parameters provided the best possible performance for classifying crop types in our dataset.

### Performance Evaluation

To assess model performance, we implemented stratified group k-fold cross-validation with three splits. This approach ensured that samples from the same field remained together within either the training or validation set, preventing data leakage that could occur if observations from the same field were split across folds. We utilized Cohen's κ as our primary evaluation metric due to its suitability for assessing classifier performance on imbalanced datasets. This metric accounts for agreement occurring by chance, providing a more robust measure of classification accuracy than simple overall accuracy when class distributions are unequal.

### Computational Efficiency
 A primary advantage of our "lite learning" approach is its computational efficiency, particularly when processing high-dimensional satellite time series. LightGBM optimizes the training process using a histogram-based approach, with a time complexity of:
 
 $$T_{LGBM} = O(n \cdot m) + O(b \cdot k \cdot d)$$
 
 Here, the first term represents the one-time cost of binning $n$ samples across $m$ features (where $m = bands \times timestamps$). The second term captures tree construction across $b$ bins, $k$ trees, and maximum depth $d$. Because split-finding operates on compressed bins rather than raw data points, the expensive $O(n \cdot m)$ operation occurs only once. In our implementation, hyperparameter optimization (100 Optuna trials, 3-fold CV) concluded within 10–15 minutes on a standard laptop (Intel i7, 16GB RAM) without GPU acceleration. 
 
 This efficiency stands in stark contrast to deep learning architectures: recurrent models (e.g., LSTMs) and spatial convolutional models (2D and 3D U-Nets) carry substantially higher training-time and memory costs, scaling far less favorably with sequence length, hidden dimensionality, and convolution kernel size. We detail their computational complexity in the Supplementary Material (Sec. S-II).
 
 For operational deployment, LightGBM inference scales at $O(n \cdot k \cdot d)$, allowing for rapid pixel-wise mapping. By relying on CPU-only computation, this approach is not only viable for resource-constrained environments but also significantly reduces the carbon footprint of large-scale agricultural monitoring.

### Interpretation and Feature Selection

To interpret the contributions of individual features to the model predictions, we employed SHapley Additive exPlanations (SHAP) [@shaps_2017]. This approach, based on game theory, quantifies the impact of each feature on the prediction outcome, providing insights into which features are most influential in determining land cover types.

Prior to SHAP analysis, we applied a variance threshold filter to remove features with low discriminative power. Features with variance below 0.5 were excluded, as low-variance features provide minimal information for distinguishing between classes. This preprocessing step reduced the initial feature set substantially while preserving features with meaningful variation across the training samples.

In our feature selection process, we incorporate both the mean and maximum SHAP values to comprehensively assess the influence of features on model predictions. The mean of the absolute SHAP values across all samples, provides a measure of the average impact of each feature, highlighting its overall importance across the dataset. This approach might underrepresent the significance of features under rare conditions. To address this, we also consider the maximum absolute SHAP values. Sorting features by their maximum absolute SHAP values allows us to identify those that have significant, albeit possibly infrequent, effects on individual predictions. This method ensures that features crucial for particular scenarios are not overlooked, thus offering a more nuanced understanding of feature importance that balances general influence with critical, situation-specific impacts.

Feature selection then is the union of the top 30 time series features found with both the mean and maximum SHAP values, resulting in 32 total features. The choice of 30 features represents a balance between model parsimony and classification performance; preliminary experiments indicated diminishing returns beyond this threshold while smaller feature sets showed reduced accuracy. This approach ensures that the selected features are both consistently influential across the dataset and capable of exerting substantial impacts under specific conditions, providing a comprehensive set of features for model training and evaluation.

For the final LightGBM model, Optuna optimized hyperparameters including the number of boosting rounds (up to 10,000 with early stopping after 200 rounds without improvement), learning rate, maximum tree depth, and regularization parameters. The early stopping mechanism prevents overfitting by halting training when validation performance plateaus, while the stratified group k-fold cross-validation ensures robust parameter selection across different data partitions (fields).

# Results & Discussion

## Crowdsourced Data

To address the substantial gap in available crop type datasets, particularly in developing regions, this study harnessed the power of crowdsourced data to enhance the robustness and applicability of our machine learning models. Crowdsourced data collection, an innovative approach in the agricultural domain, involves gathering data from a large number of volunteers or citizen scientists, who provide valuable ground truth information. This method has proven especially useful in areas where traditional data collection methods are challenging due to logistical, financial, or infrastructural constraints.

By leveraging  the YouthMappers student organization, with over 420 chapters in 80 countries, we were able to collect a large dataset of crop type observations in Tanzania. Participating YouthMappers chapters included: the Institute of Rural Development Planning - Dodoma, Institute of Rural Development Planning - Mwanza, University of Dodoma, the Nelson Mandela African Institution of Science and Technology, and the Institute of Accountancy Arusha. Moreover this exercise provided an important opportunity for students to gain practical experience in data collection, analysis, and interpretation, contributing to their professional development and capacity building in the geospatial domain.

### Challenges and Lessons Learned

There were a number of challenges involved with planning, and implementing a large-scale field operation. One of the primary challenges encountered was the variability in crop cycles across different fields and crop identification more generally. This was particularly true in Arusha, where fields were found in almost every stage of crop development and some fields were visited before the reproductive stages, as the drought delayed planting. In other regions, some crops had already been harvested. This discrepancy resulted in incomplete datasets, as certain crop types were missing or not easily accounted for. The absence of these crops in certain areas impacted our modeling efforts by reducing the representativeness of the training data. Second, although the YouthMappers teams did a commendable job, crop identification is challenging for non-agricultural experts. This task was even more challenging given the heterogeneity of local planting practices and the similarity of early stage growth between, for instance, crops like maize and sorghum. To mitigate this issue YouthMappers teams took detailed photos of each field. These images provided us the ability to verify crop types remotely before training the model. While extremely useful, the collection of more detailed single plant images could have helped us minimize removal of some observations. Third, the site selection depended on many non-crop related factors including where the training could be hosted and time constraints based on YouthMappers student’s academic calendars. This led to changes in which target crops were selected. Fourth, travel was finally approved during a drought year. This is helpful for transportation during field work, yet poses a challenge as more fields can be abandoned, harvested early, or otherwise found in a poor condition. To mitigate many of these issues, future data collection efforts should allow for more flexibility in the timing in data collection and ensure coverage that reflects local crop cycles.

## Land Cover and Crop Type

The distribution of primary land cover types within the training dataset used for the model are represented in Figure \ref{fig:lc_percentages}. The dataset consists of a diverse range of land cover types, each contributing differently to the total number of observations. Maize is the most prevalent land cover type, accounting for the highest percentage of the observations, followed by rice and sunflower. This is indicative of the agricultural dominance of maize in the region being studied. Less common land covers such as millet, sorghum, and urban areas represent intermediate percentages, reflecting the  heterogeneous landscape that includes both agricultural and urbanized zones. Peanuts, soybeans and okra are among the least represented land cover types in the dataset, highlighting the challenges associated with collecting sufficient training data for these categories, but also the small scale of production for these crops in the region. This figure also includes land cover types such as water, forest, shrub, and tidal areas, which are essential for providing context to the landscape but were not the focus of this study. The varied distribution of land cover types underscores the complexity of the classification task and the need for robust modeling techniques to accurately capture this diversity.

\begin{figure}[H]
   \centering   \includegraphics[width=0.8\linewidth]{figures/primary_land_cover.png} % Adjust the path and options
   \caption{Land Cover by Percentage of Observations}
   \label{fig:lc_percentages} %can refer to in text with \ref{fig:lc_percentages}
\end{figure}

## Feature Importance

The interpretation of model behavior using SHAP values has allowed for a deeper understanding of how different spectral features impact the model's predictions, which is critical for refining the feature selection process. By analyzing both the mean and maximum SHAP values, we were able to prioritize features based on their overall impact as well as their critical contributions to specific model decisions.

In the two summary plots below, we display the SHAP values for each feature, to identify how much impact each feature has on the model output for pixels in the validation dataset. Features are sorted by the sum of the SHAP value across all samples. The figure's bar length represents the mean contributions to explaining each predicted land class value - with different land classes represented with different colors (hues). This visualization provides a comprehensive overview of the feature importance, highlighting the key predictors that drive the model's predictions. For example, features that are highly influential for "maize" may not be as impactful for "rice" or "sorghum", reflecting the unique spectral signatures of these crops.

### Mean SHAP Values

In Figure \ref{fig:mean_shaps}, the mean SHAP values provide insights into the average impact of each feature across all predictions. This analysis highlights the features that consistently influence the model's output across various scenarios. For example, the mean value of B11 (B11.mean) and the 5th percentile of hue (hue.quantile.q.0.05) features were found to have substantial average impacts on model outputs, suggesting their strong relevance in distinguishing between different crop types. 

Reflecting on the colors of the bars we can see that 'B11.mean' is important in distinguishing sunflower, sorghum, and millet to a roughly equal degree, and has some small impact on distinguishing other classes. This pattern likely reflects fundamental differences in canopy structure and water content between these dryland crops and the more common maize and rice in the dataset. The SWIR bands are particularly sensitive to plant water content and canopy moisture status, as water strongly absorbs radiation at these wavelengths [@rs13173371]. Sunflower, sorghum, and millet are typically grown in drier conditions and exhibit distinct water use strategies compared to maize. Sunflower, with its deep taproot system and larger leaf area, maintains different canopy moisture levels throughout the growing season. Sorghum and millet, as drought-tolerant cereals with waxy leaf coatings and more efficient water use, exhibit characteristically different SWIR reflectance patterns than the more water-demanding maize crop. The mean SWIR reflectance over the growing season therefore captures these persistent biophysical differences in canopy water status, allowing the model to effectively separate these dryland-adapted crops from others in the training dataset.  The equal importance of B11.mean across these three crops suggests that while this feature helps separate them from the broader "cereal" or "broadleaf" categories, additional features are needed to distinguish among sunflower, sorghum, and millet themselves.

While 'hue.quantile.q.0.05' has the strongest effect distinguishing rice, sunflower, and to a lesser degree cotton. Hue captures the overall brightness and color saturation of the crops across the RGB spectrum. From a visual inspection we can see high values of this variable are correlated with less water stressed geographies like forests and river basins. This relative water abundance might explain the correspondence of plantings of water loving rice and long tap rooted sunflowers.  Looking down the list we can see that features like "EVI.standard.deviation" are most effective at isolating urban areas, as urban areas will have little variation in greenness across the year. 

Looking at 'B12.absolute.sum.of.changes' we can see it best differentiates shrubs, cotton, and forest. The sum of absolute changes measures the total magnitude of variation in a time series by summing the absolute differences between consecutive time steps, effectively quantifying how much a variable fluctuates over the observation period regardless of direction. When applied to the B12 band (SWIR-2, 2100-2280 nm), this metric captures the cumulative volatility in canopy water content and structural properties throughout the growing season. B12's effectiveness at distinguishing shrub, cotton, and forests using sum of absolute changes likely reflects fundamental differences in their temporal stability and management regimes. Cotton, as an intensively managed and flooded annual crop, exhibits pronounced temporal variability in the signal driven by distinct phenological transitions and management interventions [@xun2021novel]. Cotton fields progress through rapid establishment after planting, vigorous vegetative growth, flowering and boll development, and then defoliation before harvest—each transition creating sharp changes in canopy water content and structure that B12 captures.  



\begin{figure}[H]
    \centering
    \includegraphics[width=0.8\linewidth]{figures/mean_shaps_importance_no_other_30_LGBM_kappa_3.png} % Adjust the path and options
    \caption{Top 20 Mean SHAP Feature Importance by Land Cover Type}
    \label{fig:mean_shaps} %can refer to in text with \ref{fig:mean_shaps}
\end{figure}


### Maximum SHAP Values

In Figure \ref{fig:max_shaps}, the maximum SHAP values uncover features that, while perhaps not consistently influential, have high impacts under particular conditions. This aspect of the analysis can be crucial for identifying features that can cause significant shifts in model predictions, potentially corresponding to specific agricultural or environmental contexts. Typically we see a reshuffling of variable importance levels between maximum and mean SHAP values - reflecting differences in the average case vs edge cases. For instance, features such as "hue.median" and "B11.quantile.q.0.95" show high maximum SHAP values, indicating their pivotal roles in determining certain classes. For instance,  "B11.maximum" reflects peak reflectance in SWIR, which could be critical in identifying crops at their maximum biomass, like sunflower at full bloom compared to other crops at different stages of growth. Max SHAP values included two variables 'B12.abs.energy' and 'B12.quantile.q.0.95' that were not included in the mean SHAP values - indicating that these features have high impacts in specific scenarios but are not consistently influential across the dataset. The appearance of these features exclusively in the maximum SHAP analysis reveals an important dimension of model behavior: the classifier has identified unusual or boundary conditions that differentiate ambiguous cases.


\begin{figure}[H]
    \centering
    \includegraphics[width=0.8\linewidth]{figures/max_shaps_importance_no_other_30_LGBM_kappa_3.png} % Adjust the path and options
    \caption{Top 20 Max SHAP Feature Importance by Land Cover Type}
    \label{fig:max_shaps} %can refer to in text with \ref{fig:max_shaps}
\end{figure}
 
<!-- EDITOR: removed a redundant restatement of feature selection here (it repeated the Methods description and imprecisely said "all 30 of both," which fed the 33-vs-32 count ambiguity). The union is now defined once, in Methods. -->

## Model Selection Findings

Optuna tuning selected LightGBM [@ke2017lightgbm]. LightGBM is a gradient-boosting algorithm that combines many simple decision trees to produce a stronger single model, improving the model at each step. LightGBM grows decision trees leaf-wise rather than adding different levels, thereby targeting branches that most need refining.  

## Model Performance

The classification model demonstrated robust performance across multiple land cover classes, as evidenced by the out-of-sample mean confusion matrix with a Cohen's κ of 0.82 and F1-micro score of 0.85 (Table \ref{tab:metrics}), indicating substantial agreement between predicted and actual classifications. Remember that each field is treated as a ‘group’ in the group k-fold procedure to ensure that pixels from the same field are not split between the testing and training groups.  The confusion matrix (Figure \ref{fig:oos_confusion}) shows high diagonal values for most classes, highlighting the model's ability to accurately identify specific land covers. For instance, rice and maize achieved out-of-sample classification accuracies of 92% and 82%, respectively. Our 82% maize accuracy is competitive with smallholder maize mapping elsewhere in East Africa: relying likewise on Google Earth Engine and classical machine learning at national scale, @jin2019 reported maize classification accuracies of approximately 79% in Tanzania and 63% in Kenya, reinforcing that engineered features paired with traditional learners remain effective in these data-scarce settings. Other well-classified categories included millet, sunflower, tidal, water, shrubs, and forest, each with over 73% accuracy. However forest is primarily confused with the category shrub, which is likely a result of poor training data and the difficulty of visually determining trees versus shrubs from high-resolution imagery without the benefit of field visits.

Categories such as sorghum, cassava and cotton displayed moderate confusion with other classes, indicating potential areas for model improvement, especially in distinguishing features that are common between similar crop types.  Confusion between cassava and maize might reflect intercropping practices, where cassava is grown alongside other crops, making it difficult to isolate in the 10 m satellite imagery. The model's performance on these classes suggests that additional discriminative features, more extensive training data, or higher resolution imagery may be necessary to further enhance classification accuracy for these crops.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.8\linewidth]{figures/final_confusion_no_other_model_selection_feature_selection_no_otherLGBM_kappa_3.png} % Adjust the path and options
    \caption{Out of Sample Confusion Matrix}
    \label{fig:oos_confusion} %can refer to in text with \ref{fig:oos_confusion}
\end{figure}

\begin{table}[h!]
\centering
\begin{tabular}{@{}ll@{}}
\toprule
Metric              & Value                 \\ \midrule
Balanced Accuracy   & 0.84    \\
Cohen's κ           & 0.82    \\
Accuracy            & 0.85    \\
F1-micro            & 0.85    \\ 
Precision (UA)   & 0.86    \\ 
Recall (PA)   & 0.84    \\ 
\bottomrule
\end{tabular}
\caption{Summary of Classification Metrics}
\label{tab:metrics}
\end{table}

The overall high out-of-sample performance in Table \ref{tab:metrics} across the majority of categories suggests that the model is effective for practical applications in land cover classification, though further refinement is recommended for categories showing lower accuracy and higher misclassification rates.

We can compare our results across multiple models using Figure \ref{fig:model_compare} below from [@kerner2024accurate]. This plot represents multiple performance metrics of land cover models that include an ‘agricultural’ category specifically for Tanzania. Our model’s performance is indicated by the dashed line. The high level of performance - particularly for the more challenging F1 score - is not surprising given that our model is specifically trained on Tanzanian data, while the other models are typically global or regional models. On the other hand, most of these models include only a single ‘agricultural’ class, meaning their prediction task is a significantly easier one than the one presented here. Given this our strong out-of-sample performance is notable.

\begin{figure}[H]
  \centering
  \includegraphics[width=0.8\linewidth]{figures/model_performance.png}  
    \caption{Tanzanian Land Cover Model Performance Comparison \newline Land cover model performance metrics for Tanzania - dashed line indicates this paper’s model out-of-sample performance across all land covers}
    \label{fig:model_compare} %can refer to in text with \ref{fig:model_compare}
\end{figure}
Source: [@kerner2024accurate]

Overall, the integration of crowdsourced data with traditional machine learning and engineered time-series features yielded a robust model for crop classification in Tanzania. While the model performed exceptionally well for crops like maize and rice, some confusion persisted among similar crop types such as sorghum and cotton. This suggests that additional features, more training data, or higher resolution imagery may be necessary to further enhance classification accuracy for these crops. The challenges encountered, such as variability in crop cycles and challenges of crop identification, highlight the complexities of agricultural monitoring in resource-limited settings. Addressing these issues in future research could improve model performance and generalizability. In sum, our findings demonstrate the practicality of using efficient, interpretable machine learning methods in conjunction with community-driven data collection to advance agricultural monitoring in developing regions.

## Land Cover Product

Using the trained model, we generated a land cover classification map for a sample area in Dodoma and Bukumbi Tanzania, respectively. The maps (Figure \ref{fig:site_00_visualization} \& \ref{fig:site_03_visualization}) illustrate the spatial distribution of various land cover types, including most crop types. The classification results highlight the model's ability to  delineate different crop types and land covers based on the spectral and temporal features extracted from Sentinel-2 imagery.


\begin{figure}
  \centering
  \includegraphics[width=0.8\linewidth]{figures/site_00_visualization.png}  
    \caption{Land Cover Classification Visualization \newline Land cover classification results for a sample area in Dodoma, Tanzania}
    \label{fig:site_00_visualization}  
\end{figure}


\begin{figure}
  \centering
  \includegraphics[width=0.8\linewidth]{figures/site_03_visualization.png}  
    \caption{Land Cover Classification Visualization \newline Land cover classification results for a sample area in Bukumbi, Tanzania}
    \label{fig:site_03_visualization}  
\end{figure}

To illustrate how individual time-series features help delineate different crop types, we include a subset of them for each site including: 'B11.mean', 'hue.quantile.q.0.05' and 'EVI.mean.change' in Figure \ref{fig:site_00_visualization} and 'EVI.standard.deviation', 'B11.abs.energy' and 'B11.standard.deviation' in Figure \ref{fig:site_03_visualization}. These features were selected based on their high SHAP values and their relevance to distinguishing between different crop types. It is also clear that more training data for non-crop land covers like urban, forest and shrub are needed to improve classification in these areas. Moreover some window filtering or smoothing could help reduce the speckling effect seen in some areas.

# Conclusion

In this study, we introduced a novel "lite learning" methodology for crop type classification in Tanzania that demonstrates the continued viability of traditional machine learning approaches in data-scarce environments. By integrating crowdsourced data from the YouthMappers network with automated time-series feature extraction from Sentinel-2 satellite imagery, we addressed critical gaps in agricultural monitoring within resource-limited settings.

The crowdsourced data collection methodology proved both effective and scalable, engaging dozens of students to gather over 1,400 crop observations over a two-week period. This approach not only generated essential training data but also built local capacity in geospatial data collection and analysis, contributing to sustainable knowledge transfer and professional development.

Our approach achieved robust performance with a Cohen's κ of 0.82 and an F1-micro score of 0.85 across seven crop types and five land cover classes. Notably, while our model was trained specifically on Tanzanian data, it matched or exceeded the performance of broadly-used global land cover models that perform the simpler task of classifying undifferentiated 'agriculture'. This achievement is particularly important given that we classified specific crop types—including challenging crops such as cassava, millet, sorghum, sunflower, and cotton—rather than simply identifying agricultural land. Importantly, these are out of sample results using a rigorous group k-fold cross-validation scheme that prevents data leakage between training and testing plots.

The strategic use of SHAP values for feature interpretation revealed the physical and phenological mechanisms driving model predictions. For instance, we found that mean SWIR reflectance (B11.mean) effectively distinguished dryland-adapted crops (sunflower, sorghum, millet) from water-demanding crops by capturing persistent differences in canopy water status throughout the growing season. Similarly, the sum of absolute changes in B12 discriminated between cotton's managed phenological transitions and the more stable spectral signatures of natural vegetation. These insights not only enhanced model interpretability but also provided agronomically meaningful explanations that can inform future data collection and feature engineering efforts.

Our study has several important limitations that suggest directions for future research. Data collection constraints impacted our results: the 2023 drought affected crop health and planting schedules, resulting in fields at varying phenological stages and some early harvests. Our concentrated data collection window (April-May 2023) captured crops primarily in late growing season. Additionally, crop type imbalance in our dataset—the underrepresentation of crops such as peanuts, soybeans, and okra—was handled by dropping these crops from the study. Confusion between similar crops (e.g., cassava and maize) indicates that additional discriminative features or more extensive training data are needed for these challenging classification scenarios.

Geographic generalizability remains an open question. Our model was trained exclusively on data from three districts in northern Tanzania (Arusha, Dodoma, and Mwanza), and its transferability to other regions with different agro-ecological conditions, farming practices, or crop varieties remains untested. The spectral and temporal signatures of crops can vary significantly with climate, soil conditions, and management practices, potentially limiting model performance in new geographic contexts.

Finally, interpretability trade-offs persist despite our use of SHAP values: while providing valuable insights into feature importance, the large number of features (32 final features from hundreds of candidates) still presents challenges for intuitive interpretation.

Several methodological limitations warrant consideration for future applications. Our use of monthly composites, while effective for reducing cloud contamination, may miss rapid phenological changes such as double cropping events or short-duration growth stages that occur within a single month. The linear interpolation applied to fill cloud gaps assumes gradual transitions between observations, which may not accurately capture abrupt changes in crop condition due to management interventions or stress events. Additionally, persistent cloud cover during critical phenological periods could introduce systematic biases in the time-series features. These issues might be addressed by including SAR time series data.  Class boundary decisions, particularly for distinguishing between visually similar land covers such as shrub versus forest, remain challenging and depend heavily on the quality and consistency of training data. Moreover, `xr_fresh` feature extraction, while comprehensive, may not capture all relevant phenological dynamics, suggesting that additional or alternative feature engineering approaches could further enhance model performance.

Future work should focus on expanding geographic coverage to assess model transferability, incorporating multi-temporal data collection throughout the growing season, developing methods to handle class imbalance, and exploring more parsimonious feature sets that balance accuracy with interpretability.

By "turning back the clock" on deep learning, we demonstrate that carefully engineered time-series features—capturing trends, distributions, and temporal complexity—can achieve high classification accuracy without the extensive labeled datasets, computational resources, and training time required by deep learning architectures. This "lite learning" approach offers particular advantages for the resource-limited contexts that characterize much of sub-Saharan Africa and other developing regions, where both training data and computational infrastructure remain scarce.








```{=latex}
\newpage
```



# Acknowledgments

The United States Agency for International Development generously supports this program through a grant from the USAID GeoCenter under Award # AID-OAA-G-15-00007 and Cooperative Agreement Number: 7200AA18CA00015 


# References
