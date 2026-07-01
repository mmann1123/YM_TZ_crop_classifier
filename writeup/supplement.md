<!-- compile working with:
cd writeup
pandoc supplement.md --template=mytemplate.tex -o supplement.pdf --bibliography=refs.bib --pdf-engine=xelatex

pandoc supplement.md --template=mytemplate.tex \
  --from markdown+raw_tex \
  --to latex \
  --bibliography=refs.bib \
  --natbib -V biblio-style=IEEEtran -V 'natbiboptions=numbers,sort&compress' \
  -o final_submission/supplement.tex
# then: xelatex supplement && bibtex supplement && xelatex supplement && xelatex supplement
-->
---
title: "Supplementary Material: Lite Learning: Efficient Crop Classification in Tanzania Using Feature Extraction with Machine Learning & Crowd Sourcing"
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
classoption: [journal, onecolumn]
header-includes:
  - \usepackage[margin=1in]{geometry}
  - \usepackage{longtable}
  - \usepackage{amsmath}
  - \usepackage{amsfonts}
  - \usepackage{array}
  - \usepackage{booktabs}
  - \usepackage{caption}
  - \renewcommand{\thetable}{S\arabic{table}}
  - \renewcommand{\thefigure}{S\arabic{figure}}
  - \renewcommand{\thesection}{S\arabic{section}}
---

# S-I. Time Series Feature Definitions

This document is the online supplement to the main article *"Lite Learning: Efficient Crop Classification in Tanzania Using Feature Extraction with Machine Learning & Crowd Sourcing."* It provides the full list of time-series features referenced in the main text (Methods, "Time Series Features").

The features were extracted on a per-pixel basis from the 2023 Sentinel-2 monthly composites (bands B2, B6, B11, B12, the Enhanced Vegetation Index (EVI), and hue) using the `xr_fresh` toolkit [@xr_fresh_2021]. They summarize the temporal dynamics of crop growth and development across the growing season, spanning basic statistical descriptors, changes over time, and distribution-based metrics. Table S1 lists each statistic, a short description, and its defining equation. Throughout, $x_i$ denotes the value of the band or index at time step $i$ in a pixel's monthly time series of length $n$, $\bar{x}$ its mean, $\sigma$ its standard deviation, $\mu_4$ its fourth central moment, and $l$ an autocorrelation lag.

\renewcommand{\arraystretch}{1.5}
\begin{longtable}{|p{4cm}|p{5cm}|p{6cm}|}
\caption{Time-series features extracted with \texttt{xr\_fresh}, with descriptions and defining equations.}\label{tab:ts_features}\\
\hline
\textbf{Statistic} & \textbf{Description} & \textbf{Equation} \\
\hline
\endhead
Absolute energy &  sum over the squared values & $E = \sum_{i=1}^n x_i^2$ \\
Absolute Sum of Changes  & sum over the absolute value of consecutive changes in the series  & $ \sum_{i=1}^{n-1} \mid x_{i+1}- x_i \mid $ \\
Autocorrelation (1, 2 \& 3 month lag) & Correlation between the time series and its lagged values & $\frac{1}{(n-l)\sigma^{2}} \sum_{t=1}^{n-l}(x_{t}-\bar{x})(x_{t+l}-\bar{x})$\\
Count Above Mean & Number of values above the mean & $N_{\text{above}} = \sum_{i=1}^n (x_i > \bar{x})$ \\
Count Below Mean & Number of values below the mean & $N_{\text{below}} = \sum_{i=1}^n (x_i < \bar{x})$ \\
Day of Year of Maximum Value & Day of the year when the maximum value occurs in series & --- \\
Day of Year of Minimum Value & Day of the year when the minimum value occurs in series & --- \\
Kurtosis & Measure of the tailedness of the time series distribution & $G_2 = \frac{\mu_4}{\sigma^4} - 3$ \\
Linear Time Trend & Linear trend coefficient estimated over the entire time series & $b = \frac{\sum_{i=1}^n (t_i - \bar{t})(x_i - \bar{x})}{\sum_{i=1}^n (t_i - \bar{t})^2}$ \\
Longest Strike Above Mean & Longest consecutive sequence of values above the mean & --- \\
Longest Strike Below Mean & Longest consecutive sequence of values below the mean & --- \\
Maximum & Maximum value of the time series & $x_{\text{max}}$ \\
Mean & Mean value of the time series & $\bar{x} = \frac{1}{n}\sum_{i=1}^n x_i$ \\
Mean Absolute Change & Mean of absolute differences between consecutive values & $\frac{1}{n-1} \sum_{i=1}^{n-1} | x_{i+1} - x_{i}|$ \\
Mean Change & Mean of the differences between consecutive values & $ \frac{1}{n-1} \sum_{i=1}^{n-1}  (x_{i+1} - x_{i}) $ \\
Mean Second Derivative Central & measure of acceleration of changes in a time series data & $\frac{1}{2(n-2)} \sum_{i=1}^{n-2}  \frac{1}{2} (x_{i+2} - 2 \cdot x_{i+1} + x_i)
$ \\
Median & Median value of the time series & $\tilde{x}$ \\
Minimum & Minimum value of the time series & $x_{\text{min}}$ \\
Quantile (q = 0.05, 0.95) & Values representing the specified quantiles (5th and 95th percentiles) & $Q_{0.05}, Q_{0.95}$ \\
Ratio Beyond r Sigma (r=1,2,3) & Proportion of values beyond r standard deviations from the mean & $P_r = \frac{1}{n}\sum_{i=1}^{n} (|x_i - \bar{x}| > r\sigma)$ \\
Skewness & Measure of the asymmetry of the time series distribution & $\frac{n}{(n-1)(n-2)} \sum \left(\frac{x_i - \bar{x}}{\sigma}\right)^3$ \\
Standard Deviation & Standard deviation of the time series & $\sqrt{\frac{1}{n}\sum_{i=1}^{n} (x_i - \bar{x})^2}$ \\
Sum Values & Sum of all values in the time series & $S = \sum_{i=1}^{n} x_i$ \\
Symmetry Looking & Measures the similarity of the time series when flipped horizontally & $| \bar{x}-\tilde{x} | < r \cdot (x_{\text{max}} - x_{\text{min}} ) $ \\
Time Series Complexity (CID CE) & measure of number of peaks and valleys & $\sqrt{ \sum_{i=1}^{n-1} ( x_{i+1} - x_{i})^2 }$\\
Variance & Variance of the time series & $\sigma^2 = \frac{1}{n}\sum_{i=1}^{n} (x_i - \bar{x})^2$ \\
Variance Larger than Standard Deviation & check if variance is larger than standard deviation & $\sigma^2 > 1$ \\
\hline
\end{longtable}

# S-II. Computational Complexity of Deep-Learning Baselines

This section expands the efficiency comparison in the main text (Methods, "Computational Efficiency"), where the LightGBM "lite learning" model trains in 10--15 minutes on a CPU-only laptop. By contrast, the deep-learning architectures commonly applied to satellite image time series carry substantially higher training-time and memory costs. As in the main text, $n$ denotes the number of training samples.

Recurrent models such as LSTMs, frequently used for temporal land cover analysis, exhibit complexity:

$$T_{LSTM} = O(e \cdot n \cdot t \cdot (h^2 + h \cdot i))$$

where $e$ is the number of epochs, $t$ the sequence length, $h$ the hidden dimensionality, and $i$ the input feature dimension. The quadratic dependence on $h$ and the inherently sequential nature of LSTMs prevent efficient parallelization across the temporal axis, often extending training to several hours.

Spatial architectures further escalate these demands. A 2D U-Net scales linearly with the number of input channels (treating time steps as channels), but a 3D U-Net, which treats the temporal dimension as a third spatial axis, follows:

$$T_{3DUNet} = O\left(e \cdot n \cdot \sum_{l} k_l^3 \cdot c_{l-1} \cdot c_l \cdot s_l\right)$$

where the sum runs over network layers $l$ with kernel size $k_l$, input and output channel counts $c_{l-1}$ and $c_l$, and spatial size $s_l$. The cubic kernel term ($k_l^3$) means a $3 \times 3 \times 3$ convolution requires 27 operations per voxel, compared to 9 for a 2D equivalent. Furthermore, the memory (VRAM) required to store 4D tensors (Space $\times$ Time $\times$ Channels) for large-scale mapping routinely necessitates high-end GPU clusters.

# References {.unnumbered}
