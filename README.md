# TFT-multi: simultaneous forecasting of vital sign trajectories in the ICU

https://arxiv.org/abs/2409.15586 
### Abstract
Trajectory forecasting in healthcare data has been an important area of research in precision care and clinical integration for computational methods. In recent years, generative AI models have demonstrated promising results in capturing short and long range dependencies in time series data. While these models have also been applied in healthcare, most of them only predict one value at a time, which is unrealistic in a clinical setting where multiple measures are taken at once. In this work, we extend the framework temporal fusion transformer (TFT), a multi-horizon time series prediction tool, and propose TFT-multi, an end-to-end framework that can predict multiple vital trajectories simultaneously. We apply TFT-multi to forecast 5 vital signs recorded in the intensive care unit: blood pressure, pulse, SpO2, temperature and respiratory rate. We hypothesize that by jointly predicting these measures, which are often correlated with one another, we can make more accurate predictions, especially in variables with large missingness. We validate our model on the public MIMIC dataset and an independent institutional dataset, and demonstrate that this approach outperforms state-of-the-art univariate prediction tools including the original TFT and Prophet, as well as vector regression modeling for multivariate prediction. Furthermore, we perform a study case analysis by applying our pipeline to forecast blood pressure changes in response to actual and hypothetical pressor administration.

### Code organization
sample.csv: sample input format for TFT-multi

TFT-multi.ipynb: example workflow of running TFT-multi

model.py: model implementation of TFT-multi, extension from original TFT model [adopted from https://github.com/PlaytikaOSS/tft-torch?tab=readme-ov-file]

visualization.py: functions for visualizing output [adopted from https://github.com/PlaytikaOSS/tft-torch?tab=readme-ov-file]
