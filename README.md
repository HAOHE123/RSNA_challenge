# RSNA Challenge

Created by He HAO, Ph.D. student (ist194693), with the GitHub link: [HAOHE123](https://github.com/HAOHE123)

## Introduction

The Radiological Society of North America (RSNA) challenge has always been a platform for developing and benchmarking medical imaging solutions. In this year's challenge, I participated with the aim to improve the diagnostic accuracy of degenerative spine conditions using advanced machine learning techniques. As a Ph.D. student in Electrical and Computer Engineering, I applied my expertise in image processing and deep learning to tackle this significant problem.

## Solution Overview

Through my experiments, I discovered that training a single model on all available data led to suboptimal performance, particularly in distinguishing between moderate and severe cases of spinal conditions. The variability in imaging data and the subtle distinctions between different severity levels made it difficult to achieve high accuracy with a generalized approach.

To address these challenges, I adopted a strategy of training separate models for different types of bone abnormalities. This approach allowed for more tailored feature extraction and improved model sensitivity to specific pathological features present in different types of spinal conditions.

## Methodology

### Data Segregation

Instead of a one-size-fits-all model, I segmented the dataset based on the type of bone condition—focusing on categorizing images into normal, moderate, and severe categories. This segmentation helped in dealing with the imbalanced dataset where certain conditions were underrepresented.

### Model Selection and Training

For each segment, I tested various architectures including 2D CNNs for broader feature identification and 3D CNNs for capturing spatial hierarchies and dependencies in volumetric data. The 3D models, in particular, showed superior performance in identifying severe cases, likely due to their ability to analyze the spatial context of the abnormalities more effectively.

### Special Focus on Moderate and Severe Cases

Moderate and severe cases are clinically significant and often require precise diagnosis to determine the appropriate treatment path. To enhance the model's performance on these categories, I employed several techniques:
- **Weighted Loss Functions:** To counter the imbalance in training data, I utilized weighted loss functions which place greater importance on correctly predicting underrepresented classes.
- **Data Augmentation:** To artificially expand the dataset, especially for moderate and severe cases, I used data augmentation techniques such as rotation, zoom, and flip, which are standard practices in medical image processing to generate diverse training examples without collecting more data.
- **Ensemble Learning:** I used an ensemble of models for each bone condition to improve diagnostic accuracy. By averaging the predictions from multiple models, I reduced the likelihood of overfitting and boosted the robustness of the predictions.

## Results and Conclusion

The separate training and the focused attention on moderate and severe cases significantly improved the model's performance. By employing 3D detection methods and adjusting the model weights towards more severe cases, I achieved notable success in the RSNA challenge, securing a bronze medal. This approach not only enhanced the accuracy but also improved the model’s ability to generalize across different imaging modalities and patient demographics.

This experience has been incredibly fulfilling, pushing the boundaries of my understanding of both machine learning and medical imaging. The insights gained from this challenge will undoubtedly contribute to my further research and development in the field.
