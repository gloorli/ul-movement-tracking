# Personalizing Functional Upper Limb Movement Detection in Stroke Survivors using wrist-worn Inertial Measurement Units

*WORK IN PROGRESS*

This repository provides notebooks to analyze Axivity Ax6 IMU sensors data. The data were collected from chronic stroke survivors in the scope of the Future Health Technologies (FHT) programme of the Singapore-ETH Centre (SEC). The data is used to detect functional upper limb (UL) use during a semistructured protocol mimicking tasks of daily life.


## Abstract

**Introduction** Inertial measurement units (IMU) show promise in quantifying rehabilitation progress to enable more personalized therapy for stroke survivors. Gross Movement Activity Counting (GMAC) has emerged as a potential algorithm for classifying functional versus non-functional arm movements in daily life. However, it lacks the accuracy needed for clinical implementation. This study explores the potential of personalizing GMAC with respect to motor impairment.\
**Methods** In a cross-sectional study, stroke survivors wore IMU sensors on both wrists while performing tasks mimicking daily life. Simultaneous video data were labeled to obtain the ground truth. GMAC thresholds were optimized for each individual to detect functional movements with high sensitivity and specificity. The classification performance of the personalized GMAC, tailored to individual motor impairment, was evaluated using leave-one-subject-out cross-validation.\
**Results** Five hours of IMU and video data from 10 stroke survivors with mild to severe upper limb impairment were recorded. Functional use of the affected arm increased in mildly impaired stroke survivors, primarily due to increased transport use, while stabilizing use remained constant across impairment levels. Individual GMAC thresholds appeared to be influenced by the level of impairment. Functional movement detection in the affected arm improved significantly (p<0.05) with optimal GMAC thresholds for movement intensity (1.9 counts per second) and forearm elevation (42.6°). Personalized thresholds further increased the area under the receiver operating characteristic curve to 0.74 and reduced the variation in classification performance across subjects.\
**Conclusion** Personalizing GMAC thresholds based on motor impairment level shows promise for improving functional arm movement detection with high sensitivity and specificity. Though, a larger dataset is needed to develop a robust personalization model.


## Prerequisites

To use the provided code, install the requirements given in ```requirements.txt```.
```
pip install -r requirements.txt
```
To prevent dependency issues, installing the packages in a conda or venv environment is recommended.


## Relevant Publications

Subash, T., David, A., ReetaJanetSurekha, S., Gayathri, S., Samuelkamaleshkumar, S., Magimairaj, H. P., Malesevic, N., Antfolk, C., Varadhan, S.K.M., Melendez-Calderon, A., & Balasubramanian, S. (2022). [**Comparing algorithms for assessing upper limb use with inertial measurement units**](https://www.biorxiv.org/content/10.1101/2022.02.24.481756v1.full). bioRxiv.

Josse, E., Gaultier, P.-L., Kager, S., Cheng, H.-j., Gassert, R., and Lambercy, O. (2024). **Optimizing digital health metrics to quantify functional upper limb movements with inertial measurement units**. Accepted in press.

Gloor, L. **Personalizing Functional Upper Limb Movement Detection in Stroke Survivors using wrist-worn Inertial Measurement Units**. In preparation.


## General

The code is based on work from Pierre-Louis Gaultier (pierrelouis.gltr@gmail.com) and [Subash's repository](https://github.com/biorehab/upper-limb-use-assessment).

*Parts of the code were written with the aid of Large Language Models (LLM).* 

If questions arise, don't hesitate to contact me: gloorli@ethz.ch.