# Personalizing Functional Upper Limb Movement Detection in Stroke Survivors using wrist-worn Inertial Measurement Units

*WORK IN PROGRESS*

This repository provides notebooks to analyze Axivity Ax6 IMU sensors data. The data were collected from chronic stroke survivors in the scope of the Future Health Technologies (FHT) programme of the Singapore-ETH Centre (SEC). The data is used to detect functional upper limb (UL) use during a semistructured protocol mimicking tasks of daily life.


## Abstract

**Goal** Assessing upper limb (UL) use in daily life is crucial to objectively monitor post-stroke recovery. Inertial measurement units (IMUs), have emerged as valuable tools for this purpose. However, IMU-based metrics often misclassify involuntary movements as functional, i.e., purposeful UL use. This paper aims to personalize UL movement detection to account for individual movement variability.\
**Methods** A cross-sectional study was conducted with ten stroke survivors wearing wrist-mounted IMUs on both ULs while performing simulated daily life activities. Gross Movement Activity Counting (GMAC) was used to detect functional movements and was evaluated against video recordings. GMAC thresholds were optimized via grid search, regressed against individual Fugl-Meyer Assessment for Upper Extremity scores (FMA-UE), and assessed using leave-one-out cross-validation.\
**Results** The optimized GMAC thresholds of the unaffected ULs created larger functional spaces than those for affected ULs (p=0.04). While there was no monotonic relationship between individual thresholds and FMA-UE, personalizing GMAC improved detection accuracy from 70% to 74% (p=0.03).\
**Conclusions** Personalizing GMAC thresholds based on motor impairment level shows promise for improving functional arm movement detection with high sensitivity and specificity. Though, a larger dataset is needed to develop a robust personalization model.


## Prerequisites

To use the provided code, install the requirements given in ```requirements.txt```.
```
pip install -r requirements.txt
```
To prevent dependency issues, installing the packages in a conda or venv environment is recommended.


## Relevant Publications

Subash, T., David, A., ReetaJanetSurekha, S., Gayathri, S., Samuelkamaleshkumar, S., Magimairaj, H. P., Malesevic, N., Antfolk, C., Varadhan, S.K.M., Melendez-Calderon, A., & Balasubramanian, S. (2022). [**Comparing algorithms for assessing upper limb use with inertial measurement units**](https://www.biorxiv.org/content/10.1101/2022.02.24.481756v1.full). bioRxiv.

Josse, E., Gaultier, P.-L., Kager, S., Cheng, H.-j., Gassert, R., and Lambercy, O. (2024). [**Optimizing Digital Health Metrics to Quantify Functional Upper Limb Movements with Inertial Measurement Units**](https://ieeexplore.ieee.org/document/10719778). BioRob.

Gloor, L. **Personalizing Functional Upper Limb Movement Detection in Stroke Survivors using wrist-worn Inertial Measurement Units**. In preparation.


## General

The code is based on work from Pierre-Louis Gaultier (pierrelouis.gltr@gmail.com) and [Subash's repository](https://github.com/biorehab/upper-limb-use-assessment).

*Parts of the code were written with the aid of Large Language Models (LLM).* 

If questions arise, don't hesitate to contact me: gloorli@ethz.ch.