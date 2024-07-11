# IMU_study

*WORK IN PROGRESS*

This repository provides notebooks to analyze ZurichMove and Axivity Ax6 IMU sensors data. The data was collected from healthy and post-stroke participants in the scope of the Future Health Technologies (FHT) programme of the Singapore-ETH Centre (SEC). The data is used to detect functional upper limb (UL) use during a semistructured protocol mimicking tasks of daily life.\
The notebooks include code to optimize the parameters (Count threshold & functional space) of the Gross Movement Activity Count (GMAC) algorithm per participant. This allows to check wheter a GMAC with personalized parameters can improve the detection of functional UL use compared to the initaly chosen thresholds when Subash et al. introduced the GMAC.\
Ground truth is based on videotaping participants during the study.


## Prerequisites

To use the provided code, please install the requirements given in ```requirements.txt``` with pip.
```
pip install -r requirements.txt
```
To prevent dependency issues, we recommend installing the packages in a conda or venv environment.


## Relevant Publications

Subash, T., David, A., ReetaJanetSurekha, S., Gayathri, S., Samuelkamaleshkumar, S., Magimairaj, H. P., Malesevic, N., Antfolk, C., Varadhan, S.K.M., Melendez-Calderon, A., & Balasubramanian, S. (2022). [**Comparing algorithms for assessing upper limb use with inertial measurement units**](https://www.biorxiv.org/content/10.1101/2022.02.24.481756v1.full). bioRxiv.

Josse et al. (2024). **Optimizing digital health metrics to quantify functional upper limb
movements with inertial measurement units**. BioRob 2024. (in review)


## General

The code is based on work from Pierre-Louis Gaultier (pierrelouis.gltr@gmail.com) and [Subash's repository](https://github.com/biorehab/upper-limb-use-assessment).

*Parts of the code in this branch were written with the aid of Google Bard.* 

If questions arise, don't hesitate to contact me: gloorli@ethz.ch.