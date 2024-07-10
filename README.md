# IMU_study

Tracking Upper Limb Usage post-stroke using ZurichMove and Axivity Ax6 IMUs and video-based ground truth.
Dataset of healthy and post-stroke participants collected at the FHT of the SEC in Singapore. 
Optimization of parameters (Count threshold & functional space) used in Gross Movement Activity Count (GMAC).
Duration of arm use based on functional vs. non-functional movements dichotomization. 
Comparison of conventional and optimal approaches.

## Prerequisites
To use the provided code, please install the requirements given in ```requirements.txt``` with pip.
```
pip install -r requirements.txt
```
To prevent dependency issues, we recommend installing the packages in a conda or venv environment.


If questions arise, don't hesitate to contact me: gloorli@ethz.ch.

*Parts of the code in this branch were written with the aid of Google Bard.* \
The code is based on work from Pierre-Louis Gaultier (pierrelouis.gltr@gmail.com)

## Relevant Publications

Subash, T., David, A., ReetaJanetSurekha, S., Gayathri, S., Samuelkamaleshkumar, S., Magimairaj, H. P., Malesevic, N., Antfolk, C., Varadhan, S.K.M., Melendez-Calderon, A., & Balasubramanian, S. (2022). [**Comparing algorithms for assessing upper limb use with inertial measurement units**](https://www.biorxiv.org/content/10.1101/2022.02.24.481756v1.full). bioRxiv.

Josse E. Optimizing digital health metrics to quantify functional upper limb
movements with inertial measurement units; 2024.