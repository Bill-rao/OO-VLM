# Dataset Preparation

Since the video backbone is frozen during both training and testing, in order to reduce the computational overhead, we independently run the video backbone to extract and save the feature information. 
You can obtain our pre-extracted feature files from [here](https://1drv.ms/f/s!ApS0ZXMgcC11jA71odMufDvEO2eE?e=LvHRcC).

We also provide the [annotation files](https://1drv.ms/f/s!ApS0ZXMgcC11jBsGf-DLtdOaYBg8?e=OPS4vh) and the required [pre-trained weight files](https://1drv.ms/f/s!ApS0ZXMgcC11jBB3M1177DlfGeZB?e=m6lEPm).

## SSV2

The RGB raw data should be prepared according to [VideoMAEV2](https://github.com/OpenGVLab/VideoMAEv2), and the bounding box annotations can be downloaded from [here](https://1drv.ms/f/s!ApS0ZXMgcC11jSMfw8A5ODwT5GtO?e=C1BV9C).

## Note
- After downloading, place both the `data_list` (annotation folder) and `model` (model weights folder) into the root directory of the project.
