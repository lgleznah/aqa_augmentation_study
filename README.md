# AQA augmentation study

This repository includes all the source code employed in the paper "*Evaluation of data augmentation techniques on aesthetic quality assessment*".

## Requirements

This project depends on the following Python packages:

```
pyyaml==6.0
scikit-learn==1.1.1
pandas==1.4.3
matplotlib==3.5.2
seaborn==0.11.2
keras-cv==0.3.1
```

Additionaly, you can use the following command to get a Docker container with the requirements already installed:

```docker pull lgleznah/tensorflow-2.9.1-gpu-yaml:v6-tf2.10```

## Running experiments

Prior to running any experiments, regardless of the execution environment, you must make sure to set the following environment variables:

```
AQA_AUGMENT_EXPERIMENTS_PATH="Path to the experiments folder. In this repo, they are in the experiments/ folder"
AQA_AUGMENT_ROOT="Path to this repository"
```

Plus, you must set the following environment variables to point to the correct paths for loading AVA and Photozilla datasets:

```
AVA_info_folder="Path to the folder containing the information CSV for AVA"
AVA_images_folder="Path to the folder containing the images in AVA"
Photozilla_info_folder="Path to the folder containing the information CSV for Photozilla"
Photozilla_images_folder="Path to the folder containing the images in Photozilla"
```

These image folders may contain subfolders; the info.csv file of each dataset determines the relative paths to each image, taking the given images folder as the starting point for these relative paths
