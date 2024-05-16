## Turion Space assessment

This repository contains experiments and implementations for uncertainty quantification in deep neural networks applied to image segmenetation tasks using a UNet CNN architecture. The goal is to leverage dropout layers and other uncertainity qunatifications to measure prediction uncertainty.

- [Turion Space assessment](#turion-space-assessment)
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
  - [Primary Files:](#primary-files)
  - [Directory Structure:](#directory-structure)
- [Satellite Dataset](#satellite-dataset)
- [Information](#information)
- [Acknowledgements](#acknowledgements)
  - [Dataset:](#dataset)
  - [MC Dropout](#mc-dropout)
  - [Further UQ in Deep Learning](#further-uq-in-deep-learning)

## Introduction

Uncertainty quantification is crucial in critical applications like space object detection. This project aims to implement techniques to measure uncertainty in image segmentation models using MC dropout and other UQ techniques.

## Installation

Clone the repository:

```bash
git clone https://github.com/mustachemo/Turion-Space-assessment.git
cd Turion-Space-assessment
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Primary Files:

- `main.py`: The main script to run the experiments.
- `train.py`: The training script for the U-Net model.
- `predict.py`: The prediction script for the U-Net model.
- `configs.py`: The configuration file for the experiments.

### Directory Structure:

- `model/`: The U-Net model implementation.
- `uncertainty_quantification/`: helper functions for Uncertainty Quantification experiments.
- `utils/`: utility functions for data loading, visualization, and custom loss functions.
- `checkpoints/`: saved model checkpoints (empty/nonexistant until a model is trained).
- `data/`: the dataset directory (empty/nonexistant until you download the [dataset](https://github.com/Yurushia1998/SatelliteDataset) and put it in here).
- `logs/`: logs for training and validation metrics.
- `prepped_data/`: preprocessed data for training and validation (empty/nonexistant until data from data/ directory is preprocessed).
- `reports/`: report directory. Includes images, tex file, and pdf for report.

## Satellite Dataset

The satellite dataset is primarly for object detection and segmentation using both synthetic and real satellite images. The dataset includes 3117 images (1280x720), bounding boxes, and masks (1280x720). Each satellite is segmented into at most 3 parts, including body, solar panel, and antenna, represented by three colors: green, red, and blue.

![Example sample](report/images/original_input_sample.png)

**Dataset Details**

- Images with index 0-1002 have fine masks, while images from index 1003-3116 have coarse masks.
- Training data includes 403 fine masks from index 0-402 and 2114 coarse masks from index 1003-3116.
- The validation dataset includes 600 images with fine masks indexed from 403 to 1002.
- the file `all_bbox.txt` includes bounding boxes of all satellites inside the dataset based on segmentation masks. It's in the form of a dictionary with the index of images as the key. Each bounding box has the format [max_x, max_y, min_x, min_y].

## Information

- `model/bayesian_unet.py`: This attempts to implement a Bayesian U-Net model using MC Dropout, could not verify if it works because of memory constraints.

## Acknowledgements

### Dataset:

- Link to the paper: [A Spacecraft Dataset for Detection, Segmentation and Parts Recognition](https://arxiv.org/abs/2106.08186)
- Link to the dataset: [https://github.com/Yurushia1998/SatelliteDataset](https://github.com/Yurushia1998/SatelliteDataset)

### MC Dropout

- [Yarin Gal](http://www.cs.ox.ac.uk/people/yarin.gal/website/index.html)

### Further UQ in Deep Learning

- [A review of uncertainty quantification in deep learning: Techniques, applications and challenges](https://www.sciencedirect.com/science/article/pii/S1566253521001081).
