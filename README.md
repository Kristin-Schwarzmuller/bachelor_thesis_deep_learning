# ReadMe

## Table of Contents
- [Introduction](#introduction)
- [Data](#data)
- [Models](#models)
- [Results](#results)
- [Scripts](#scripts)

## Introduction

This repository contains code for building convolutional neural networks (CNNs) using Keras and TensorFlow to predict the direction of a light source from a single input image. The goal is to predict the light source direction to improve shadow fall and light reflections for augmented reality objects. The following image shows an augmented box with a misallocated shadow compared to the point light and other objects.

<img src="https://github.com/MolineraNegra/thesis_deep_learning/blob/master/images/AR.jpeg" 
alt="Spherical Coordinate System" 
width="300"/>

*Figure 1: Augmented Reality Image*

**Note:** Neither the input data nor the resulting models are contained in this repository due to their storage-intensive nature.

## Data

The data used kept the output direction in spherical coordinates. The spherical coordinate system offers the advantage of using only two variables: the elevation θ (theta) and the azimuth Φ (phi). Since the light source is directional, the distance is negligible.

<img src="https://github.com/MolineraNegra/thesis_deep_learning/blob/master/images/spericalCoo.PNG" 
alt="Spherical Coordinate System" 
width="500"/>

*Figure 2: Spherical Coordinate System*

The input data contained a simple scenario with one model in the center of the image and an illuminating directional light source. The input images could either contain three input channels for red, green, and blue (RGB) or four channels for RGB-Depth (RGBD) values. The resolution of the input images is 224x224.

**Exaple RGB images  with the light direction: azimuth = 315° and elevation = 45°** 

<img src="https://github.com/MolineraNegra/thesis_deep_learning/blob/master/images/rgb/buddha00002438-0-30-315-45.png" 
alt="RGB image of an Buddha" 
width="300"/>
<img src="https://github.com/MolineraNegra/thesis_deep_learning/blob/master/images/rgb/bunny00043910-0-30-315-45.png" 
alt="RGB image of an Bunny" 
width="300"/>

**Exaple RGBD images  with the light direction: azimuth = 315° and elevation = 45°**

<img src="https://github.com/MolineraNegra/thesis_deep_learning/blob/master/images/rgba/buddha00002438-0-30-315-45.png" 
alt="RGBD image of an Buddha" 
width="300"/>
<img src="https://github.com/MolineraNegra/thesis_deep_learning/blob/master/images/rgba/bunny00043910-0-30-315-45.png" 
alt="RGBD image of an Bunny" 
width="300"/>

## Models

### RGB vs. RGB-D Neural Networks

Two different neural network models were trained and compared using RGB and RGB-Depth images. Surprisingly, no significant difference in performance was observed between the two models. Both models were trained and evaluated using the same framework and hyperparameters.

## Results

The aim was to investigate whether the addition of the depth parameter produces a significant improvement. The findings of this study indicate that neural networks appear to favor RGB input information over RGB-D. Using networks that predict on RGB information allows the usage of more common, inexpensive hardware, as special camera devices to capture additional depth information are not required.

<img src="https://github.com/MolineraNegra/thesis_deep_learning/blob/master/images/ResNet_RGB_hist_1_.png" 
alt="Historam of angualr estiamtion error of the ResNet50 on RGB images" 
width="500"/>

<img src="https://github.com/MolineraNegra/thesis_deep_learning/blob/master/images/RESNET_RGBD_hist_1_.png" 
alt="Historam of angualr estiamtion error of the ResNet50 on RGBD images" 
width="500"/>

### Comparison with Box Whisker Plot
Graphic processing of the results for better comparability 

<img src="https://github.com/MolineraNegra/thesis_deep_learning/blob/master/images/Boxplot_2_.png" 
alt="Box Whisker Plot" 
width="700"/>

## Scripts

For more detailed information on the scripts and their functionalities, please refer to the individual notebook files mentioned.

The provided Jupyter notebook scripts offer functionalities for hyperparameter optimization, full training, graphical evaluation, and statistical tests:

### 1_full_HyperParaOpt.ipynb
Trains a model with the selected net architecture and input channels on 20,000 images for one epoch using various hyperparameter combinations. A grid search over all hyperparameter combinations is performed using [Talos](https://github.com/autonomio/talos). Optimal combinations are determined based on mean absolute errors for full training in the next script.

### 2_full_Training.ipynb
Fully trains the selected net architecture and input channel using the best hyperparameter combination (lowest mean absolute error during hyperparameter optimization) on 100,000 images and for 400 epochs. The model is saved if it is better than previous models, with adjustments for reduced learning rate and early stopping if needed.

### 3_Graphical_Evaluation_Real_Single_Network_Data_Generation.ipynb
Tests on 1000 images and calculates mean angular estimation error for the selected net architecture and input channel.

### 4_statistical_Tests.ipynb
Conducts statistical tests showing that the difference between models trained on RGB images is slightly better than RGB-D images, but the effect size is negligible.

### 5_Graphical_Evaluation_Box_Whisker_Plot.ipynb
Graphic processing of the results for better comparability 

<img src="https://github.com/MolineraNegra/thesis_deep_learning/blob/master/images/Boxplot_2_.png" 
alt="Box Whisker Plot" 
width="700"/>



