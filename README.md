# ReadMe

This repository contains the code to build convolutional neuronal networks (CNNs) with Keras and Tensorflow to predict the direction of a lights source from a single input picture. 

**Note:** neither the input data nor the resulting models are contained in this repository as the is too storage-intensive.

The used data held the output direction in spherical coordinates but other coordinate systems could be feasible, too. The used input data contained a simple scenario with one model in the centre of the image and one illuminating directional light source. The input images could either contain three input channels for red, green and blue (RGB) or four channels for RGB-Depth (RGBD) values. The resolution of the input images is 224x224. 

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


The aim was to investigate whether the addition of the depth parameter produces a significant improvement. The findings of this study indicate that neural networks appear to favour RGB input information over RGB-D, for RGB-trained networks do not perform significantly worse than RGB-D-trained networks. Using networks that predict on RGB information allows the usage of more common, inexpensive hardware, as special camera devices to capture additional depth information is not required.

The code is capable to optimize the hyperparameters and train three different net architectures: AlexNet, VGG16 and ResNet50. 

**Note:** The code for the VGG16 is more detailed with more hyperparameter options as the model could not learn at the beginning of the study due to the vanishing gradient problem. By expanding the range of hyperparameters and adding various initialisation functions, the problem could be solved.

With two parameters in the class "global_parameter" the desired output model can be set:    
   - net_architecture: 'ALEX' for the AlexNet, 'VGG16' or 'RESNET' for the ResNet50
   - global.image_channels: 'rgb' or 'rgbd' 

So in total six models where trained whereas the AlexNet had the worse performance and the VGG16 slightly outperformed the ResNet50. Against the initial assumption, the depth parameter did not lead to slightly worse results.

The five Jupiter notebook scripts provide the following functionalities:

### 1_full_HyperParaOpt.ipynb
Trains a model with the selected net architecture and input channels on 20,000 images for one epoch on various hyperparameter combinations. A grid search over all hyperparameter combinations is proceeded by [Talos](https://github.com/autonomio/talos). By storing the mean absolute errors per model the optimal combination can be used for full training in the next script. 

### 2_full_Training.ipynb 
Full training of the selected net architecture and input channel with the best hyperparameter combination (= lowest mean absolute error during the hyperparameter optimization) on 100,000 images and for 400 epochs. If the model is better than all others before it is automatically saved. If no improvements can be achieved the learning rate is reduced after 13 epochs and after 20 epochs the training is stopped. 

### 3_Graphical_Evaluation_Real_Single_Network_Data_Generation.ipynb
Test on 1000 images and calculation of the mean angular estimation error of the selected net architecture and input channel.

### 4_statistical_Tests.ipynb
Statistical tests that the difference between the models trained on RGB images is slightly better than RGB-D images but the effect size is neglectable. 
Distribution of the angular estiamtion error of the ResNet50 on RGB and RGBD images. The blue lines and the µ value show the mean significaltlly differ: 

<img src="https://github.com/MolineraNegra/thesis_deep_learning/tree/master/images/ResNet_RGB_hist(1).png" 
alt="Historam of angualr estiamtion error of the ResNet50 on RGB images" 
width="500"/>

<img src="https://github.com/MolineraNegra/thesis_deep_learning/tree/master/images/RESNET_RGBD_hist(1).png" 
alt="Historam of angualr estiamtion error of the ResNet50 on RGBD images" 
width="500"/>

### 5_Graphical_Evaluation_Box_Whisker_Plot.ipynb
Graphic processing of the results for better comparability 

<img src="https://github.com/MolineraNegra/thesis_deep_learning/tree/master/images/Boxplot_2_.png" 
alt="Box Whisker Plot" 
width="700"/>
