# Cell Nuclei Detection for Semantic Segmentation

This project aims to develop a machine learning model that can automatically detect cell nuclei in images, which is crucial for advancing biomedical research, particularly in the study of diseases like cancer, Alzheimer's, diabetes, and more. The goal of this project is to create a semantic segmentation model using U-Net, a popular architecture for image segmentation tasks. The model will be trained on the Data Science Bowl 2018 dataset from Kaggle, with the goal of achieving high performance (above 80% accuracy).

## Project Overview

Detecting cell nuclei is an important first step in biomedical research, enabling researchers to examine DNA and understand how cells respond to treatments. The goal of this project is to build an AI model capable of segmenting cell nuclei in microscopic images, streamlining drug testing and aiding in the discovery of cures for various diseases.

The model will be developed and evaluated following these steps:

- Problem Formulation

- Data Preparation

- Model Development

- Model Deployment

- The final model will be used for predictions on unseen test data, with the aim of ensuring high accuracy and generalization while avoiding overfitting.

## Dataset

The dataset used for this project is from the Kaggle competition Data Science Bowl 2018. You can access and download the dataset from Kaggle using the link below:

https://www.kaggle.com/competitions/data-science-bowl-2018/overview

Data Science Bowl 2018 Dataset

The dataset consists of images containing cell nuclei and corresponding segmentation masks. These masks highlight the boundaries of cell nuclei, which will be the target for the semantic segmentation model.

## Requirements
Before running the code, ensure that the following dependencies are installed:

Python 3.x

TensorFlow 2.x

Keras

NumPy

Matplotlib

Pandas

OpenCV

Scikit-learn

Kaggle (for downloading dataset)

# Workflow

## 1. Problem Formulation

The task at hand is a semantic segmentation problem, where the model needs to identify the regions in the image corresponding to cell nuclei. The output of the model is a mask image where each pixel is classified as part of a nucleus or not.

## 2. Data Preparation

Download the Dataset: The dataset is available on Kaggle, and can be downloaded using the provided Kaggle link.

Data Preprocessing: The images and masks are resized and normalized to prepare them for input into the model. Data augmentation techniques are applied to improve the robustness of the model.

Train/Test Split: The dataset is split into training and validation sets to assess model performance.


## 3. Model Development
A U-Net model is used for semantic segmentation, with the following architecture:

Downsampling Path: The downsampling path uses transfer learning to leverage pre-trained models, which helps extract relevant features.

Upsampling Path: The upsampling path is built using Conv2DTranspose layers to reconstruct the image and generate the final segmentation mask.

The model is trained to minimize the cross-entropy loss, and various performance metrics such as accuracy and IoU (Intersection over Union) are used to evaluate the model.

## 4. Model Deployment

The trained model is used to make predictions on unseen test data. The model is tested for its ability to generalize to new images, and the results are compared with the ground truth.

Model Evaluation

Accuracy: The model is evaluated for both training and validation accuracy, with the goal of achieving more than 80% accuracy.

Overfitting Prevention: Regularization techniques like dropout, batch normalization, and data augmentation are employed to prevent overfitting.

## Notes

This project uses a pre-trained model for the downsampling path (e.g., MobileNet or ResNet), and builds the upsampling path using U-Net architecture.

The model has been trained with the goal of preventing overfitting, ensuring that the performance on unseen data remains strong.

##  Credits

Dataset: Data Science Bowl 2018 Dataset 
https://www.kaggle.com/competitions/data-science-bowl-2018/overview

U-Net Architecture: Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. arXiv link

## License

This project is licensed under the MIT License - see the LICENSE file for details.

# Screenshot

## Model Architecture: 
<img width="496" alt="Model_Architecture" src="https://github.com/user-attachments/assets/bf60f36e-3a77-49f0-86e0-2c4d00dfab33">

