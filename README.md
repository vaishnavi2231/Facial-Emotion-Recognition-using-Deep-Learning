# Facial-Emotion-Recognition-using-Deep-Learning

Project Overview :
This project focuses on enhancing facial emotion recognition performance using the FER-2013 dataset through the integration of image preprocessing techniques and deep learning models. Facial Emotion Recognition (FER) is a challenging task due to the low resolution, low textural variations, and noise present in real-world facial images. As state-of-art models like Convolutional Neural Network and pre-defined architectures have shown good results on benchmark datasets, the accuracy depends on the quality of input images.
In this work, we explore the impact of various preprocessing techniques such as Histogram Equalization, Median Filtering, Sobel Edge Detection, Canny Edge Detection, and Unsharp Masking (a combination of Gaussian blur and 2D convolution) on FER performance. We have also evaluated the effectiveness of stacking the filtered image with original images to improve the input image quality. Additionally, we implemented a real-time emotion detection system using OpenCV, where facial input from a webcam is passed through two trained models - one using original images and the other using sharpened image stacks - to predict facial expressions in real time.

1.	Code : 

This folder has a total of 5 coding files : 
1.	1_Preprocessing.ipynb :  This coding file performs all the preprocessing tasks and creates filtered image datasets.

2.	2_Base_CNN_Models.ipynb :  This applies Base CNN model to Original, Filtered and Stacked Dataset.

3.	3_ResNet_on_Filtered_Dataset.ipynb  : This applies the Resenet model on Filtered Dataset.

4.	4_Resnet_on Stacked_Dataset.ipynb : This applies the Resnet model on Stacking image dataset.

5.	5_Real_time_prediction.py : This performs the real time prediction of facial emotion recognition. To run this file below .pth file of saved models should be added to the executing folder - 
	1. Resnet50_Original.pth
	2. Resnet50_Stack_Ori_Sharpen.pth
These files can be found in the Saved_Models folder shared within the submission folder.

2.	Saved Models : 
a.	BaseCNN.pth
b.	Resnet50_Original.pth : Model trained on Original Dataset
c.	Resnet50_Stack_Ori_Sharpen.pth : Model trained on Stacked Original + Sharpened Filtered dataset

<img width="361" height="154" alt="Accuracy Comparison" src="https://github.com/user-attachments/assets/0828f973-3ae3-49e6-8e17-baef13855a1f" />

