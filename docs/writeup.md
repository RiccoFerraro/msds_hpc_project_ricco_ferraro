# HPC term project - Diabetic Retinopathy

### Ricco Ferraro

## Goal:

Predict whether or not an image of a retina has diabetic retinopathy. Profile and/or time running this on the m2 cluseter with varying degrees of optimization to compare the time to train. 
## The Dataset

A 8Gb training set was available in kaggle here: from https://www.kaggle.com/c/diabetic-retinopathy-detection

This dataset contains a large collection of over 45,000 retinal images with binary classifications of retinopathy or not retinopathy for each image in a csv alongside the images.

## The Model

The model used in this study was inspired and largely a port of a pytorch implementation submitted to kaggle here: https://www.kaggle.com/code/fanbyprinciple/pytorch-diabetic-retinopathy  The goal was to port and or wrap this solution to pytorch lightning.

The solution uses google's inception v3 transfer deep Convolutional Neural Network model as implemented in a pytorch neural network model. Training this model locally takes about 24 hours to train on 5000 images as a baseline.
