# Breast Cancer Classification

This project uses the Breast Cancer Wisconsin dataset from `sklearn` to build a machine learning classification model that predicts whether a tumor is malignant or benign.

## Project Overview

The goal of this project is to practice a complete machine learning workflow, including:

- Loading the dataset
- Creating a pandas DataFrame
- Checking missing values
- Exploring feature distributions
- Detecting skewness
- Reducing skewness using Yeo-Johnson PowerTransformer
- Splitting data into training and testing sets
- Training a Logistic Regression model
- Evaluating the model using accuracy and confusion matrix

## Dataset

The dataset is loaded from:

```python
from sklearn.datasets import load_breast_cancer
