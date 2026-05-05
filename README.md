# Breast Cancer Classification

This project builds a machine learning classification pipeline using the Breast Cancer Wisconsin dataset from `scikit-learn` to predict whether a tumor is **malignant** or **benign**. The project was developed as a structured beginner-to-intermediate classical machine learning workflow, with emphasis on data understanding, preprocessing, model training, evaluation, and clean project organization.

## Project Objective

The main objective of this project is to apply the full machine learning pipeline to a real classification problem and understand each stage conceptually, not just run code. The project focuses on:

- understanding the dataset and target variable
- exploring feature distributions
- checking data quality
- preprocessing the data correctly
- handling skewed features
- scaling input features
- training a baseline classification model
- evaluating the model using appropriate metrics
- organizing the project in a clean and reusable structure

## Dataset

The dataset used in this project is the **Breast Cancer Wisconsin Diagnostic Dataset**, loaded directly from `scikit-learn`.

It contains numerical features computed from digitized images of breast mass cell nuclei. The target variable indicates whether the tumor is:

- **malignant**
- **benign**

This is a **supervised binary classification** problem.

## Workflow

The project follows a complete machine learning workflow:

1. Load the dataset from `scikit-learn`
2. Convert the dataset into a pandas DataFrame
3. Inspect the data structure and summary statistics
4. Check for missing values
5. Explore feature distributions visually
6. Detect skewed features
7. Apply preprocessing:
   - train/test split
   - feature standardization
   - Yeo-Johnson transformation on highly skewed features
8. Train a **Logistic Regression** model
9. Evaluate the model using:
   - accuracy
   - precision
   - recall
   - F1-score
   - confusion matrix

## Project Structure

```text
breast-cancer-classification/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── breast_cancer_analysis.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── train_model.py
│   └── evaluate_model.py
│
├── results/
│   ├── confusion_matrix.png
│   ├── feature_distributions.png
│   └── metrics.txt
│
└── images/
    └── project_banner.png
