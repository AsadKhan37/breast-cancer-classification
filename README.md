Below is a stronger, portfolio-ready `README.md`. It directly addresses the review points: repo structure, dataset details, model comparison, results, interpretation, limitations, future work, and reproducibility.

You should replace the metric values in the **Results** section with the exact numbers from your `results_preprocessed_data.txt` and `results_raw_data.txt`.


# Breast Cancer Classification

This project implements a classical machine learning pipeline for breast cancer tumor classification using the **Breast Cancer Wisconsin Diagnostic Dataset** from `scikit-learn`.

The goal is to predict whether a tumor is **malignant** or **benign** using numerical features extracted from digitized images of breast mass cell nuclei. The project compares Logistic Regression trained on **raw data** against Logistic Regression trained on **preprocessed data** using scaling and skewness transformation.

The final workflow includes data loading, exploratory analysis, preprocessing, model training, model comparison, evaluation, visualization, model saving, and clean project organization.

---

## Project Summary

This project uses **Logistic Regression** as the main classification model.

Two model versions were trained and compared:

1. **Raw-data Logistic Regression**
   - Trained directly on the original dataset features.
   - Used mainly as a baseline for comparison.

2. **Preprocessed-data Logistic Regression**
   - Trained after feature scaling using `StandardScaler`.
   - Highly skewed features were transformed using `PowerTransformer` with the Yeo-Johnson method.
   - This is the preferred model because preprocessing improves optimization stability and coefficient interpretability.

Both models achieved strong performance, but the preprocessed model is considered more reliable because Logistic Regression is sensitive to feature scale and distribution.

---

## Project Objective

The main objective of this project is to build and evaluate a binary classification model for breast cancer diagnosis.

This project demonstrates:

- Loading a real-world dataset from `scikit-learn`
- Converting the dataset into a pandas DataFrame
- Understanding the target variable
- Checking data quality and missing values
- Exploring feature distributions
- Detecting skewed features
- Applying preprocessing correctly
- Training Logistic Regression models
- Comparing raw-data and preprocessed-data models
- Evaluating classification performance
- Saving trained models
- Generating model evaluation graphs
- Organizing a machine learning project in a reusable structure

---

## Dataset

The dataset used in this project is the **Breast Cancer Wisconsin Diagnostic Dataset**, available directly from `scikit-learn`.

It contains numerical measurements computed from digitized images of fine needle aspirate samples of breast masses.

### Dataset Details

| Item | Description |
|---|---|
| Dataset | Breast Cancer Wisconsin Diagnostic Dataset |
| Source | `sklearn.datasets.load_breast_cancer()` |
| Problem type | Binary classification |
| Number of samples | 569 |
| Number of features | 30 |
| Target classes | Malignant and Benign |
| Missing values | None |

### Target Variable

The target variable contains two classes:

```text
0 = malignant
1 = benign
````

In a medical context, malignant tumors are especially important because failing to detect a malignant case can be more serious than misclassifying a benign case.

---

## Features

The dataset contains 30 numerical features describing tumor cell nuclei. These include measurements such as:

* mean radius
* mean texture
* mean perimeter
* mean area
* mean smoothness
* mean compactness
* mean concavity
* mean concave points
* worst radius
* worst texture
* worst perimeter
* worst area
* worst concavity
* worst symmetry
* worst fractal dimension

These features are used by the model to classify whether a tumor is malignant or benign.

---

## Machine Learning Workflow

The project follows a structured machine learning workflow:

1. Load the Breast Cancer Wisconsin dataset
2. Convert the dataset into a pandas DataFrame
3. Separate input features and target labels
4. Split the data into training and testing sets
5. Train Logistic Regression on raw data
6. Apply preprocessing:

   * Standardization using `StandardScaler`
   * Skewness detection
   * Yeo-Johnson transformation using `PowerTransformer`
7. Train Logistic Regression on preprocessed data
8. Save both trained models using `joblib`
9. Evaluate both models using classification metrics
10. Generate and save visual evaluation graphs
11. Compare raw-data and preprocessed-data performance

---

## Preprocessing

Two versions of the dataset are used in this project.

### 1. Raw Data

The raw-data model is trained directly on the original feature values.

This model is useful as a baseline, but raw features have very different numerical ranges. For example, area-related features can have values in the hundreds or thousands, while smoothness and fractal-dimension features are much smaller.

Because Logistic Regression is sensitive to feature scaling, training on raw data can make optimization harder and coefficients less interpretable.

### 2. Preprocessed Data

The preprocessed-data model applies two major preprocessing steps.

#### Standard Scaling

All features are standardized using `StandardScaler`.

This transforms the data so that each feature has approximately:

```text
mean = 0
standard deviation = 1
```

This helps Logistic Regression train more effectively because all features are placed on a comparable scale.

#### Skewness Transformation

Highly skewed columns are detected using the training data only.

Columns with absolute skewness greater than `1` are transformed using:

```python
PowerTransformer(method="yeo-johnson")
```

The Yeo-Johnson transformation helps make feature distributions more symmetric.

This is important because skewed features can affect model behavior and make linear model coefficients harder to interpret.

---

## Models

This project trains and saves two Logistic Regression models.

### Raw-Data Model

```text
models/logistic_regression_model_raw_data.joblib
```

This model is trained directly on the original dataset without preprocessing.

### Preprocessed-Data Model

```text
models/logistic_regression_model_preprocessed_data.joblib
```

This model is trained after applying scaling and skewness transformation.

This is the preferred model for final interpretation because:

* features are on a comparable scale
* skewed distributions are reduced
* model optimization is more stable
* coefficients are more meaningful
* results are easier to explain

---

## Results

Both models were evaluated on the test set using accuracy, precision, recall, F1-score, confusion matrix, ROC curve, Precision-Recall curve, predicted probability distribution, and feature coefficient plots.

> Replace the values below with the exact values from your generated result files.

### Model Performance Comparison

| Model                                   |  Accuracy | Precision |    Recall |  F1-score | ROC-AUC | Average Precision |
| --------------------------------------- | --------: | --------: | --------: | --------: | ------: | ----------------: |
| Logistic Regression - Raw Data          | Add value | Add value | Add value | Add value |    1.00 |              1.00 |
| Logistic Regression - Preprocessed Data | Add value | Add value | Add value | Add value |    1.00 |              1.00 |

The ROC and Precision-Recall curves showed excellent class separation for both models. However, the preprocessed model is preferred because the raw-data model can be affected by feature-scale differences and may produce less reliable coefficient interpretation.

---

## Model Comparison

### Raw-Data Model

The raw-data model performs well, but it is trained on features with different units and scales.

This can cause two issues:

1. Logistic Regression may take longer to converge.
2. Feature coefficients become harder to compare directly.

In this project, the raw-data model was included to show why preprocessing is important for linear models.

### Preprocessed-Data Model

The preprocessed model uses standardized and transformed features.

This makes the model more reliable for interpretation because the coefficients are based on features that are placed on a more comparable scale.

Even when both models achieve similar predictive performance, the preprocessed model is more appropriate for final reporting because:

* the optimization process is cleaner
* feature coefficients are easier to compare
* the workflow follows better machine learning practice

---

## Evaluation Graphs

The project generates separate visualizations for both raw-data and preprocessed-data models.

### Preprocessed Data Results

```text
results/logistic_regression_preprocessed_data/
├── results_preprocessed_data.txt
├── confusion_matrix_preprocessed_data.png
├── roc_curve_preprocessed_data.png
├── precision_recall_curve_preprocessed_data.png
├── probability_distribution_preprocessed_data.png
└── feature_coefficients_preprocessed_data.png
```

### Raw Data Results

```text
results/logistic_regression_raw_data/
├── results_raw_data.txt
├── confusion_matrix_raw_data.png
├── roc_curve_raw_data.png
├── precision_recall_curve_raw_data.png
├── probability_distribution_raw_data.png
└── feature_coefficients_raw_data.png
```

---

## Interpretation of Evaluation Outputs

### Confusion Matrix

The confusion matrix shows correct and incorrect predictions for each class.

It helps identify:

* malignant tumors correctly predicted as malignant
* benign tumors correctly predicted as benign
* malignant tumors incorrectly predicted as benign
* benign tumors incorrectly predicted as malignant

In a medical classification problem, incorrectly predicting a malignant tumor as benign is especially important because it could represent a missed cancer case.

---

### ROC Curve

The ROC curve shows the relationship between:

```text
True Positive Rate
False Positive Rate
```

Both the raw-data and preprocessed-data models achieved a ROC-AUC score of approximately `1.00`, indicating excellent separation between malignant and benign tumors on the test set.

However, ROC performance alone is not enough to choose the best model. The preprocessed model is preferred because it is more stable and interpretable.

---

### Precision-Recall Curve

The Precision-Recall curve shows the relationship between:

```text
Precision
Recall
```

Both models achieved an Average Precision score of approximately `1.00`.

By default, the positive label is:

```text
1 = benign
```

For a more medical-focused evaluation, future work should also generate Precision-Recall and ROC curves using:

```text
0 = malignant
```

This would directly evaluate the model’s ability to detect malignant tumors.

---

### Feature Coefficient Plot

The feature coefficient plot shows how each feature affects the Logistic Regression model.

For Logistic Regression:

```text
positive coefficient = pushes prediction toward class 1
negative coefficient = pushes prediction toward class 0
larger absolute value = stronger model influence
smaller absolute value = weaker model influence
```

Since the dataset uses:

```text
0 = malignant
1 = benign
```

positive coefficients push predictions toward benign, while negative coefficients push predictions toward malignant.

The preprocessed model’s coefficient plot is more meaningful than the raw model’s coefficient plot because the features were scaled and transformed before training.

---

## Project Structure

```text
breast-cancer-classification/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── Breast_cancer.ipynb
│
├── notebook/
│   └── breast_cancer_analysis.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── train_model.py
│   └── evaluate_model.py
│
├── models/
│   ├── logistic_regression_model_preprocessed_data.joblib
│   └── logistic_regression_model_raw_data.joblib
│
├── results/
│   ├── logistic_regression_preprocessed_data/
│   │   ├── results_preprocessed_data.txt
│   │   ├── confusion_matrix_preprocessed_data.png
│   │   ├── roc_curve_preprocessed_data.png
│   │   ├── precision_recall_curve_preprocessed_data.png
│   │   ├── probability_distribution_preprocessed_data.png
│   │   └── feature_coefficients_preprocessed_data.png
│   │
│   └── logistic_regression_raw_data/
│       ├── results_raw_data.txt
│       ├── confusion_matrix_raw_data.png
│       ├── roc_curve_raw_data.png
│       ├── precision_recall_curve_raw_data.png
│       ├── probability_distribution_raw_data.png
│       └── feature_coefficients_raw_data.png
│
└── images/
```

---

## Repository Structure Review

The repository structure has improved and now includes important project folders such as:

* `src/`
* `results/`
* `models/`
* `requirements.txt`
* `.gitignore`

Recommended cleanup:

1. Move the final notebook fully into the notebook folder.
2. Rename `notebook/` to `notebooks/`, which is a more common convention.
3. Remove the duplicate root-level notebook if it is no longer needed.
4. Remove Python cache files such as `__pycache__/` from Git tracking.
5. Add common ignored files to `.gitignore`.

Recommended `.gitignore` content:

```gitignore
__pycache__/
*.pyc
.ipynb_checkpoints/
.env
.venv/
venv/
.DS_Store
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/AsadKhan37/breast-cancer-classification.git
```

Move into the project directory:

```bash
cd breast-cancer-classification
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is empty, install the main dependencies manually:

```bash
pip install pandas numpy scikit-learn matplotlib joblib
```

Recommended `requirements.txt`:

```text
pandas
numpy
scikit-learn
matplotlib
joblib
```

---

## How to Run

### Train the Models

Run:

```bash
python src/train_model.py
```

This trains and saves:

```text
models/logistic_regression_model_preprocessed_data.joblib
models/logistic_regression_model_raw_data.joblib
```

### Evaluate the Models

Run:

```bash
python src/evaluate_model.py
```

This generates model reports and graphs inside the `results/` folder.

---

## Source Code Overview

### `data_preprocessing.py`

Contains functions for:

* loading the dataset
* splitting features and target
* splitting train and test data
* standardizing features
* detecting skewed columns
* applying Yeo-Johnson transformation

Main functions:

```python
load_data()
load_and_preprocess_data()
```

---

### `train_model.py`

Trains two Logistic Regression models:

```python
train_logistic_regression_model_preprocessed_data()
train_logistic_regression_model_raw_data()
```

It also saves the trained models into the `models/` folder using `joblib`.

---

### `evaluate_model.py`

Evaluates both trained models and saves:

* classification report
* confusion matrix
* ROC curve
* Precision-Recall curve
* probability distribution plot
* feature coefficient plot

---

## Limitations

This project is a strong baseline classical machine learning project, but it still has some limitations:

* Only Logistic Regression is currently emphasized.
* No cross-validation has been implemented yet.
* No hyperparameter tuning has been performed.
* The model is evaluated on a single train-test split.
* The current workflow saves models, but does not yet save the full preprocessing pipeline.
* ROC and Precision-Recall curves currently focus on class `1`, which represents benign tumors.
* The project is not yet deployed as an application.
* The notebook and scripts can be further cleaned for production-style use.

---

## Future Work

Future improvements could include:

* Add cross-validation for more reliable performance estimation.
* Compare additional models such as:

  * Support Vector Machine
  * Random Forest
  * K-Nearest Neighbors
  * Gradient Boosting
* Add hyperparameter tuning using `GridSearchCV`.
* Build a complete scikit-learn `Pipeline`.
* Save preprocessing transformers along with the model.
* Generate ROC and Precision-Recall curves for the malignant class.
* Add a prediction script for new patient/tumor input data.
* Add model comparison tables automatically.
* Deploy the model using Streamlit or FastAPI.
* Add unit tests for preprocessing and training scripts.
* Improve the README with embedded result images.

---

## Key Takeaways

This project shows that Logistic Regression can perform very well on the Breast Cancer Wisconsin dataset.

The comparison between raw-data and preprocessed-data models highlights an important machine learning lesson: high performance alone is not the only factor when selecting a model. Preprocessing improves optimization stability and makes model coefficients easier to interpret.

The preprocessed Logistic Regression model is the preferred final model because it provides strong performance, better training behavior, and more meaningful feature interpretation.

---

## Conclusion

This project demonstrates a complete classical machine learning workflow for breast cancer classification.

It includes data preprocessing, model training, raw-versus-preprocessed model comparison, evaluation metrics, visualizations, and saved model files.

The final recommendation is to use the **preprocessed Logistic Regression model** because it combines strong predictive performance with better reliability and interpretability.

````

For your repo cleanup, I strongly recommend doing this before the next push:

```powershell
git rm -r --cached src/__pycache__
````

Then update `.gitignore`, commit, and push:

```powershell
git add README.md requirements.txt .gitignore
git commit -m "Update README with model comparison and project review"
git push origin main
```
