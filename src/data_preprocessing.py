from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer


def load_and_preprocess_data():
    data = load_breast_cancer()

    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    standard_scaler = StandardScaler()
    X_train_scaled = standard_scaler.fit_transform(X_train)
    X_test_scaled = standard_scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=data.feature_names)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=data.feature_names)

    skewed_cols = X_train_scaled.skew()[X_train_scaled.skew().abs() > 1].index
    pt = PowerTransformer(method='yeo-johnson')
    X_train_scaled[skewed_cols] = pt.fit_transform(X_train_scaled[skewed_cols])
    X_test_scaled[skewed_cols] = pt.transform(X_test_scaled[skewed_cols])

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=data.feature_names)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=data.feature_names)

    return X_train_scaled, X_test_scaled, y_train, y_test