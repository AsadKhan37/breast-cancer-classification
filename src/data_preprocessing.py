from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer
import pandas as pd


def load_data():
    data = load_breast_cancer()

    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test


def load_and_preprocess_data():
    X_train, X_test, y_train, y_test = load_data()

    standard_scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(
        standard_scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )

    X_test_scaled = pd.DataFrame(
        standard_scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    skewed_cols = X_train_scaled.skew()[
        X_train_scaled.skew().abs() > 1
    ].index

    pt = PowerTransformer(method="yeo-johnson")

    X_train_scaled[skewed_cols] = pt.fit_transform(
        X_train_scaled[skewed_cols]
    )

    X_test_scaled[skewed_cols] = pt.transform(
        X_test_scaled[skewed_cols]
    )

    return X_train_scaled, X_test_scaled, y_train, y_test