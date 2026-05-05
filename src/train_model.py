import os
import joblib
import data_preprocessing

from sklearn.linear_model import LogisticRegression


def train_logistic_regression_model_preprocessed_data(save_model=True):
    X_train_scaled, X_test_scaled, y_train, y_test = (
        data_preprocessing.load_and_preprocess_data()
    )

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    if save_model:
        save_folder = "models"
        os.makedirs(save_folder, exist_ok=True)

        model_path = os.path.join(
            save_folder,
            "logistic_regression_model_preprocessed_data.joblib"
        )

        joblib.dump(model, model_path)

        print(f"Model saved to: {model_path}")

    return model, X_test_scaled, y_test


def train_logistic_regression_model_raw_data(save_model=True):
    X_train, X_test, y_train, y_test = data_preprocessing.load_data()

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    if save_model:
        save_folder = "models"
        os.makedirs(save_folder, exist_ok=True)

        model_path = os.path.join(
            save_folder,
            "logistic_regression_model_raw_data.joblib"
        )

        joblib.dump(model, model_path)

        print(f"Model saved to: {model_path}")

    return model, X_test, y_test


if __name__ == "__main__":
    train_logistic_regression_model_preprocessed_data(save_model=True)
    train_logistic_regression_model_raw_data(save_model=True)