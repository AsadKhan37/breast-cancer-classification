import os
import train_model
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay
)


def evaluate_model(model_type):
    if model_type == "preprocessed_data":
        model, X_test, y_test = train_model.train_logistic_regression_model_preprocessed_data()
    elif model_type == "raw_data":
        model, X_test, y_test = train_model.train_logistic_regression_model_raw_data()
    else:
        raise ValueError("model_type must be either 'preprocessed_data' or 'raw_data'")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return model, X_test, y_test, y_pred, y_proba, accuracy, report, cm


def save_confusion_matrix(cm, save_folder, model_type):
    fig, ax = plt.subplots(figsize=(6, 5))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, values_format="d")

    ax.set_title(f"Confusion Matrix - {model_type}")
    plt.tight_layout()

    path = os.path.join(save_folder, f"confusion_matrix_{model_type}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return path


def save_roc_curve(model, X_test, y_test, save_folder, model_type):
    fig, ax = plt.subplots(figsize=(6, 5))

    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)

    ax.set_title(f"ROC Curve - {model_type}")
    plt.tight_layout()

    path = os.path.join(save_folder, f"roc_curve_{model_type}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return path


def save_precision_recall_curve(model, X_test, y_test, save_folder, model_type):
    fig, ax = plt.subplots(figsize=(6, 5))

    PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=ax)

    ax.set_title(f"Precision-Recall Curve - {model_type}")
    plt.tight_layout()

    path = os.path.join(save_folder, f"precision_recall_curve_{model_type}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return path


def save_probability_distribution(y_proba, save_folder, model_type):
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.hist(y_proba, bins=20)

    ax.set_title(f"Predicted Probability Distribution - {model_type}")
    ax.set_xlabel("Predicted Probability for Class 1")
    ax.set_ylabel("Frequency")

    plt.tight_layout()

    path = os.path.join(save_folder, f"probability_distribution_{model_type}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return path


def save_feature_coefficients(model, X_test, save_folder, model_type):
    coefficients = model.coef_[0]
    feature_names = X_test.columns

    sorted_idx = abs(coefficients).argsort()

    fig, ax = plt.subplots(figsize=(8, 10))

    ax.barh(feature_names[sorted_idx], coefficients[sorted_idx])

    ax.set_title(f"Logistic Regression Feature Coefficients - {model_type}")
    ax.set_xlabel("Coefficient Value")
    ax.set_ylabel("Features")

    plt.tight_layout()

    path = os.path.join(save_folder, f"feature_coefficients_{model_type}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return path


def save_results(model_type):
    save_folder = f"results/logistic_regression_{model_type}"
    os.makedirs(save_folder, exist_ok=True)

    model, X_test, y_test, y_pred, y_proba, accuracy, report, cm = evaluate_model(model_type)

    results_path = os.path.join(save_folder, f"results_{model_type}.txt")

    with open(results_path, "w") as f:
        f.write(f"Accuracy: {accuracy}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(cm))

    confusion_matrix_path = save_confusion_matrix(cm, save_folder, model_type)
    roc_curve_path = save_roc_curve(model, X_test, y_test, save_folder, model_type)
    precision_recall_path = save_precision_recall_curve(model, X_test, y_test, save_folder, model_type)
    probability_distribution_path = save_probability_distribution(y_proba, save_folder, model_type)
    feature_coefficients_path = save_feature_coefficients(model, X_test, save_folder, model_type)

    print(f"Results saved to: {results_path}")
    print(f"Confusion matrix saved to: {confusion_matrix_path}")
    print(f"ROC curve saved to: {roc_curve_path}")
    print(f"Precision-recall curve saved to: {precision_recall_path}")
    print(f"Probability distribution saved to: {probability_distribution_path}")
    print(f"Feature coefficients saved to: {feature_coefficients_path}")


if __name__ == "__main__":
    save_results("preprocessed_data")
    save_results("raw_data")