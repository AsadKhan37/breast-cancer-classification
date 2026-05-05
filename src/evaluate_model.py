import os
import train_model
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


def evaluate_model():
    model, X_test_scaled, y_test = train_model.train_logistic_regression_model()

    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return accuracy, report, cm


def save_results():
    save_folder = "results/logistic_regression"
    os.makedirs(save_folder, exist_ok=True)

    accuracy, report, cm = evaluate_model()

    results_path = os.path.join(save_folder, "results.txt")

    with open(results_path, "w") as f:
        f.write(f"Accuracy: {accuracy}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(cm))

    fig, ax = plt.subplots(figsize=(6, 5))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, values_format="d")

    ax.set_title("Logistic Regression Confusion Matrix")

    plt.tight_layout()

    cm_path = os.path.join(save_folder, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Results saved to: {results_path}")
    print(f"Confusion matrix graph saved to: {cm_path}")


if __name__ == "__main__":
    save_results()