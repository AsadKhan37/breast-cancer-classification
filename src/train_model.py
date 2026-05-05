import data_preprocessing
from sklearn.linear_model import LogisticRegression

def train_logistic_regression_model():
    X_train_scaled, X_test_scaled, y_train, y_test = data_preprocessing.load_and_preprocess_data()

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    return model, X_test_scaled, y_test