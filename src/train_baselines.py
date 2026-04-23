# train_baselines.py — Treina e salva os modelos de baseline (Logística e Dummy).
#
# A API espera três arquivos .pkl em src/models/:
#   - churn_prediction_logistic_regression_model.pkl
#   - churn_prediction_dummy_classifier_model.pkl
#   - churn_prediction_mlp_pytorch_model.pkl  (gerado por train_mlp.py)
#
# Execute com: python -m src.train_baselines

import os

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from src.pipeline import BINARY_COLS, NUMERICAL_COLS

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# As mesmas colunas usadas pela API em _build_features(), na mesma ordem.
FEATURE_COLS = NUMERICAL_COLS + BINARY_COLS


def load_and_split(data_path: str):
    """Carrega o CSV e divide em treino/teste (80/20 estratificado)."""
    df = pd.read_csv(data_path)
    X = df[FEATURE_COLS]
    y = df['target']
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def train_and_save(model, name: str, X_train, X_test, y_train, y_test, models_dir: str) -> None:
    """Treina o modelo, imprime métricas e salva em disco."""
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    print(f"{name} — Accuracy: {acc:.4f} | F1: {f1:.4f}")

    path = os.path.join(models_dir, f"{name}.pkl")
    joblib.dump(model, path)
    print(f"Modelo salvo em: {path}")


def train_baselines() -> None:
    data_path = os.path.join(ROOT_DIR, 'data', 'processed', 'telco_customer_churn_processed.csv')
    models_dir = os.path.join(ROOT_DIR, 'src', 'models')
    os.makedirs(models_dir, exist_ok=True)

    X_train, X_test, y_train, y_test = load_and_split(data_path)

    train_and_save(
        LogisticRegression(random_state=42, solver='liblinear', max_iter=1000),
        'churn_prediction_logistic_regression_model',
        X_train, X_test, y_train, y_test, models_dir,
    )

    train_and_save(
        DummyClassifier(random_state=42, strategy='most_frequent'),
        'churn_prediction_dummy_classifier_model',
        X_train, X_test, y_train, y_test, models_dir,
    )

    print("\nModelos de baseline prontos.")


if __name__ == "__main__":
    train_baselines()
