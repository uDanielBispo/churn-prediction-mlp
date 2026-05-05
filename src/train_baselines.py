# train_baselines.py — Treina e salva os modelos de baseline (Logística e Dummy).
#
# Faz duas coisas em paralelo:
#   1. Salva os .pkl em src/models/ para a API carregar localmente.
#   2. Loga métricas e artefatos no MLflow para rastreamento e registro de modelos.
#
# Execute com: python -m src.train_baselines

import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from src.pipeline import BINARY_COLS, NUMERICAL_COLS
from src.utils import setup_mlflow

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FEATURE_COLS = NUMERICAL_COLS + BINARY_COLS


def load_and_split(data_path: str):
    """Carrega o CSV e divide em treino/teste (80/20 estratificado)."""
    df = pd.read_csv(data_path)
    X = df[FEATURE_COLS]
    y = df['target']
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def train_and_save(model, name: str, experiment_name: str,
                   X_train, X_test, y_train, y_test, models_dir: str) -> None:
    """Treina o modelo, loga no MLflow e salva o .pkl em disco."""
    # Garante que o experimento exista no MLflow antes de iniciar o run.
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(name=experiment_name)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        print(f"{name} — Accuracy: {acc:.4f} | F1: {f1:.4f}")

        # Loga métricas — usadas pelo register_models.py para comparar com produção.
        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_f1_score", f1)
        mlflow.log_param("model_type", name)

        # Loga o artefato do modelo — necessário para registrar no MLflow Registry.
        mlflow.sklearn.log_model(model, "model")

    # Salva .pkl para a API carregar no ambiente local (sem MLflow).
    path = os.path.join(models_dir, f"{name}.pkl")
    joblib.dump(model, path)
    print(f"Modelo salvo em: {path}")


def train_baselines() -> None:
    setup_mlflow()

    data_path = os.path.join(ROOT_DIR, 'data', 'processed', 'telco_customer_churn_processed.csv')
    models_dir = os.path.join(ROOT_DIR, 'src', 'models')
    os.makedirs(models_dir, exist_ok=True)

    X_train, X_test, y_train, y_test = load_and_split(data_path)

    train_and_save(
        LogisticRegression(random_state=42, solver='liblinear', max_iter=1000),
        name='churn_prediction_logistic_regression_model',
        experiment_name='churn_prediction_logistic_regression',
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        models_dir=models_dir,
    )

    train_and_save(
        DummyClassifier(random_state=42, strategy='most_frequent'),
        name='churn_prediction_dummy_classifier_model',
        experiment_name='churn_prediction_dummy_classifier',
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        models_dir=models_dir,
    )

    print("\nModelos de baseline prontos.")


if __name__ == "__main__":
    train_baselines()
