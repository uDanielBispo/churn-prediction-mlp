# utils.py - Funções utilitárias compartilhadas entre os módulos do projeto.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def load_data(path):
    """Carrega o dataset processado a partir do caminho especificado."""
    return pd.read_csv(path)


def split_data(df, target_col='target', test_size=0.2, random_state=42):
    """Separa features e target, e divide em conjuntos de treino e teste."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def compute_metrics(y_test, y_pred_test, y_train, y_pred_train):
    """Calcula as métricas de avaliação do modelo."""
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test, zero_division=0)
    test_precision = precision_score(y_test, y_pred_test, zero_division=0)
    test_recall = recall_score(y_test, y_pred_test, zero_division=0)
    overfitting = train_accuracy - test_accuracy

    return {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'test_f1_score': test_f1,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'overfitting': overfitting,
    }
