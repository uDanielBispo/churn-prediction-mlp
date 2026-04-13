# train.py - Responsável por ler os dados e treinar o modelo de predição de churn.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import mlflow
import mlflow.sklearn
import os

# lendo o dataset processado
df = pd.read_csv('data/processed/telco_customer_churn_processed.csv')

X = df.drop(columns=['target'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# configurando o mlflow
mlflow_db = os.path.join(os.getcwd(), 'mlflow.db')
mlruns_dir = os.path.join(os.getcwd(), 'mlruns')

mlflow.set_tracking_uri(f'sqlite:///{mlflow_db}')
mlflow.set_registry_uri(f'file://{mlruns_dir}')

# experimento 1: logistic regression
experiment_name = 'chrun_prediction_logistic_regression'

if mlflow.get_experiment_by_name(experiment_name) is None:
    mlflow.create_experiment(name=experiment_name, artifact_location=f'file://{mlruns_dir}')

mlflow.set_experiment(experiment_name)

with mlflow.start_run(run_name=experiment_name):
    model = LogisticRegression(random_state=42, solver='liblinear')
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test)
    test_precision = precision_score(y_test, y_pred_test)
    test_recall = recall_score(y_test, y_pred_test)
    overfitting = train_accuracy - test_accuracy

    print(f'Train Accuracy: {train_accuracy:.4f}')
    print(f'Test Accuracy:  {test_accuracy:.4f}')
    print(f'Test F1 Score:  {test_f1:.4f}')
    print(f'Overfitting:    {overfitting:.4f}')

    mlflow.log_param('model_type', 'Logistic Regression')
    mlflow.log_metric('train_accuracy', train_accuracy)
    mlflow.log_metric('test_accuracy', test_accuracy)
    mlflow.log_metric('test_f1_score', test_f1)
    mlflow.log_metric('test_precision', test_precision)
    mlflow.log_metric('test_recall', test_recall)
    mlflow.log_metric('overfitting', overfitting)

    mlflow.log_artifact('data/processed/telco_customer_churn_processed.csv', artifact_path='dataset')
    mlflow.sklearn.log_model(model, name='model')

    mlflow.set_tag('stage', 'baseline')
    mlflow.set_tag('dataset', 'telco_churn_processed')

# experimento 2: dummy classifier
experiment_name = 'chrun_prediction_dummy_classifier'

if mlflow.get_experiment_by_name(experiment_name) is None:
    mlflow.create_experiment(name=experiment_name, artifact_location=f'file://{mlruns_dir}')

mlflow.set_experiment(experiment_name)

with mlflow.start_run(run_name=experiment_name):
    dummy = DummyClassifier(random_state=42, strategy='most_frequent')
    dummy.fit(X_train, y_train)

    y_pred_train = dummy.predict(X_train)
    y_pred_test = dummy.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test, zero_division=0)
    test_precision = precision_score(y_test, y_pred_test, zero_division=0)
    test_recall = recall_score(y_test, y_pred_test, zero_division=0)
    overfitting = train_accuracy - test_accuracy

    print(f'Train Accuracy: {train_accuracy:.4f}')
    print(f'Test Accuracy:  {test_accuracy:.4f}')
    print(f'Test F1 Score:  {test_f1:.4f}')
    print(f'Overfitting:    {overfitting:.4f}')

    mlflow.log_param('model_type', 'Dummy Classifier')
    mlflow.log_metric('train_accuracy', train_accuracy)
    mlflow.log_metric('test_accuracy', test_accuracy)
    mlflow.log_metric('test_f1_score', test_f1)
    mlflow.log_metric('test_precision', test_precision)
    mlflow.log_metric('test_recall', test_recall)
    mlflow.log_metric('overfitting', overfitting)

    mlflow.log_artifact('data/processed/telco_customer_churn_processed.csv', artifact_path='dataset')
    mlflow.sklearn.log_model(dummy, name='model')

    mlflow.set_tag('stage', 'baseline')
    mlflow.set_tag('dataset', 'telco_churn_processed')
