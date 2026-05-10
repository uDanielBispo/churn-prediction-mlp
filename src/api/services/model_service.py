# model_service.py — carrega os modelos e executa a inferência.
#
# Os modelos são carregados uma única vez quando o módulo é importado
# (na inicialização da API), evitando o custo de carregamento a cada requisição.
#
# O comportamento muda conforme a variável de ambiente ENVIRONMENT:
#   - local (padrão): lê os .pkl do disco — sem dependência de servidor externo.
#   - production:     carrega do MLflow Registry da VPS — sem .pkl no disco.

import os

import joblib
import mlflow
import mlflow.pytorch
import mlflow.sklearn
import numpy as np
import pandas as pd
import torch

from src.api.core.logger import get_logger
from src.pipeline import OUTPUT_COLS

logger = get_logger(__name__)

# Lê o ambiente. Se não definido, assume local para não quebrar em desenvolvimento.
ENV = os.getenv("ENVIRONMENT", "local")

logger.info("Carregando modelos em memória...")

if ENV == "production":
    # Em produção, o MLflow Registry é a fonte da verdade.
    # MLFLOW_TRACKING_URI aponta para o servidor MLflow da VPS
    # (definido no docker-compose.prod.yml como http://mlflow:5000).
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

    dummy_model = mlflow.sklearn.load_model("models:/ChurnDummy/Production")
    logger.info("Modelo carregado: DummyClassifier (MLflow Registry)")

    logistic_model = mlflow.sklearn.load_model("models:/ChurnLogistic/Production")
    logger.info("Modelo carregado: LogisticRegression (MLflow Registry)")

    mlp_model = mlflow.pytorch.load_model("models:/ChurnMLP/Production")
    logger.info("Modelo carregado: MLP PyTorch (MLflow Registry)")

    # O preprocessor é versionado junto com a MLP — sempre carregados da mesma versão.
    preprocessor = mlflow.sklearn.load_model("models:/ChurnPreprocessor/Production")
    logger.info("Preprocessor carregado: StandardScaler (MLflow Registry)")

else:
    # Ambiente local: lê os .pkl gerados por make train.
    dummy_model = joblib.load("src/models/churn_prediction_dummy_classifier_model.pkl")
    logger.info("Modelo carregado: DummyClassifier")

    logistic_model = joblib.load("src/models/churn_prediction_logistic_regression_model.pkl")
    logger.info("Modelo carregado: LogisticRegression")

    mlp_model = joblib.load("src/models/churn_prediction_mlp_pytorch_model.pkl")
    logger.info("Modelo carregado: MLP PyTorch")

    preprocessor = joblib.load("src/models/preprocessor.pkl")
    logger.info("Preprocessor carregado: StandardScaler")


def _build_features(data) -> np.ndarray:
    """Converte os campos do schema Pydantic em um array NumPy para o modelo.

    A ordem dos campos deve corresponder à ordem de OUTPUT_COLS definida em pipeline.py,
    que é a mesma ordem usada durante o treino.
    """
    return np.array([[
        data.Tenure_Months,
        data.Monthly_Charges,
        data.Gender_Male,
        data.Partner_Yes,
        data.Dependents_Yes,
        data.Phone_Service_Yes,
        data.Multiple_Lines_No_phone_service,
        data.Multiple_Lines_Yes,
        data.Internet_Service_Fiber_optic,
        data.Internet_Service_No,
        data.Online_Security_No_internet_service,
        data.Online_Security_Yes,
        data.Online_Backup_No_internet_service,
        data.Online_Backup_Yes,
        data.Device_Protection_No_internet_service,
        data.Device_Protection_Yes,
        data.Tech_Support_No_internet_service,
        data.Tech_Support_Yes,
        data.Streaming_TV_No_internet_service,
        data.Streaming_TV_Yes,
        data.Streaming_Movies_No_internet_service,
        data.Streaming_Movies_Yes,
        data.Contract_One_year,
        data.Contract_Two_year,
        data.Paperless_Billing_Yes,
        data.Payment_Method_Credit_card_automatic,
        data.Payment_Method_Electronic_check,
        data.Payment_Method_Mailed_check,
    ]])


def dummy_predict(data) -> int:
    """Realiza previsão usando o DummyClassifier (modelo baseline ingênuo).

    O DummyClassifier sempre prediz a classe mais frequente nos dados de treino,
    servindo como linha de base mínima para comparação.
    """
    features = _build_features(data)
    return int(dummy_model.predict(features)[0])


def logistic_predict(data) -> int:
    """Realiza previsão usando a Regressão Logística.

    Modelo linear treinado com sklearn, mais interpretável que a MLP.
    """
    features = _build_features(data)
    return int(logistic_model.predict(features)[0])


def mlp_predict(data) -> int:
    """Realiza previsão usando a rede neural MLP com PyTorch.

    model.eval() desativa o Dropout e coloca o BatchNorm em modo de inferência.
    torch.no_grad() desativa o cálculo de gradientes, pois não há treino aqui —
    isso economiza memória e acelera a inferência.
    torch.sigmoid() converte o logit bruto da rede em probabilidade [0, 1].

    O preprocessor aplica o mesmo StandardScaler usado no treino — sem isso,
    Tenure Months e Monthly Charges chegam na escala errada e a rede prevê mal.
    """
    features = _build_features(data)

    # Cria DataFrame com os nomes de coluna que o preprocessor espera.
    df = pd.DataFrame(features, columns=OUTPUT_COLS)
    features_scaled = preprocessor.transform(df)

    mlp_model.eval()
    with torch.no_grad():
        x = torch.tensor(features_scaled, dtype=torch.float32)
        logit = mlp_model(x)
        prob = torch.sigmoid(logit)
        return int((prob > 0.5).item())
