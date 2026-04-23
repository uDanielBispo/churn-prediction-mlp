# model_service.py — responsável por carregar os modelos e executar inferência.
#
# Os modelos são carregados uma única vez quando o módulo é importado
# (na inicialização da API), evitando o custo de carregamento a cada requisição.

import joblib
import numpy as np
import torch

from src.api.core.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Carregamento dos modelos em disco
# ---------------------------------------------------------------------------
# joblib.load() lê o arquivo .pkl e reconstrói o objeto Python original
# (seja um modelo sklearn ou um modelo PyTorch serializado).

logger.info("Carregando modelos em memória...")

dummy_model = joblib.load("src/models/churn_prediction_dummy_classifier_model.pkl")
logger.info("Modelo carregado: DummyClassifier")

logistic_model = joblib.load("src/models/churn_prediction_logistic_regression_model.pkl")
logger.info("Modelo carregado: LogisticRegression")

mlp_model = joblib.load("src/models/churn_prediction_mlp_pytorch_model.pkl")
logger.info("Modelo carregado: MLP PyTorch")


def _build_features(data) -> np.ndarray:
    """Converte os campos do schema Pydantic em um array NumPy para o modelo.

    Centraliza a extração de features em um único lugar, eliminando a duplicação
    que existia antes em cada função de predição separada.

    A ordem dos campos deve corresponder à ordem das colunas usadas no treino.
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
    prediction = dummy_model.predict(features)
    return int(prediction[0])


def logistic_predict(data) -> int:
    """Realiza previsão usando a Regressão Logística.

    Modelo linear treinado com sklearn, mais interpretável que a MLP.
    """
    features = _build_features(data)
    prediction = logistic_model.predict(features)
    return int(prediction[0])


def mlp_predict(data) -> int:
    """Realiza previsão usando a rede neural MLP com PyTorch.

    model.eval() desativa o Dropout e coloca o BatchNorm em modo de inferência.
    torch.no_grad() desativa o cálculo de gradientes, pois não há treino aqui —
    isso economiza memória e acelera a inferência.
    torch.sigmoid() converte o logit bruto da rede em probabilidade [0, 1].
    """
    features = _build_features(data)

    mlp_model.eval()
    with torch.no_grad():
        x = torch.tensor(features, dtype=torch.float32)
        logit = mlp_model(x)
        prob = torch.sigmoid(logit)
        return int((prob > 0.5).item())
