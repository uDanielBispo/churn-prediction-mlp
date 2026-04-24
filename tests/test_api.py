# test_api.py — testes dos endpoints da API FastAPI.
#
# Usa TestClient do FastAPI/Starlette, que simula requisições HTTP sem precisar
# subir um servidor real. (equivalente ao MockMvc do Spring Boot)
# você chama client.get("/health") e ele retorna uma Response com status code,
# headers e body — tudo em memória, sem abrir porta TCP.
#
# Problema: model_service.py carrega os modelos .pkl ao ser importado.
# Se os arquivos não existirem, o import falha. Solução: usar unittest.mock
# para substituir joblib.load por um objeto simulado ANTES da importação.
# Isso garante que os testes de API rodem mesmo sem os modelos treinados.

import sys
from unittest.mock import MagicMock, patch

import pytest
import torch

# ---------------------------------------------------------------------------
# Setup dos mocks ANTES de importar o app
# ---------------------------------------------------------------------------
# Remove quaisquer módulos da API já carregados em memória pelo pytest.
# Sem isso, se outro arquivo de teste importou src.api primeiro, o patch
# não teria efeito — os modelos reais já estariam carregados.
for _modulo in list(sys.modules.keys()):
    if "model_service" in _modulo or "src.api" in _modulo:
        del sys.modules[_modulo]

# Mock para modelos sklearn (DummyClassifier, LogisticRegression).
# .predict() retorna uma lista com um inteiro, imitando o comportamento real.
_mock_sklearn = MagicMock()
_mock_sklearn.predict.return_value = [0]

# Mock para o modelo MLP PyTorch.
# Quando chamado como função (model(x)), retorna um tensor com um logit.
# torch.sigmoid(tensor([0.0])) = 0.5, que com threshold 0.5 resulta em churn=0.
_mock_mlp = MagicMock()
_mock_mlp.return_value = torch.tensor([0.0])


def _joblib_load_mock(path):
    """Retorna o mock correto baseado no nome do arquivo .pkl."""
    if "mlp" in str(path):
        return _mock_mlp
    return _mock_sklearn


# O patch é aplicado no momento da importação para interceptar os joblib.load
# que acontecem no topo de model_service.py.
with patch("joblib.load", side_effect=_joblib_load_mock):
    from fastapi.testclient import TestClient

    from src.api.main import app

client = TestClient(app)

# ---------------------------------------------------------------------------
# Payload de exemplo com todos os campos obrigatórios do schema CustomerData
# ---------------------------------------------------------------------------
CLIENTE_VALIDO = {
    "Tenure_Months": 24.0,
    "Monthly_Charges": 65.5,
    "Gender_Male": 1,
    "Partner_Yes": 0,
    "Dependents_Yes": 0,
    "Phone_Service_Yes": 1,
    "Multiple_Lines_No_phone_service": 0,
    "Multiple_Lines_Yes": 1,
    "Internet_Service_Fiber_optic": 1,
    "Internet_Service_No": 0,
    "Online_Security_No_internet_service": 0,
    "Online_Security_Yes": 0,
    "Online_Backup_No_internet_service": 0,
    "Online_Backup_Yes": 1,
    "Device_Protection_No_internet_service": 0,
    "Device_Protection_Yes": 0,
    "Tech_Support_No_internet_service": 0,
    "Tech_Support_Yes": 0,
    "Streaming_TV_No_internet_service": 0,
    "Streaming_TV_Yes": 1,
    "Streaming_Movies_No_internet_service": 0,
    "Streaming_Movies_Yes": 0,
    "Contract_One_year": 0,
    "Contract_Two_year": 0,
    "Paperless_Billing_Yes": 1,
    "Payment_Method_Credit_card_automatic": 0,
    "Payment_Method_Electronic_check": 1,
    "Payment_Method_Mailed_check": 0,
}


# ---------------------------------------------------------------------------
# Testes do endpoint GET /health
# ---------------------------------------------------------------------------

def test_health_retorna_status_200():
    """Verifica que o endpoint de health check responde com sucesso."""
    response = client.get("/health")
    assert response.status_code == 200


def test_health_retorna_corpo_correto():
    """Verifica que o corpo da resposta de /health é {'status': 'ok'}."""
    response = client.get("/health")
    assert response.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# Testes do endpoint POST /predict
# ---------------------------------------------------------------------------

def test_predict_logistic_retorna_200():
    """Verifica que /predict com model_type=logistic retorna 200."""
    response = client.post("/predict?model_type=logistic", json=CLIENTE_VALIDO)
    assert response.status_code == 200


def test_predict_dummy_retorna_200():
    """Verifica que /predict com model_type=dummy retorna 200."""
    response = client.post("/predict?model_type=dummy", json=CLIENTE_VALIDO)
    assert response.status_code == 200


def test_predict_mlp_retorna_200():
    """Verifica que /predict com model_type=mlp retorna 200."""
    response = client.post("/predict?model_type=mlp", json=CLIENTE_VALIDO)
    assert response.status_code == 200


def test_predict_retorna_campo_churn():
    """Verifica que a resposta de /predict contém o campo 'churn'."""
    response = client.post("/predict", json=CLIENTE_VALIDO)
    assert "churn" in response.json()


def test_predict_churn_e_valor_binario():
    """Verifica que o valor de 'churn' na resposta é 0 ou 1."""
    response = client.post("/predict", json=CLIENTE_VALIDO)
    churn = response.json()["churn"]
    assert churn in (0, 1), f"Valor de churn inesperado: {churn}"


def test_predict_sem_dados_retorna_422():
    """Verifica que uma requisição sem body retorna erro de validação (422).

    O Pydantic valida automaticamente os campos obrigatórios do CustomerData.
    Se o body estiver ausente ou incompleto, o FastAPI retorna 422 Unprocessable
    Entity — equivalente ao @Valid com BindingResult do Spring Boot.
    """
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_predict_dados_parciais_retorna_422():
    """Verifica que dados com apenas alguns campos retorna 422."""
    dados_incompletos = {"Tenure_Months": 12.0, "Monthly_Charges": 50.0}
    response = client.post("/predict", json=dados_incompletos)
    assert response.status_code == 422


def test_predict_model_type_invalido_usa_logistic_como_padrao():
    """Verifica que um model_type desconhecido cai no modelo logístico (padrão).

    A rota tem model_type='logistic' como default. Qualquer valor desconhecido
    resulta no else da função predict(), que chama logistic_predict().
    Esperamos 200 (não 400), pois não é um erro — é o comportamento definido.
    """
    response = client.post("/predict?model_type=inexistente", json=CLIENTE_VALIDO)
    assert response.status_code == 200
