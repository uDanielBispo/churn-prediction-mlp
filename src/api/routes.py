# routes.py — define os endpoints disponíveis na API.

from fastapi import APIRouter

from src.api.core.logger import get_logger
from src.api.schemas import CustomerData
from src.api.services.model_service import dummy_predict, logistic_predict, mlp_predict

router = APIRouter()
logger = get_logger(__name__)


@router.get("/health")
def health():
    """Endpoint de verificação de saúde da API.

    Usado por ferramentas de monitoramento para confirmar que o serviço está no ar.
    Retorna 200 com status 'ok' se a API estiver funcionando.
    """
    return {"status": "ok"}


@router.post("/predict")
def predict(data: CustomerData, model_type: str = "logistic"):
    """Recebe os dados de um cliente e retorna a previsão de churn.

    O parâmetro 'model_type' define qual modelo será usado:
      - 'logistic' (padrão): Regressão Logística
      - 'dummy': DummyClassifier (baseline ingênuo)
      - 'mlp': Rede Neural MLP com PyTorch

    Retorna 0 (não vai dar churn) ou 1 (vai dar churn).
    """
    logger.info(f"Requisição de previsão recebida | model_type={model_type}")

    if model_type == "dummy":
        resultado = dummy_predict(data)
    elif model_type == "mlp":
        resultado = mlp_predict(data)
    else:
        resultado = logistic_predict(data)

    logger.info(f"Previsão gerada | model_type={model_type} | churn={resultado}")
    return {"churn": resultado}
