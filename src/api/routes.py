from fastapi import APIRouter
from src.api.schemas import CustomerData
from src.api.services.model_service import logistic_predict, dummy_predict, mlp_predict

router = APIRouter()

@router.get("/health")
def health():
    return {"status": "ok"}

@router.post("/predict")
def predict(data: CustomerData, model_type="logistic"):
    print(data)
    if model_type == "dummy":
        return dummy_predict(data)

    if model_type == "mlp":
        return mlp_predict(data)

    return logistic_predict(data)

