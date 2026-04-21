from fastapi import FastAPI
from src.api.routes import router

app = FastAPI(title="Churn Prediction API")

app.include_router(router)
