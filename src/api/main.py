# main.py — ponto de entrada da API FastAPI.
# Inicializa o app, registra o middleware de latência e inclui as rotas.

import time

from fastapi import FastAPI, Request

from src.api.core.logger import get_logger
from src.api.routes import router

logger = get_logger(__name__)

app = FastAPI(title="Churn Prediction API")

@app.middleware("http")
async def log_request_latency(request: Request, call_next):
    """Middleware que intercepta todas as requisições HTTP para registrar latência.

    Um middleware em FastAPI funciona como um "porteiro": toda requisição passa
    por ele antes de chegar à rota e toda resposta passa por ele antes de sair.

    O que este middleware faz:
      1. Registra o momento de início da requisição.
      2. Deixa a requisição seguir normalmente para a rota correspondente.
      3. Após receber a resposta, calcula o tempo decorrido em milissegundos.
      4. Loga o método HTTP, caminho, status code e tempo de resposta.

    Exemplo de saída no console:
      2026-04-23 17:30:01 | INFO | src.api.main | POST /predict → 200 (4.3ms)
    """
    inicio = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - inicio) * 1000
    logger.info(f"{request.method} {request.url.path} → {response.status_code} ({elapsed_ms:.1f}ms)")
    return response


app.include_router(router)
