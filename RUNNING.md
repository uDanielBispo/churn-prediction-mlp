# Como Rodar o Projeto

Este arquivo explica ao professor como executar o projeto completo com Docker, acessar a API, a documentação e o MLflow.

## Objetivo

Executar a solução de churn de ponta a ponta em um ambiente isolado:
- API FastAPI para inferência
- MLflow UI para visualização de experimentos
- Treinamento PyTorch para registrar métricas e artefatos

## Requisitos

- Docker Desktop instalado e em execução
- Repositório clonado no host

## Passos para rodar

1. Abra o Docker Desktop.
2. Navegue até a raiz do projeto:
   ```powershell
   cd "./churn-prediction-mlp"
   ```
3. Suba a API e o MLflow:
   ```powershell
   docker compose up -d api mlflow
   ```
4. Rode o treinamento para gerar experimentos e artefatos:
   ```powershell
   docker compose run --rm train
   ```

> Se os modelos já estiverem treinados e salvos em `src/models/`, o passo 4 é opcional.

## URLs de acesso

- API: `http://localhost:8000`
- Documentação da API: `http://localhost:8000/docs`
- MLflow UI: `http://localhost:5000`

## O que cada serviço faz

- `api` → executa a API FastAPI na porta `8000`
- `mlflow` → executa o MLflow UI na porta `5000`
- `train` → executa `python -m src.train_mlp` e registra resultados em `mlflow.db` e `mlruns/`

## Testes rápidos da API

### Verificar se a API está no ar

```powershell
curl http://localhost:8000/health
```

Resposta esperada:

```json
{"status":"ok"}
```

### Fazer uma predição com o modelo MLP

```powershell
curl -X POST http://localhost:8000/predict?model_type=mlp \
  -H "Content-Type: application/json" \
  -d '{
    "Tenure_Months": 12,
    "Monthly_Charges": 70.3,
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
    "Device_Protection_Yes": 1,
    "Tech_Support_No_internet_service": 0,
    "Tech_Support_Yes": 0,
    "Streaming_TV_No_internet_service": 0,
    "Streaming_TV_Yes": 1,
    "Streaming_Movies_No_internet_service": 0,
    "Streaming_Movies_Yes": 1,
    "Contract_One_year": 0,
    "Contract_Two_year": 0,
    "Paperless_Billing_Yes": 1,
    "Payment_Method_Credit_card_automatic": 0,
    "Payment_Method_Electronic_check": 1,
    "Payment_Method_Mailed_check": 0
  }'
```

Resposta esperada:

```json
{"churn": 0}
```

## Testes automatizados

Para validar o projeto com pytest:

```powershell
pytest tests/ -v
```

## Observações para o avaliador

Este fluxo cobre os principais requisitos do Tech Challenge Fase 01:

- execução reprodutível em Docker
- API funcional com FastAPI e validação de entrada
- MLflow para rastreamento de experimentos
- modelo MLP em PyTorch comparado com baselines
- testes automatizados

## Parar os serviços

```powershell
docker compose down
```
