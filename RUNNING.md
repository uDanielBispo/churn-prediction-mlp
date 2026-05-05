# Como Rodar o Projeto

Este arquivo explica como executar o projeto completo com Docker, acessar a API, a documentação e o MLflow.

## Objetivo

Executar a solução de churn de ponta a ponta em um ambiente isolado:
- API FastAPI para inferência
- MLflow UI para visualização de experimentos
- Treinamento PyTorch para registrar métricas e artefatos

## Requisitos

- Docker Desktop instalado e em execução
- Repositório clonado no host

---

## Ambiente local (desenvolvimento)

### Passos para rodar

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

   > O serviço `train` executa `make train`, que treina os modelos de baseline
   > (Logística + Dummy) e a MLP em sequência, salvando os `.pkl` em `src/models/`
   > e logando métricas no MLflow local.

5. Para parar os serviços:
   ```powershell
   docker compose down
   ```

### URLs de acesso

- API: `http://localhost:8000`
- Documentação da API: `http://localhost:8000/docs`
- MLflow UI: `http://localhost:5000`

### O que cada serviço faz

- `api` → executa a API FastAPI na porta `8000`
- `mlflow` → executa o MLflow UI na porta `5000`
- `train` → treina Logística, Dummy e MLP; registra resultados no MLflow local

---

## Ambiente de produção (VPS)

Usa `docker-compose.prod.yml`, que não inclui o serviço de treino — o treino acontece
automaticamente no pipeline CI/CD do GitHub Actions a cada merge na branch `main`.

### Primeira configuração na VPS

Execute estes passos uma única vez ao configurar a VPS:

```bash
# 1. Subir o MLflow persistente
docker compose -f docker-compose.prod.yml up -d mlflow

# 2. Treinar os modelos e registrar no MLflow Registry
source .venv/bin/activate
MLFLOW_TRACKING_URI=http://localhost:5000 make train
MLFLOW_TRACKING_URI=http://localhost:5000 make register

# 3. Subir a API (carrega os modelos do MLflow Registry)
docker compose -f docker-compose.prod.yml up -d api
```

### Após a primeira configuração

Todo push na branch `main` dispara o pipeline CI/CD automaticamente:

```
push → lint → testes → treino na VPS → registro no MLflow → restart da API
```

Nenhuma intervenção manual é necessária.

### Segredos necessários no GitHub Actions

Configure em **Settings → Secrets and variables → Actions** do repositório:

| Secret | Exemplo | Descrição |
|---|---|---|
| `VPS_HOST` | `123.45.67.89` | IP ou hostname da VPS |
| `VPS_USER` | `ubuntu` | Usuário SSH |
| `VPS_SSH_KEY` | `-----BEGIN...` | Chave privada SSH |
| `VPS_PROJECT_PATH` | `/home/ubuntu/churn-prediction-mlp` | Caminho do projeto na VPS |

---

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

**macOS/Linux:**
```bash
curl -X POST "http://localhost:8000/predict?model_type=mlp" \
  -H "Content-Type: application/json" \
  -d '{
    "Tenure_Months": 2,
    "Monthly_Charges": 70.0,
    "Gender_Male": 0,
    "Partner_Yes": 0,
    "Dependents_Yes": 0,
    "Phone_Service_Yes": 1,
    "Multiple_Lines_No_phone_service": 0,
    "Multiple_Lines_Yes": 0,
    "Internet_Service_Fiber_optic": 1,
    "Internet_Service_No": 0,
    "Online_Security_No_internet_service": 0,
    "Online_Security_Yes": 0,
    "Online_Backup_No_internet_service": 0,
    "Online_Backup_Yes": 0,
    "Device_Protection_No_internet_service": 0,
    "Device_Protection_Yes": 0,
    "Tech_Support_No_internet_service": 0,
    "Tech_Support_Yes": 0,
    "Streaming_TV_No_internet_service": 0,
    "Streaming_TV_Yes": 0,
    "Streaming_Movies_No_internet_service": 0,
    "Streaming_Movies_Yes": 0,
    "Contract_One_year": 0,
    "Contract_Two_year": 0,
    "Paperless_Billing_Yes": 1,
    "Payment_Method_Credit_card_automatic": 0,
    "Payment_Method_Electronic_check": 1,
    "Payment_Method_Mailed_check": 0
  }'
```

**Windows (PowerShell):**
```powershell
curl -X POST "http://localhost:8000/predict?model_type=mlp" `
  -H "Content-Type: application/json" `
  -d '{
    "Tenure_Months": 2,
    "Monthly_Charges": 70.0,
    "Gender_Male": 0,
    "Partner_Yes": 0,
    "Dependents_Yes": 0,
    "Phone_Service_Yes": 1,
    "Multiple_Lines_No_phone_service": 0,
    "Multiple_Lines_Yes": 0,
    "Internet_Service_Fiber_optic": 1,
    "Internet_Service_No": 0,
    "Online_Security_No_internet_service": 0,
    "Online_Security_Yes": 0,
    "Online_Backup_No_internet_service": 0,
    "Online_Backup_Yes": 0,
    "Device_Protection_No_internet_service": 0,
    "Device_Protection_Yes": 0,
    "Tech_Support_No_internet_service": 0,
    "Tech_Support_Yes": 0,
    "Streaming_TV_No_internet_service": 0,
    "Streaming_TV_Yes": 0,
    "Streaming_Movies_No_internet_service": 0,
    "Streaming_Movies_Yes": 0,
    "Contract_One_year": 0,
    "Contract_Two_year": 0,
    "Paperless_Billing_Yes": 1,
    "Payment_Method_Credit_card_automatic": 0,
    "Payment_Method_Electronic_check": 1,
    "Payment_Method_Mailed_check": 0
  }'
```

> Perfil de alto risco: contrato mês a mês, 2 meses de relacionamento, fibra ótica
> e boleto eletrônico. Resposta esperada: `{"churn": 1}`.
>
> Substitua `model_type=mlp` por `logistic` ou `dummy` para testar os outros modelos.

---

## Testes automatizados

Para validar o projeto com pytest:

```powershell
pytest tests/ -v
```

Os testes de smoke e schema pulam automaticamente se o CSV processado não estiver presente.

---

## Observações para o avaliador

Este fluxo cobre os principais requisitos do Tech Challenge:

- execução reprodutível em Docker
- API funcional com FastAPI e validação de entrada
- MLflow para rastreamento de experimentos e Model Registry
- modelo MLP em PyTorch comparado com baselines
- testes automatizados
- pipeline CI/CD com deploy automatizado na VPS
