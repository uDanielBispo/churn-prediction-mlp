# Churn Prediction com Machine Learning (MLP)

## Links uteis:
[Clique aqui para acessar o video STAR no youtube](https://www.youtube.com/watch?v=oqNWGabEh1o)

[Fast API](http://82.29.57.75:8000/docs)

[MLFlow](http://82.29.57.75:5000/)

## Introdução

A retenção de clientes é um dos principais desafios em empresas de telecomunicações. A perda de clientes (churn) impacta diretamente a receita e o crescimento do negócio, tornando essencial a identificação antecipada de clientes com maior risco de cancelamento.

Este projeto tem como objetivo desenvolver um modelo de Machine Learning capaz de prever churn, permitindo que a empresa atue de forma proativa com estratégias de retenção.

O desenvolvimento segue uma abordagem **end-to-end**, cobrindo desde a análise exploratória dos dados até a preparação do modelo para consumo via API em ambiente de produção.

---

## Objetivo

Desenvolver um sistema preditivo de churn que:

* Classifique clientes com risco de cancelamento
* Apoie decisões estratégicas de retenção
* Permita integração futura via API para uso em produção

Além disso, o projeto busca aplicar boas práticas de engenharia de Machine Learning.

---

## Dataset

O projeto utiliza o dataset **Telco Customer Churn**, contendo:

* Dados demográficos
* Tipo de contrato
* Serviços contratados
* Tempo de relacionamento (`tenure`)
* Valor mensal (`MonthlyCharges`)

A variável alvo (`target`) indica churn.

---

## Feature Engineering

Após a etapa de preparação dos dados, foram selecionadas e transformadas 25 features para treinamento dos modelos.

# Variáveis numéricas:
* **Tenure Months**
* **Churn Score**
* **CLTV**

# Variáveis categóricas (após encoding):
* **Parceiro e dependentes:**
    * Partner_Yes, Dependents_Yes
* **Serviços de internet:**
    * Internet Service_Fiber optic, Internet Service_No
* **Serviços adicionais:**
    * Segurança, backup, proteção, suporte técnico
    * Streaming de TV e filmes
(incluindo categorias "No internet service")
* **Tipo de contrato:**
    * Contract_One year, Contract_Two year
* **Faturamento:**
    * Paperless Billing_Yes
* **Métodos de pagamento:**
    * Payment Method_Credit card (automatic)
    * Payment Method_Electronic check
    * Payment Method_Mailed check

Essas variáveis foram escolhidas com base na análise exploratória e representam fatores relevantes para o comportamento de churn dos clientes.

## Estrutura do Projeto

```bash
.
├── data/                         # Armazenamento dos dados utilizados no projeto
│   ├── raw/                      # Dados originais (sem tratamento)
│   └── processed/                # Dados limpos e preparados para modelagem

├── docs/                         # Documentação do projeto
│   ├── architecture.md           # Fluxo do sistema end-to-end
│   ├── monitoring.md             # Estratégia de monitoramento em produção
│   └── refatoracao.md            # Comparativo antes/depois da refatoração

├── notebooks/                    # Ambiente exploratório (EDA e experimentos)
│   └── mlp_training_and_comparison.ipynb  # Treinamento da MLP e comparação com baselines

├── src/                          # Código-fonte principal
│   ├── dataset.py                # Dataset customizado para PyTorch (ChurnDataset)
│   ├── early_stopping.py         # Early stopping para evitar overfitting
│   ├── model.py                  # Arquitetura da rede neural MLP (ChurnMLP)
│   ├── pipeline.py               # Pipeline sklearn de pré-processamento (StandardScaler)
│   ├── train_baselines.py        # Treina Logística e Dummy; loga no MLflow
│   ├── train_mlp.py              # Treina a MLP; loga modelo e preprocessor no MLflow
│   ├── register_models.py        # Promove modelos para Production no MLflow Registry
│   ├── utils.py                  # Utilitários: set_seed, find_best_threshold, setup_mlflow
│   ├── models/                   # Artefatos treinados (.pkl) — usados apenas localmente
│   └── api/                      # Serviço de inferência
│       ├── main.py               # Inicialização do FastAPI e middleware de latência
│       ├── routes.py             # Endpoints: GET /health, POST /predict
│       ├── schemas.py            # Validação de entrada com Pydantic (CustomerData)
│       ├── services/
│       │   └── model_service.py  # Carrega modelos (.pkl local ou MLflow Registry em prod)
│       └── core/
│           └── logger.py         # Logger estruturado (timestamp | nível | módulo)

├── tests/                        # Testes automatizados
│   ├── test_smoke.py             # Smoke tests do pipeline de treinamento
│   ├── test_schema.py            # Validação do schema do CSV com Pandera
│   └── test_api.py               # Testes dos endpoints da API

├── eda_ciclo_de_vida_de_modelos_sem_mlp_pytorch/  # Fase 1 do projeto (baseline)
│   ├── src/                      # Regressão Logística + DummyClassifier com MLflow
│   └── tests/                    # Testes unitários dos modelos baseline

├── .github/workflows/ci-cd.yml   # Pipeline CI/CD: lint, testes, treino e deploy
├── docker-compose.yml            # Compose para desenvolvimento local
├── docker-compose.prod.yml       # Compose para a VPS em produção
├── Dockerfile                    # Imagem da API usada em ambos os ambientes
├── pyproject.toml                # Dependências, ruff e pytest centralizados
├── Makefile                      # Atalhos: make lint | test | run | train | register
└── README.md                     # Documentação principal
```


--- 
## Como Rodar o Projeto

### Objetivo

Executar a solução de churn de ponta a ponta em um ambiente isolado:
- API FastAPI para inferência
- MLflow UI para visualização de experimentos
- Treinamento PyTorch para registrar métricas e artefatos

### Requisitos

- Docker Desktop instalado e em execução
- Repositório clonado no host

---

### Ambiente local (desenvolvimento)

#### Passos para rodar

1. Abra o Docker Desktop.
2. Navegue até a raiz do projeto.
3. Suba o MLflow e aguarde alguns minutos:
   ```powershell
   docker compose up -d mlflow
   ```
4. Rode o treinamento para gerar experimentos e artefatos:
   ```powershell
   docker compose run --rm train
   ```
    > O serviço `train` executa `make train`, que treina os modelos de baseline
   > (Logística + Dummy) e a MLP em sequência, salvando os `.pkl` em `src/models/`
   > e logando métricas no MLflow local.
5. Suba a API :
   ```powershell
   docker compose up -d api
   ```

5. Para parar os serviços:
   ```powershell
   docker compose down
   ```

#### URLs de acesso

- API: `http://localhost:8000`
- Documentação da API: `http://localhost:8000/docs`
- MLflow UI: `http://localhost:5000`

#### O que cada serviço faz

- `api` → executa a API FastAPI na porta `8000`
- `mlflow` → executa o MLflow UI na porta `5000`
- `train` → treina Logística, Dummy e MLP; registra resultados no MLflow local

---

### Ambiente de produção (VPS)

#### Links de prod:
Os links abaixo ja estão implementados em uma VPS do Hostinger, disponiveis para teste dos avaliadores.
- [Fast API](http://82.29.57.75:8000/docs)
- [MLFlow](http://82.29.57.75:5000/)

#### Processo para deploy na VPS

Usa `docker-compose.prod.yml`. A VPS só precisa ter **Docker e git** instalados —
todo o restante (Python, dependências, treino) roda dentro de containers.

##### Primeira configuração na VPS

A primeira execução do pipeline CI/CD já cuida de tudo automaticamente:
clona o repositório, sobe o MLflow, treina os modelos e faz o deploy da API.

Caso queira executar manualmente antes do primeiro merge na `main`:

```bash
# 1. Clonar o repositório
git clone https://github.com/uDanielBispo/churn-prediction-mlp.git
cd churn-prediction-mlp

# 2. Subir o MLflow persistente
docker compose -f docker-compose.prod.yml up -d mlflow

# 3. Treinar os modelos e registrar no MLflow Registry (via Docker)
docker compose -f docker-compose.prod.yml run --rm train
docker compose -f docker-compose.prod.yml run --rm \
  -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
  train python -m src.register_models

# 4. Subir a API (carrega os modelos do MLflow Registry)
docker compose -f docker-compose.prod.yml up -d api
```

##### Acessar o MLflow UI da VPS

A porta do MLflow é exposta apenas em `127.0.0.1` da VPS (não acessível pela
internet, pois o MLflow não tem autenticação). Para ver a UI a partir da sua
máquina local, abra um túnel SSH:

```bash
ssh -L 5000:localhost:5000 <usuario>@82.29.57.75
```

Enquanto a sessão SSH estiver aberta, acesse no navegador:

```
http://localhost:5000
```

##### Após a primeira configuração

Todo push na branch `main` dispara o pipeline CI/CD automaticamente:

```
push → lint → testes → treino na VPS → registro no MLflow → restart da API
```

Nenhuma intervenção manual é necessária.

##### Segredos necessários no GitHub Actions

Configure em **Settings → Secrets and variables → Actions** do repositório:

| Secret | Exemplo | Descrição |
|---|---|---|
| `VPS_HOST` | `123.45.67.89` | IP ou hostname da VPS |
| `VPS_USER` | `mlfiap2026` | Usuário SSH |
| `VPS_SSH_KEY` | `-----BEGIN...` | Chave privada SSH |
| `VPS_PROJECT_PATH` | `/home/mlfiap2026/churn-prediction-mlp` | Caminho do projeto na VPS |

---

##### Testes rápidos da API

> A API está publicada na VPS em `http://82.29.57.75:8000`. Para testar contra produção,
> basta substituir `http://localhost:8000` pelo endereço público nos exemplos abaixo.

##### Verificar se a API está no ar

```powershell
curl http://localhost:8000/health
```

Em produção:

```bash
curl http://82.29.57.75:8000/health
```

Resposta esperada:

```json
{"status":"ok"}
```

##### Fazer uma predição com o modelo MLP

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

##### Testar a API em produção

Mesmo payload, apontando para a VPS:

```bash
curl -X POST "http://82.29.57.75:8000/predict?model_type=mlp" \
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

Documentação interativa (Swagger UI): `http://82.29.57.75:8000/docs`

---

##### Testes automatizados

Para validar o projeto com pytest:

```powershell
pytest tests/ -v
```

Os testes de smoke e schema pulam automaticamente se o CSV processado não estiver presente.

---

## Métricas de Avaliação

Os modelos serão avaliados utilizando métricas adequadas para problemas de classificação binária:

* AUC-ROC
* F1-Score
* Precision
* Recall

---

## Abordagem do Projeto

1. EDA e entendimento do problema
2. Baselines (Dummy + Regressão Logística)
3. MLP com PyTorch
4. API com FastAPI
5. Containerização com Docker
6. Pipeline CI/CD com GitHub Actions e deploy automatizado na VPS
7. Documentação e estratégia de monitoramento

---

## Documentação

A documentação do projeto está organizada na pasta `docs/` e cobre desde aspectos técnicos do modelo até arquitetura e monitoramento:

* **Model Card** (`docs/model_card.md`)
  Descreve o modelo de churn, incluindo dados utilizados, metodologia, métricas, limitações e aplicações no negócio.

* **Architecture** (`docs/architecture.md`)
  Explica o funcionamento do sistema end-to-end, incluindo fluxo de dados, pipeline de treinamento e integração com a API.

* **Monitoring** (`docs/monitoring.md`)
  Define a estratégia de monitoramento do modelo em produção, incluindo métricas, detecção de drift e plano de re-treinamento.

* **ML Canvas** (`docs/ML Canvas.md`)
  Apresenta o contexto de negócio, problema a ser resolvido e proposta de valor do projeto.


---

## Status do Projeto

* ✅ EDA e análise exploratória
* ✅ Modelos baseline (Regressão Logística + DummyClassifier com MLflow)
* ✅ Rede neural MLP com PyTorch
* ✅ Pipeline sklearn reprodutível (sem data leakage)
* ✅ API FastAPI com logging estruturado e middleware de latência
* ✅ 29 testes automatizados (smoke, schema com Pandera, endpoints)
* ✅ Infraestrutura: pyproject.toml, Makefile, ruff
* ✅ Containerização com Docker
* ✅ Pipeline CI/CD com GitHub Actions e deploy automatizado na VPS
* ✅ MLflow Registry para versionamento e promoção de modelos
* ⏳ Monitoramento contínuo em produção

---

## Próximos Passos

* Implementar monitoramento contínuo de métricas e data drift em produção

---

## Contribuição

Projeto desenvolvido em grupo como parte de desafio acadêmico de Machine Learning.

---

## Licença

Uso educacional.
