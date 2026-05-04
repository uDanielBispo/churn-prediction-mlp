# Churn Prediction com Machine Learning (MLP)

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

O projeto utiliza o dataset **Telco Customer Churn**, amplamente utilizado para problemas de previsão de churn em telecomunicações.

* **Tipo:** Dados tabulares
* **Volume:** ~7.000 clientes
* **Problema:** Classificação binária (churn vs não churn)

### Características:

* Dados demográficos
* Tipo de contrato
* Serviços contratados
* Tempo de relacionamento (`tenure`)
* Valor mensal (`MonthlyCharges`)

A variável alvo (`target`) indica se o cliente cancelou o serviço.

### Considerações importantes:

* O dataset apresenta **desbalanceamento de classes**, comum em problemas de churn
* Foi adotado um pipeline de pré-processamento para evitar **data leakage**
* Os dados são separados em:

  * `data/raw/` → dados originais
  * `data/processed/` → dados tratados

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

├── data/                         # Dados utilizados no projeto
│   ├── raw/                      # Dados originais (imutáveis)
│   │   └── Telco_customer_churn.csv
│   └── processed/                # Dados tratados para modelagem
│       └── telco_customer_churn_processed.csv

├── docs/                         # Documentação técnica e de negócio
│   ├── model_card.md             # Documentação do modelo (uso, métricas, riscos)
│   ├── architecture.md           # Arquitetura end-to-end do sistema
│   ├── monitoring.md             # Estratégia de monitoramento em produção
│   ├── ML Canvas.md              # Contexto de negócio e proposta de valor
│   └── refatoracao.md            # Evolução e melhorias do projeto

├── notebooks/                    # Ambiente exploratório e experimentação
│   ├── eda.ipynb                 # Análise exploratória dos dados
│   ├── baseline_models.ipynb     # Modelos baseline (Dummy + Regressão Logística)
│   └── mlp_training_and_comparison.ipynb  # Treino da MLP e comparação de desempenho

├── src/                          # Código-fonte principal (produção)
│   ├── dataset.py                # Dataset customizado para PyTorch (ChurnDataset)
│   ├── pipeline.py               # Pipeline de pré-processamento (padronização e consistência)
│   ├── model.py                  # Arquitetura da rede neural MLP
│   ├── train_baselines.py        # Treinamento dos modelos baseline (Dummy + Logística)
│   ├── train_mlp.py              # Treinamento da MLP com MLflow
│   ├── early_stopping.py         # Early stopping para evitar overfitting
│   ├── utils.py                  # Funções auxiliares do projeto
│   ├── models/                   # Artefatos treinados (modelos e preprocessador)
│   └── api/                      # API de inferência (FastAPI)
│       ├── main.py               # Inicialização da aplicação e middlewares
│       ├── routes.py             # Definição dos endpoints (/health, /predict)
│       ├── schemas.py            # Validação de entrada com Pydantic
│       ├── services/
│       │   └── model_service.py  # Lógica de carregamento e predição do modelo
│       └── core/
│           └── logger.py         # Logger estruturado da aplicação

├── tests/                        # Testes automatizados
│   ├── test_smoke.py             # Teste básico do pipeline de treinamento
│   ├── test_schema.py            # Validação de dados com Pandera
│   └── test_api.py               # Testes dos endpoints da API

├── eda_ciclo_de_vida_de_modelos_sem_mlp_pytorch/  # Versão inicial do projeto (baseline)
│   ├── src/                      # Implementação inicial dos modelos
│   └── tests/                    # Testes da fase inicial

├── Dockerfile                    # Definição da imagem Docker da aplicação
├── docker-compose.yml            # Orquestração do serviço da API
├── pyproject.toml                # Gerenciamento de dependências e ferramentas (ruff, pytest)
├── requirements.txt              # Dependências para ambientes alternativos
├── Makefile                      # Comandos utilitários (train, test, run, lint)
├── RUNNING.md                    # Guia adicional de execução do projeto
└── README.md                     # Documentação principal

---

## Como Executar o Projeto

### 1. Clonar o repositório

```bash
git clone https://github.com/uDanielBispo/churn-prediction-mlp.git
cd churn-prediction-mlp
```

---

### 2. Criar o ambiente virtual

> **Atenção:** use Python 3.11 especificamente. Versões mais recentes (3.12+) ainda não possuem suporte completo ao PyTorch.

```bash
python3.11 -m venv .venv
source .venv/bin/activate    # macOS/Linux
.venv\Scripts\activate       # Windows
```

---

### 3. Instalar o projeto e suas dependências

O comando abaixo instala todas as dependências (produção + desenvolvimento) e registra o projeto como pacote, permitindo que os imports `from src.X import Y` funcionem corretamente:

```bash
pip install -e ".[dev]"
```

---

### 4. Treinar o modelo MLP

```bash
python -m src.train_mlp
```

> O `-m` é necessário para rodar como módulo a partir da raiz do projeto. Usar `python src/train_mlp.py` diretamente causa erro de import.

Os artefatos gerados ficam em:
- `src/models/churn_prediction_mlp_pytorch_model.pkl` — modelo treinado
- `src/models/preprocessor.pkl` — pipeline de pré-processamento
- `best_mlp_model.pth` — pesos do melhor checkpoint

---

### 5. Executar os testes

```bash
pytest tests/ -v
```

Ou usando o Makefile:

```bash
make test
```

---

### 6. Subir a API

```bash
uvicorn src.api.main:app --reload
```

Ou usando o Makefile:

```bash
make run
```

Acesse a documentação interativa em: [http://localhost:8000/docs](http://localhost:8000/docs)

**Endpoints disponíveis:**

| Método | Rota | Descrição |
|---|---|---|
| `GET` | `/health` | Verifica se a API está no ar |
| `POST` | `/predict?model_type=logistic` | Previsão com Regressão Logística (padrão) |
| `POST` | `/predict?model_type=mlp` | Previsão com rede neural MLP |
| `POST` | `/predict?model_type=dummy` | Previsão com modelo baseline |

**Exemplo de requisição:**

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

> Perfil de alto risco: contrato mês a mês, 2 meses de relacionamento, internet fibra ótica e pagamento por boleto eletrônico. Resposta esperada: `{"churn": 1}`.
>
> Substitua `model_type=mlp` por `logistic` ou `dummy` para testar os outros modelos.

---

### 7. Verificar qualidade do código

```bash
make lint
```

---

## Atalhos do Makefile

| Comando | O que faz |
|---|---|
| `make train` | Treina os modelos de baseline (Logística + Dummy) e a MLP |
| `make test` | Executa os 29 testes automatizados |
| `make run` | Sobe a API FastAPI com hot-reload |
| `make lint` | Verifica o código com ruff |

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
5. Documentação e monitoramento

---

## Pipeline de Dados

O projeto utiliza um pipeline de pré-processamento centralizado (`src/pipeline.py`) responsável por garantir consistência entre as etapas de treinamento e inferência.

Principais responsabilidades:

* Padronização dos dados (StandardScaler)
* Reutilização do mesmo pipeline em treino e produção
* Prevenção de data leakage

O pipeline é salvo como artefato (`preprocessor.pkl`) e carregado automaticamente pela API durante as previsões.

## Rastreamento de Experimentos

O treinamento dos modelos utiliza **MLflow** para rastreamento de experimentos.

Isso permite:

* Comparar diferentes modelos (baseline vs MLP)
* Registrar métricas automaticamente
* Versionar execuções de treinamento

Facilitando análise de performance e reprodutibilidade.

## Execução com Docker

O projeto também pode ser executado utilizando Docker.

### Build da imagem:

```bash
docker build -t churn-mlp .
```

### Subir o serviço:

```bash
docker-compose up
```

A API ficará disponível em:

```
http://localhost:8000
```

Essa abordagem facilita a portabilidade e o deploy do sistema.


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

## Contribuição

Projeto desenvolvido em grupo como parte de desafio acadêmico de Machine Learning.

---

## Licença

Uso educacional.
