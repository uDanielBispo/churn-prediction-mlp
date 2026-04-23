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
│   ├── train_mlp.py              # Script de treinamento da MLP com MLflow
│   ├── models/                   # Artefatos treinados (.pkl)
│   └── api/                      # Serviço de inferência
│       ├── main.py               # Inicialização do FastAPI e middleware de latência
│       ├── routes.py             # Endpoints: GET /health, POST /predict
│       ├── schemas.py            # Validação de entrada com Pydantic (CustomerData)
│       ├── services/
│       │   └── model_service.py  # Carregamento dos modelos e lógica de predição
│       └── core/
│           └── logger.py         # Logger estruturado (timestamp | nível | módulo)

├── tests/                        # Testes automatizados
│   ├── test_smoke.py             # Smoke tests do pipeline de treinamento
│   ├── test_schema.py            # Validação do schema do CSV com Pandera
│   └── test_api.py               # Testes dos endpoints da API

├── eda_ciclo_de_vida_de_modelos_sem_mlp_pytorch/  # Fase 1 do projeto (baseline)
│   ├── src/                      # Regressão Logística + DummyClassifier com MLflow
│   └── tests/                    # Testes unitários dos modelos baseline

├── pyproject.toml                # Dependências, ruff e pytest centralizados
├── Makefile                      # Atalhos: make lint | test | run | train
└── README.md                     # Documentação principal
```


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
* ⏳ Deploy em nuvem

---

## Próximos Passos

* Deploy em ambiente cloud
* Containerização com Docker
* Monitoramento contínuo em produção

---

## Contribuição

Projeto desenvolvido em grupo como parte de desafio acadêmico de Machine Learning.

---

## Licença

Uso educacional.
