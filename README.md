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
│   ├── model_card.md             # Descrição detalhada do modelo, limitações e uso
│   ├── architecture.md           # Descrição do fluxo do modelo detalhado
│   ├── monitoring.md             # Descrição do fluxo de monitoramento do modelo
│   └── ML Canvas.md              # Definição do problema de negócio e contexto

├── notebooks/                   # Ambiente exploratório (EDA e experimentos)
│   ├── eda.ipynb                # Análise exploratória dos dados
│   ├── baseline_models.ipynb    # Treinamento e avaliação dos modelos baseline
│   └── mlp_training_and_comparison.ipynb  # Treinamento da MLP e comparação com baselines

├── src/                         # Código-fonte principal (estrutura produtiva)
│   ├── dataset.py               # Carregamento e preparação dos dados
│   ├── model.py                 # Definição da arquitetura da rede neural (MLP)
│   ├── train.py                 # Pipeline geral de treinamento (modelos clássicos)
│   ├── train_mlp.py             # Script de treinamento específico da MLP
│   ├── early_stopping.py        # Implementação de early stopping para evitar overfitting
│   ├── utils.py                 # Funções auxiliares reutilizáveis
│   ├── register.py              # Registro e persistência de modelos/artefatos

│   └── api/                     # Camada de API para inferência do modelo
│       ├── main.py              # Inicialização da aplicação FastAPI
│       ├── routes.py            # Definição dos endpoints (ex: /predict, /health)
│       ├── schemas.py           # Validação de dados de entrada (Pydantic)
│       ├── services/            # Camada de lógica de negócio da API
│       │   └── model_service.py # Serviço responsável por carregar modelo e prever
│       └── core/                # Configurações centrais da API
│           └── loggin.py        # Configuração de logging estruturado

├── tests/                       # Testes automatizados do projeto
│   └── unit_tests.py            # Testes unitários das principais funcionalidades

└── README.md                   # Documentação principal e guia do projeto
```


---

## Como Executar o Projeto

### 1. Clonar o repositório

```bash
git clone https://github.com/uDanielBispo/churn-prediction-mlp.git
cd churn-prediction-mlp
```

---

### 2. Criar ambiente virtual

```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
```

---

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

> Caso o arquivo não esteja completo, instale manualmente:

```bash
pip install pandas numpy scikit-learn torch fastapi uvicorn matplotlib seaborn
```

---

## Execução das Etapas

### 📊 1. Análise Exploratória (EDA)

```bash
jupyter notebook notebooks/eda.ipynb
```

---

### 📈 2. Modelos Baseline

```bash
jupyter notebook notebooks/baseline_models.ipynb
```

---

### 🤖 3. Treinamento da Rede Neural (MLP)

```bash
jupyter notebook notebooks/mlp_training_and_comparison.ipynb
```

Ou via script:

```bash
python src/train_mlp.py
```

---

## Executando a API

### 1. Subir o servidor

```bash
uvicorn src.api.main:app --reload
```

---

### 2. Endpoints disponíveis

* `GET /health` → Verifica status da API
* `POST /predict` → Realiza previsão de churn

---

### 3. Acessar documentação interativa

Abra no navegador:

```bash
http://127.0.0.1:8000/docs
```

---

## Executando Testes

```bash
pytest tests/
```

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

* EDA: concluído
* Baselines: concluído
* MLP: implementado
* API: implementada
* Testes: em evolução
* MLflow: pendente

---

## Próximos Passos

* Integração com MLflow
* Expansão de testes automatizados
* Monitoramento em produção
* Deploy em nuvem

---

## Contribuição

Projeto desenvolvido em grupo como parte de desafio acadêmico de Machine Learning.

---

## Licença

Uso educacional.
