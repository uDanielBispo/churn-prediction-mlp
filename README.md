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
├── data/
│   ├── raw/                # Dados originais
│   └── processed/          # Dados tratados
├── docs/
│   ├── model_card.md
│   └── ML Canvas.md
├── notebooks/
│   ├── eda.ipynb
│   ├── baseline_models.ipynb
│   └── mlp_training_and_comparison.ipynb
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── train_mlp.py
│   ├── early_stopping.py
│   ├── utils.py
│   ├── register.py
│   └── api/
│       ├── main.py
│       ├── routes.py
│       ├── schemas.py
│       ├── services/
│       │   └── model_service.py
│       └── core/
│           └── loggin.py
├── tests/
│   └── unit_tests.py
└── README.md
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

* Model Card: `docs/model_card.md`
* ML Canvas: `docs/ML Canvas.md`

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
