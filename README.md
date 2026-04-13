# Churn Prediction com Machine Learning (MLP)

## Introdução

A retenção de clientes é um dos principais desafios em empresas de telecomunicações. A perda de clientes (churn) impacta diretamente a receita e o crescimento do negócio.

Este projeto tem como objetivo desenvolver um modelo de Machine Learning capaz de prever quais clientes possuem maior probabilidade de cancelamento, permitindo ações proativas de retenção.

O projeto segue uma abordagem **end-to-end**, desde a análise exploratória dos dados até a preparação para deploy em produção.

---

## Objetivo

Desenvolver um sistema preditivo de churn que:

* Classifique clientes com risco de cancelamento
* Apoie decisões estratégicas de retenção
* Permita integração futura via API para uso em produção

Além disso, o projeto busca aplicar boas práticas de engenharia de Machine Learning, incluindo:

* Reprodutibilidade
* Versionamento de experimentos
* Comparação entre modelos
* Estruturação profissional do código

---

## Dataset

O projeto utiliza um dataset de churn de clientes de telecomunicações, contendo informações como:

* Dados demográficos dos clientes
* Tipo de contrato
* Serviços contratados
* Tempo de relacionamento (tenure)
* Valor mensal (Monthly Charges)

A variável alvo (`target`) indica se o cliente cancelou o serviço (churn) ou não.

> **Observação:** O dataset será versionado e rastreado ao longo do projeto para garantir reprodutibilidade.

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

O projeto segue uma estrutura modular inspirada em boas práticas de engenharia de ML:

```bash
.
├── data/           # Dados brutos e processados
├── docs/           # Documentação (Model Card, arquitetura, etc.)
├── notebooks/      # Análises exploratórias e experimentos
├── src/            # Código-fonte (pipeline, modelos, utils)
├── tests/          # Testes automatizados
└── README.md       # Documentação principal do projeto
```

> A estrutura está em evolução e será refinada ao longo do desenvolvimento do projeto.

---

## Status do Projeto

Atualmente, o projeto encontra-se nas etapas iniciais:

* ✅ Análise exploratória dos dados (EDA)
* ✅ Modelos baseline (ex: Regressão Logística)
* 🔄 Implementação de rede neural (MLP)
* ⏳ Construção de API para inferência
* ⏳ Documentação completa e deploy

---

## Tecnologias Utilizadas

* Python
* Scikit-Learn
* PyTorch *(em desenvolvimento)*
* Pandas & NumPy
* Matplotlib / Seaborn
* MLflow *(planejado)*
* FastAPI *(planejado)*

---

## Métricas de Avaliação

Os modelos serão avaliados utilizando métricas adequadas para problemas de classificação binária:

* AUC-ROC
* F1-Score
* Precision
* Recall

A escolha das métricas considera o impacto de falsos positivos e falsos negativos no contexto de churn.

## Próximos Passos

* Implementar modelo de rede neural (MLP) com PyTorch
* Comparar desempenho com modelos baseline
* Integrar pipeline de dados reprodutível
* Construir API de inferência com FastAPI
* Adicionar rastreamento de experimentos com MLflow
* Criar Model Card e documentação completa

---

## Contribuição

Este projeto está sendo desenvolvido em grupo como parte de um desafio acadêmico de Machine Learning.

A colaboração é feita via branches no GitHub, com versionamento contínuo e organização por responsabilidades.

---

## Licença

Projeto para fins educacionais.

