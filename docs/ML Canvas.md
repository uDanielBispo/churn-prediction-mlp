# Telco Chrun Prediction | ML Canva

## Business Understanding

### Perguntas iniciais

1. **Qual é exatamente a dor do negócio?**
   A Telco enfrenta uma alta na taxa de cancelamento de clientes (Churn), isso gera perda de receita recorrente e aumento do custo de aquisição de novos clientes. A empresa não conta com uma forma eficiente de identificar quais clientes possuem maior risco de cancelamento.
3. **Qual o numero precisamos mover?**
   Redução da taxa de Churn em 15%
5. **Qual o retorno esperado?**
   Redução da taxa de Churn, retendo clientes e garantindo a receita recorrente da Telco

### Objetivos


| Objetivo de negócio | Objetivo Técnico    |
| ---------------------- | ---------------------- |
| Reduzir Churn em 15% | Modelo de previsão de churn, priorizando recall ≥ 80% para capturar clientes em risco de saída, com F1-Score ≥ 75% como métrica de equilíbrio.)           |
| Prvisão de Chrun    | Modelo treinado e avaliado com métricas AUC-ROC ≥ 0,85, F1 ≥ 0,80, PR-AUC ≥ 0,80; endpoint de inferência via API FastAPI |

### Cronograma

* **Fase 1 – 10 dias (27/03/2026 – 05/04/2026)**
* **Fase 2 – 10 dias (06/04/2026 – 15/04/2026)**
* **Fase 3 – 10 dias (16/04/2026 – 25/04/2026)**
* **Fase 4 – 10 dias (26/04/2026 – 05/05/2026)**

### Ferramentas

* **Python** : Linguagem principal para desenvolvimento do projeto
* **PyTorch** : Construção e treinamento da rede neural
* **Scikit-Learn** : Pipelines de pré-processamento e modelos baseline
* **MLflow** : Rastreamento de experimentos
* **FastAPI** : Criação da API para servir o modelo
* **Pytest** : Testes automatizados
* **Pandera** : Validação de schema e qualidade dos dados
* **Ruff** : Linting e padronização do código
* **Git/GitHub** : Versionamento e colaboração do projeto
* **pyproject.toml** : Gerenciamento de dependências e configuração do projeto
* **AWS / Azure / GCP** : Nuvem para deploy da aplicação

### Stakeholders

#### Quem se beneficia ou precisa das informações do modelo:

* **Diretoria** : quer reduzir churn e entender o risco de cancelamento de clientes.
* **Marketing** : precisa priorizar campanhas para clientes com alto risco de churn.
* **Suporte ao Cliente** : podem usar insights do modelo para interações personalizadas e melhorias no atendimento.

#### Escopo

##### O que será desenvolvido

**Modelo de Previsão de Churn**

* Rede neural (MLP) em PyTorch treinada e comparada com modelos baseline.
* Pipeline de pré-processamento de dados reprodutível (Scikit-Learn + transformadores customizados).

**API de Inferência (FastAPI)**

* Endpoints:`/predict` para previsão e`/health` para monitoramento.
* Validação de dados de entrada com Pydantic.
* Logging estruturado e monitoramento de latência.

**Deploy em Nuvem (AWS)**

* Serviço de API hospedado em AWS com endpoint público.
* Configuração de escalabilidade e disponibilidade mínima (SLA).

**Documentação e Controle de Experimentos**

* Registro de métricas, parâmetros e artefatos no MLflow.
* Model Card completo (performance, limitações e vieses).
* README com instruções de setup, execução e arquitetura.

##### **O que não será desenvolvido**

* Dashboards interativos ou front-end além da API.
* Coleta de dados externos ao dataset fornecido.
* Modelos complexos fora do escopo de classificação tabular (ex.: NLP ou visão computacional).
* Estratégias de marketing ou ações de retenção reais, apenas métricas simuladas.

### Métricas de negócio

* **Redução do churn (15%)** : quantidade de clientes que deixam a operadora antes e depois da implementação do modelo.
* **Precisão de classificação de alto risco (≥ 80%)** : quantos clientes realmente propensos a cancelar foram identificados corretamente.

### SLOs

* **Acurácia mínima do modelo** : ≥ 80% no conjunto de validação.
* **AUC-ROC** : ≥ 0.85 para assegurar boa separação entre clientes que churnam e não churnam.
* **Tempo de inferência da API** : ≤ 200 ms por requisição.
* **Disponibilidade da API** : ≥ 99% uptime.
* **Reprodutibilidade dos resultados** : todos os experimentos rastreados com MLflow, com seeds fixadas.
