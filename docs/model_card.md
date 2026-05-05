# Model Card — Churn Prediction (MLP)

---

## 1. Visão Geral

Este modelo tem como objetivo prever a probabilidade de churn (cancelamento) de clientes em uma empresa de telecomunicações.

* **Tipo de problema:** Classificação binária
* **Saída:** Probabilidade de churn (0 a 1) e classificação (churn / não churn)
* **Objetivo de negócio:** Identificar clientes com alto risco de cancelamento para permitir ações proativas de retenção

### Aplicações no negócio:

* Priorização de clientes para campanhas de retenção
* Redução do churn rate
* Otimização de custos de marketing
* Suporte à tomada de decisão em CRM

---

## 2. Dados Utilizados

* **Dataset:** Telco Customer Churn
* **Tipo:** Dados tabulares
* **Volume:** ~7.000 registros de clientes

### Tipos de variáveis:

* **Comportamentais:**

  * Tempo de relacionamento (`Tenure Months`)
* **Financeiras:**

  * `Monthly Charges`, `CLTV`
* **Serviços contratados:**

  * Internet, streaming, suporte técnico
* **Perfil do cliente:**

  * Partner, Dependents

### Pré-processamento:

* Encoding de variáveis categóricas (one-hot encoding)
* Seleção de 25 features relevantes
* Padronização com **StandardScaler**
* Pipeline centralizado em `src/pipeline.py`

### Considerações importantes:

* O dataset apresenta **desbalanceamento de classes**
* Foi adotado pipeline para evitar **data leakage**
* Separação clara entre:

  * `data/raw/` → dados originais
  * `data/processed/` → dados tratados

---

## 3. Metodologia e Modelagem

### Pipeline de Machine Learning:

1. Análise exploratória dos dados (EDA)
2. Feature engineering
3. Treinamento de modelos baseline
4. Treinamento de rede neural (MLP)
5. Avaliação e comparação
6. Deploy via API

### Modelos utilizados:

* DummyClassifier (baseline mínimo)
* Regressão Logística
* **MLP (Multi-Layer Perceptron com PyTorch)**

### MLP:

* Arquitetura definida em `src/model.py`
* Treinamento via `src/train_mlp.py`
* Early stopping para evitar overfitting
* Salvamento de pesos (`.pth`) e modelo final (`.pkl`)

### Pipeline de Dados:

* Implementado em `src/pipeline.py`
* Reutilizado em treino e inferência
* Garante consistência e reprodutibilidade

### Rastreamento de Experimentos:

* Uso de **MLflow** para:

  * registro de métricas
  * comparação entre modelos
  * versionamento de execuções

### Divisão dos dados:

* Treino / validação / teste (configurável)

---

## 4. Métricas de Performance

Os modelos são avaliados utilizando:

* ROC-AUC
* Precision
* Recall
* F1-Score

### Estratégia de avaliação:

* Priorização de **Recall**, pois:

  * é mais importante identificar clientes que irão churnar
* Aceita-se aumento de falsos positivos dentro de limites operacionais

> As métricas são registradas via MLflow e podem ser comparadas entre execuções.

---

## 5. Interpretação do Modelo

Com base na análise exploratória e features selecionadas:

### Principais fatores associados ao churn:

* Baixo tempo de relacionamento
* Contratos mensais (vs contratos longos)
* Ausência de serviços adicionais (ex: suporte técnico)
* Uso de internet fibra
* Métodos de pagamento como *electronic check*

### Observação:

* Modelos lineares (Logística) oferecem maior interpretabilidade
* MLP apresenta maior capacidade de modelar relações complexas, porém com menor interpretabilidade

> Técnicas como SHAP não foram implementadas nesta versão.

---

## 6. Limitações e Riscos

* Modelo baseado em dados históricos (pode não refletir mudanças futuras)
* Possível correlação entre variáveis
* Não considera fatores externos (mercado, concorrência)
* Dependência da qualidade dos dados de entrada
* MLP possui menor interpretabilidade

---

## 7. Fairness e Viés

* Não foram utilizadas variáveis sensíveis explícitas
* Avaliações formais de fairness não foram realizadas

> Caso aplicado em produção real, recomenda-se avaliação de viés.

---

## 8. Deploy e Uso em Produção

O modelo está disponível via API utilizando **FastAPI**.

### Arquitetura:

* API definida em `src/api/`
* Separação em camadas:

  * rotas
  * schemas (validação)
  * serviços (lógica de modelo)

### Endpoint principal:

* `POST /predict` → retorna previsão de churn

### Características:

* Suporte a múltiplos modelos:

  * `mlp`
  * `logistic`
  * `dummy`
* Validação de entrada com Pydantic
* Logging estruturado
* Middleware de latência

### Execução:

* Local (uvicorn)
* Containerizada via Docker

---

## 9. Monitoramento

A estratégia de monitoramento considera:

* Performance do modelo (AUC, Recall, F1)
* Data drift nas features
* Distribuição das variáveis ao longo do tempo

### Boas práticas adotadas:

* Logs estruturados na API
* Testes automatizados garantindo integridade

### Ações recomendadas:

* Re-treinamento periódico
* Alertas para degradação de performance

---

## 10. Reprodutibilidade

O projeto foi estruturado para garantir reprodutibilidade:

* Pipeline centralizado (`pipeline.py`)
* Código modular em `src/`
* MLflow para tracking de experimentos
* Testes automatizados (`tests/`)

### Execução:

* Treinamento:

```bash
python -m src.train_mlp
```

* API:

```bash
uvicorn src.api.main:app --reload
```

* Docker:

```bash
docker-compose up
```

Dependências gerenciadas via `pyproject.toml`.

---

## 11. Impacto de Negócio (Simulado)

O modelo permite:

* Identificar clientes com maior risco de churn
* Priorizar ações de retenção
* Reduzir perdas de receita

### Exemplo de uso:

* Selecionar os **top 10% clientes com maior risco**
* Direcionar campanhas específicas

---

## 12. Responsáveis

Projeto desenvolvido em grupo como parte de desafio acadêmico de Machine Learning.

---

## 13. Features Selecionadas

As features foram selecionadas com base em relevância estatística e impacto no comportamento de churn.

### Critérios de exclusão:

* Baixa variância
* Alta cardinalidade
* Redundância
* Risco de data leakage

### Observação importante:

Variáveis como `Churn Score`, `CLTV` e `Churn Reason` foram **excluídas** por representarem **data leakage**, garantindo que o modelo utilize apenas informações disponíveis no momento da predição.
