# Monitoring — Churn Prediction Model

## 1. Visão Geral

Este documento descreve a estratégia de monitoramento do modelo de churn em ambiente de produção.

O objetivo é garantir que o modelo mantenha sua performance ao longo do tempo e continue gerando valor para o negócio.

O monitoramento considera tanto o desempenho do modelo quanto o comportamento do sistema de inferência via API.

---

## 2. O que deve ser monitorado

### Performance do Modelo

Métricas principais:

* ROC-AUC
* Recall
* Precision
* F1-Score

Objetivo:

* Detectar degradação de performance ao longo do tempo
* Comparar versões de modelos

As métricas podem ser registradas e analisadas via MLflow.

---

### Distribuição dos Dados (Data Drift)

Monitorar mudanças nas distribuições das features:

* Tempo de relacionamento (tenure)
* Tipo de contrato
* Serviços contratados
* Métodos de pagamento

Objetivo:

* Identificar quando os dados atuais diferem dos dados de treino
* Detectar necessidade de re-treinamento

---

### Taxa de Churn

Monitorar:

* Percentual real de clientes que cancelam

Objetivo:

* Identificar mudanças no comportamento do negócio
* Validar se o modelo continua alinhado à realidade

---

### Monitoramento da API (Inferência)

Monitorar:

* Latência das requisições
* Taxa de erro (4xx, 5xx)
* Volume de requisições

Objetivo:

* Garantir estabilidade do serviço
* Detectar falhas na inferência

O projeto já inclui logging estruturado em:

* `src/api/core/logger.py`

---

## 3. Tipos de Problemas Monitorados

### Model Drift

* Queda nas métricas (ex: AUC, Recall)
* Indica perda de performance do modelo

---

### Data Drift

* Mudança no perfil dos dados de entrada
* Pode impactar a qualidade das previsões

---

### Concept Drift

* Mudança na relação entre variáveis e churn
* Exemplo: novos comportamentos de clientes

---

## 4. Ações Recomendadas

Quando detectar problemas:

### Queda de performance

* Re-treinar o modelo com dados mais recentes
* Comparar com versões anteriores (MLflow)

---

### Data drift

* Reavaliar pipeline de features (`pipeline.py`)
* Ajustar pré-processamento

---

### Problemas na API

* Analisar logs
* Validar entradas
* Verificar integridade do modelo carregado

---

### Mudanças no negócio

* Revisar variáveis utilizadas
* Atualizar modelo

---

## 5. Frequência de Monitoramento

* Avaliação semanal das métricas
* Monitoramento contínuo da API
* Re-treinamento periódico (mensal ou trimestral)

---

## 6. Monitoramento Preventivo

O projeto inclui testes automatizados que ajudam a prevenir falhas:

* `test_smoke.py` → valida pipeline
* `test_schema.py` → valida dados de entrada
* `test_api.py` → valida endpoints

Esses testes devem ser executados continuamente durante o desenvolvimento.

---

## 7. Alertas (Evolução Futura)

Sugestões:

* Alertas automáticos para queda de AUC
* Monitoramento de drift automatizado
* Alertas para falhas na API
* Integração com dashboards

---

## 8. Integração com Sistema

O monitoramento pode ser integrado com:

* Logs da API (`src/api/core/logger.py`)
* MLflow (métricas e experimentos)
* Ferramentas externas de observabilidade

---

## 9. Considerações Finais

O monitoramento é essencial para garantir que:

* O modelo continue relevante
* As decisões baseadas nele sejam confiáveis
* O sistema funcione de forma estável

---
