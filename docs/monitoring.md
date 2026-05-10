# Monitoring — Churn Prediction Model

## 1. Visão Geral

Este documento descreve a estratégia de monitoramento do modelo de churn em ambiente de produção.

O objetivo é garantir que o modelo mantenha sua performance ao longo do tempo e continue gerando valor para o negócio.

---

## 2. O que deve ser monitorado

### 📊 Performance do Modelo

Métricas principais:

* ROC-AUC
* Recall
* Precision
* F1-Score

Objetivo:

* Detectar degradação de performance ao longo do tempo

---

### 📈 Distribuição dos Dados (Data Drift)

Monitorar mudanças nas distribuições das features:

* Tenure
* Tipo de contrato
* Serviços contratados
* Métodos de pagamento

Objetivo:

* Identificar quando os dados atuais diferem dos dados de treino

---

### 🎯 Taxa de Churn

Monitorar:

* Percentual de clientes que realmente cancelam

Objetivo:

* Identificar mudanças no comportamento do negócio

---

## 3. Tipos de Problemas Monitorados

### ⚠️ Model Drift

* Queda nas métricas (ex: AUC, Recall)
* Indica que o modelo está ficando obsoleto

---

### ⚠️ Data Drift

* Mudança no perfil dos dados de entrada
* Pode afetar a qualidade das previsões

---

### ⚠️ Concept Drift

* Mudança na relação entre features e churn
* Exemplo: novos comportamentos de clientes

---

## 4. Ações Recomendadas

Quando detectar problemas:

### 📉 Queda de performance:

* Re-treinar o modelo com dados mais recentes

---

### 📊 Data drift:

* Reavaliar pipeline de features
* Ajustar pré-processamento

---

### 🔄 Mudanças no negócio:

* Revisar variáveis utilizadas
* Atualizar modelo

---

## 5. Frequência de Monitoramento

* Avaliação semanal das métricas
* Monitoramento contínuo em produção
* Re-treinamento periódico (ex: mensal ou trimestral)

---

## 6. Alertas (Futuro)

Sugestões:

* Alertas automáticos para queda de AUC
* Monitoramento de distribuição de features
* Logs estruturados via API

---

## 7. Integração com Sistema

Já disponível no projeto:

* Logs estruturados da API (`src/api/core/logger.py`) — registra latência de cada requisição
* MLflow Tracking — métricas de treino e avaliação registradas a cada run
* MLflow Registry — histórico de versões dos modelos com métricas associadas

A ser implementado:

* Alertas automáticos para queda de AUC ou mudança de distribuição nas features

---

## 8. Considerações Finais

O monitoramento é essencial para garantir que:

* O modelo continue relevante
* As decisões baseadas nele sejam confiáveis
* O impacto no negócio seja positivo

---
