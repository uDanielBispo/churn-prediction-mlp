# Architecture — Churn Prediction System

## 1. Visão Geral

Este documento descreve a arquitetura do sistema de previsão de churn, cobrindo o fluxo completo desde os dados até a inferência via API.

O projeto segue uma abordagem modular, separando claramente:

* Processamento de dados
* Treinamento de modelos
* Serviço de inferência (API)

---

## 2. Fluxo do Sistema

O pipeline do sistema pode ser representado da seguinte forma:

```
Dados Brutos (data/raw)
        ↓
Processamento de Dados (src/dataset.py)
        ↓
Feature Engineering
        ↓
Treinamento do Modelo (src/train.py / src/train_mlp.py)
        ↓
Modelo Treinado (artefato salvo)
        ↓
Serviço de Inferência (src/api/)
        ↓
Endpoint /predict (FastAPI)
        ↓
Resposta com probabilidade de churn
```

---

## 3. Componentes do Sistema

### 📁 Data Layer

* `data/raw/` → dados originais
* `data/processed/` → dados tratados

Responsável por armazenar e versionar os dados utilizados no projeto.

---

### ⚙️ Data Processing

Arquivo: `src/dataset.py`

Responsável por:

* Carregamento dos dados
* Limpeza e transformação
* Preparação das features para treino

---

### 🤖 Model Layer

Arquivos principais:

* `src/model.py` → definição da arquitetura da MLP
* `src/train.py` → treinamento de modelos baseline
* `src/train_mlp.py` → treinamento da rede neural
* `src/early_stopping.py` → controle de overfitting

Responsável por:

* Treinamento dos modelos
* Validação
* Geração de artefatos

---

### 🧠 Model Management

Arquivo: `src/register.py`

Responsável por:

* Registro e armazenamento de modelos treinados
* Gerenciamento de artefatos

---

### 🌐 API Layer

Local: `src/api/`

Componentes:

* `main.py` → inicialização da aplicação
* `routes.py` → definição dos endpoints
* `schemas.py` → validação de entrada (Pydantic)
* `services/model_service.py` → lógica de inferência
* `core/loggin.py` → configuração de logs

Responsável por:

* Servir o modelo em produção
* Receber requisições externas
* Retornar previsões

---

## 4. Comunicação entre Componentes

* O modelo treinado é salvo e posteriormente carregado pela API
* A API utiliza o `model_service` para realizar inferência
* As entradas são validadas via `schemas.py` antes do processamento

---

## 5. Padrões de Projeto Utilizados

* Separação de responsabilidades (data, model, API)
* Arquitetura modular
* Service Layer na API
* Uso de schemas para validação de dados

---

## 6. Escalabilidade (Futuro)

Possíveis melhorias:

* Deploy em ambiente cloud
* Containerização com Docker
* Orquestração de pipelines
* Integração com MLflow

---

## 7. Considerações Finais

A arquitetura foi projetada para:

* Facilitar manutenção
* Permitir reprodutibilidade
* Suportar evolução para produção real

---
