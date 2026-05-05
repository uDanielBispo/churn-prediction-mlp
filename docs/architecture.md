# Architecture — Churn Prediction System

---

## 1. Visão Geral

Este documento descreve a arquitetura do sistema de previsão de churn, cobrindo o fluxo completo desde os dados até a inferência via API.

O projeto segue uma abordagem modular, separando claramente:

* Processamento de dados
* Pipeline de Machine Learning
* Treinamento de modelos
* Serviço de inferência (API)
* Testes e validação

---

## 2. Fluxo do Sistema

O pipeline do sistema pode ser representado da seguinte forma:

```
Dados Brutos (data/raw)
        ↓
Dados Processados (data/processed)
        ↓
Pipeline de Pré-processamento (src/pipeline.py)
        ↓
Treinamento dos Modelos (baseline + MLP)
        ↓
Modelos Treinados + Preprocessor (.pkl / .pth)
        ↓
API (FastAPI)
        ↓
Validação de Entrada (schemas)
        ↓
Serviço de Inferência (model_service)
        ↓
Resposta com probabilidade de churn
```

---

## 3. Componentes do Sistema

### Data Layer

* `data/raw/` → dados originais (imutáveis)
* `data/processed/` → dados tratados

Responsável por armazenar e versionar os dados utilizados no projeto.

---

### Pipeline de Dados

Arquivo: `src/pipeline.py`

Responsável por:

* Padronização dos dados (StandardScaler)
* Garantia de consistência entre treino e inferência
* Prevenção de data leakage

O pipeline é salvo como artefato (`preprocessor.pkl`) e reutilizado pela API.

---

### Data Handling

Arquivo: `src/dataset.py`

Responsável por:

* Carregamento dos dados
* Preparação para uso em PyTorch
* Conversão para tensores

---

### Model Layer

Arquivos principais:

* `src/model.py` → arquitetura da MLP
* `src/train_baselines.py` → modelos baseline (Dummy + Logística)
* `src/train_mlp.py` → treinamento da rede neural
* `src/early_stopping.py` → controle de overfitting

Responsável por:

* Treinamento dos modelos
* Avaliação
* Geração de artefatos

---

### Experiment Tracking

Ferramenta: **MLflow**

Responsável por:

* Registro de métricas
* Comparação entre modelos
* Versionamento de experimentos

---

### API Layer

Local: `src/api/`

Componentes:

* `main.py` → inicialização da aplicação e middlewares
* `routes.py` → definição dos endpoints
* `schemas.py` → validação de entrada (Pydantic)
* `services/model_service.py` → lógica de inferência
* `core/logger.py` → logging estruturado

Responsável por:

* Servir o modelo em produção
* Receber requisições externas
* Retornar previsões

---

### Test Layer

Local: `tests/`

Componentes:

* `test_smoke.py` → valida pipeline básico
* `test_schema.py` → valida estrutura dos dados
* `test_api.py` → valida endpoints da API

Responsável por:

* Garantir integridade do sistema
* Evitar regressões

---

### Infraestrutura

Arquivos:

* `Dockerfile`
* `docker-compose.yml`
* `Makefile`
* `pyproject.toml`

Responsável por:

* Padronizar execução
* Facilitar deploy
* Automatizar tarefas

---

## 4. Comunicação entre Componentes

* O pipeline é utilizado tanto no treinamento quanto na inferência
* O modelo treinado é salvo e carregado pela API
* A API utiliza o `model_service` para realizar predições
* Os dados de entrada são validados via `schemas.py`

---

## 5. Padrões de Projeto Utilizados

* Separação de responsabilidades
* Arquitetura modular
* Service Layer na API
* Validação de dados com schemas (Pydantic)
* Pipeline reutilizável (treino e inferência)

---

## 6. Escalabilidade

O sistema foi projetado para permitir evolução futura:

* Deploy em ambiente cloud
* Containerização com Docker (já implementado)
* Monitoramento contínuo
* Integração com pipelines automatizados

---

## 7. Considerações Finais

A arquitetura foi projetada para:

* Garantir reprodutibilidade
* Facilitar manutenção
* Separar claramente treino e produção
* Permitir evolução para um sistema real em produção

---
