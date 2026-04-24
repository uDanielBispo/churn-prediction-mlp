# Resultado da Refatoração — Churn Prediction MLP

---

## 1. Infraestrutura do projeto

**Antes:** nenhum arquivo de configuração central. Dependências espalhadas só no `requirements.txt`, sem separação entre produção e desenvolvimento. Para rodar qualquer comando era preciso lembrar flags longas.

**Depois:**

| Arquivo | O que resolve |
|---|---|
| `pyproject.toml` | Declaração formal de dependências (prod e dev separadas), configuração do ruff e pytest em um só lugar, projeto instalável com `pip install -e .` |
| `Makefile` | `make lint`, `make test`, `make run`, `make train` — ninguém precisa lembrar flags |

---

## 2. `src/train_mlp.py` — eliminação da função Deus

**Antes:** uma única função `train_model()` com ~160 linhas fazendo tudo: ler dados, dividir, criar datasets, criar dataloaders, instanciar modelo, treinar, avaliar, calcular threshold, logar métricas e salvar. Impossível testar qualquer etapa isoladamente.

**Depois:** 7 funções com responsabilidade única:

| Função | Responsabilidade |
|---|---|
| `load_and_split_data()` | Carrega CSV e divide em treino/val/teste |
| `create_dataloaders()` | Converte DataFrames em DataLoaders PyTorch |
| `build_model()` | Instancia MLP, critério e otimizador |
| `run_training_loop()` | Loop de épocas com early stopping |
| `evaluate_model()` | Coleta probabilidades no conjunto de teste |
| `find_best_threshold()` | Busca o threshold que maximiza F1 |
| `log_metrics()` | Calcula, imprime e registra métricas |
| `train_model()` | Orquestra as funções acima (~20 linhas) |

Também removido: `import argparse` sem uso (código morto) e `root_dir` calculado duas vezes virou a constante `ROOT_DIR`.

---

## 3. Pipeline de pré-processamento — data leakage eliminado

**Antes:** os dados eram passados direto ao `ChurnDataset` sem nenhuma normalização. `Tenure Months` (valores 1–72) e `Monthly Charges` (valores ~20–120) estavam em escalas muito diferentes das colunas binárias (0 ou 1), prejudicando a convergência da rede.

**Depois:** `src/pipeline.py` com `ColumnTransformer` do sklearn:

```
StandardScaler  → Tenure Months, Monthly Charges
passthrough     → 26 colunas binárias
```

A regra de ouro é garantida: `fit()` apenas no treino, `transform()` nos três conjuntos. O preprocessor é salvo em `src/models/preprocessor.pkl` para ser reutilizado pela API.

**Efeito observado:** com as features normalizadas, o early stopping passou da época 30 para a 40 — a rede convergiu de forma mais estável.

---

## 4. API FastAPI — observabilidade e duplicação

**Antes:**
- Arquivo com typo no nome: `loggin.py`
- `logging.basicConfig()` sem formato definido — saída sem timestamp, sem nível, sem módulo
- `print(data)` esquecido em `routes.py` (código de debug em produção)
- Array de 28 features copiado manualmente em cada uma das 3 funções de predição (duplicação total)
- Nenhum registro de quanto tempo cada requisição levava

**Depois:**

| Melhoria | Detalhe |
|---|---|
| `logger.py` (nome corrigido) | Formato estruturado: `2026-04-23 17:30:00 \| INFO \| src.api.routes \| mensagem` |
| Middleware de latência | Toda requisição registra: `POST /predict → 200 (4.3ms)` |
| `print(data)` removido | Substituído por `logger.info()` com contexto |
| `_build_features()` extraída | Array de features centralizado — uma única alteração atualiza os 3 modelos |

---

## 5. Testes — de zero para 29

**Antes:** nenhum teste para o pipeline da MLP, API ou qualidade dos dados.

**Depois:**

| Arquivo | Tipo | O que valida |
|---|---|---|
| `test_smoke.py` | Fumaça | 11 testes: cada função do pipeline retorna os tipos corretos e não lança exceção |
| `test_schema.py` | Contrato de dados | 8 testes: colunas, tipos, ausência de nulos, proporção de churn, intervalos de valores |
| `test_api.py` | Endpoints | 10 testes: `/health`, `/predict` com 3 modelos, validação Pydantic (422), model_type inválido |

---

## 6. Organização do repositório

**Antes:** `train.py`, `register.py`, `utils.py` e `unit_tests.py` misturados com os arquivos da MLP em `src/` e `tests/`.

**Depois:** trabalho da fase 1 isolado em `eda_ciclo_de_vida_de_modelos_sem_mlp_pytorch/` com sua própria estrutura `src/` e `tests/`, sem interferir no código principal.

---

## Resumo

> O projeto saiu de scripts lineares sem testes e com logging inexistente para um pipeline modular, com pré-processamento reprodutível, API observável e 29 testes automatizados cobrindo dados, modelo e endpoints.
