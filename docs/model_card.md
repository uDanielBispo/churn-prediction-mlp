# Model Card — Churn Prediction (MLP)

## 1. Visão Geral

Este modelo tem como objetivo prever a probabilidade de churn (cancelamento) de clientes em uma empresa de telecomunicações.

* **Tipo de problema:** Classificação binária
* **Saída:** Probabilidade de churn (0 a 1) e classificação (churn / não churn)
* **Objetivo de negócio:** Identificar clientes com alto risco de cancelamento para permitir ações proativas de retenção

### Aplicações no negócio:

* Priorização de clientes para campanhas de retenção
* Redução do churn rate
* Otimização de custos de marketing

---

## 2. Dados Utilizados

* **Dataset:** Telco Customer Churn
* **Tipo:** Dados tabulares
* **Volume:** Dataset com milhares de registros de clientes

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
* Tratamento de valores ausentes (quando aplicável)

### Considerações importantes:

* Possível desbalanceamento entre classes (churn vs não churn)
* Exclusão de variáveis com risco de data leakage

---

## 3. Metodologia e Modelagem

### Pipeline:

1. Análise exploratória dos dados (EDA)
2. Feature engineering
3. Treinamento de modelos baseline
4. Treinamento de rede neural (MLP)
5. Avaliação e comparação

### Modelos utilizados:

* DummyClassifier (baseline mínimo)
* Regressão Logística
* **MLP (Multi-Layer Perceptron com PyTorch)**

### MLP:

* Arquitetura definida em `src/model.py`
* Treinamento via `src/train_mlp.py`
* Early stopping para evitar overfitting

### Divisão dos dados:

* Treino / validação / teste *(ajustável no pipeline)*

---

## 4. Métricas de Performance

Os modelos são avaliados utilizando:

* ROC-AUC
* Precision
* Recall
* F1-Score

### Considerações:

* O problema de churn prioriza **Recall**, pois:

  * É mais crítico identificar clientes que irão cancelar
* Falsos positivos são aceitáveis dentro de limites operacionais

> Os valores específicos das métricas devem ser atualizados conforme experimentos finais.

---

## 5. Interpretação do Modelo

Com base na análise exploratória e features selecionadas:

### Principais fatores associados ao churn:

* Baixo tempo de relacionamento
* Contratos mensais (vs contratos longos)
* Ausência de serviços adicionais (ex: suporte técnico)
* Métodos de pagamento específicos (ex: electronic check)

> A interpretação pode ser aprofundada com técnicas como feature importance ou SHAP (não implementado nesta versão).

---

## 6. Limitações e Riscos

* O modelo é baseado em dados históricos e pode não refletir mudanças futuras
* Possível presença de variáveis correlacionadas
* Não considera fatores externos (ex: concorrência, mercado)
* MLP pode ter menor interpretabilidade comparado a modelos lineares

---

## 7. Fairness e Viés

* Não foram identificadas variáveis sensíveis explícitas (ex: raça, gênero)
* Avaliações de fairness não foram realizadas nesta versão

> Recomenda-se análise futura caso o modelo seja aplicado em produção real.

---

## 8. Deploy e Uso em Produção

O modelo foi preparado para uso via API utilizando FastAPI.

### Arquitetura:

* API definida em `src/api/`
* Endpoint principal:

  * `POST /predict` → retorna previsão de churn

### Uso esperado:

* Integração com sistemas de CRM
* Execução em batch ou near real-time

---

## 9. Monitoramento

Para uso em produção, recomenda-se monitorar:

* Performance do modelo (AUC, Recall)
* Data drift nas features
* Taxa de churn ao longo do tempo

### Ações recomendadas:

* Re-treinamento periódico
* Alertas para degradação de performance

---

## 10. Reprodutibilidade

O projeto foi estruturado para garantir reprodutibilidade:

* Código modular em `src/`
* Notebooks para experimentação
* API desacoplada para inferência

### Execução:

* Treinamento: `python src/train_mlp.py`
* API: `uvicorn src.api.main:app --reload`

Dependências disponíveis em `requirements.txt`.

---

## 11. Impacto de Negócio (Simulado)

O modelo permite:

* Identificar clientes com maior risco de churn
* Priorizar ações de retenção
* Reduzir perdas de receita

Exemplo de uso:

* Selecionar os **top 10% clientes com maior risco**
* Direcionar campanhas específicas


---

## 12. Responsáveis

Projeto desenvolvido em grupo como parte de desafio acadêmico de Machine Learning.

## 13. Features Selecionadas

| Coluna (Source) | Descrição | Foi usada no final? | Justificativa |
|----------------|----------|----------------------|---------------|
| CustomerID | Identificador único do cliente | Não | ID sem valor preditivo, causa overfitting |
| Count | Valor constante (1) | Não | Sem variância, não agrega informação |
| Country | País do cliente | Não | Sem variância |
| State | Estado | Não | Baixa relevância |
| City | Cidade | Não | Alta cardinalidade |
| Zip Code | Código postal | Não | Baixo valor preditivo |
| Lat Long | Coordenadas combinadas | Não | Redundante |
| Latitude | Coordenada geográfica | Não | Baixa relevância |
| Longitude | Coordenada geográfica | Não | Baixa relevância |
| Gender | Gênero | Sim | Pode influenciar comportamento |
| Senior Citizen | Idoso | Não | Baixa relevância |
| Partner | Possui parceiro | Sim | Indica estabilidade |
| Dependents | Dependentes | Sim | Relacionado à retenção |
| Tenure Months | Tempo como cliente | Sim | Muito importante |
| Phone Service | Serviço telefônico | Sim | Uso de serviço |
| Multiple Lines | Múltiplas linhas | Sim | Engajamento |
| Internet Service | Tipo de internet | Sim | Impacta churn |
| Online Security | Segurança | Sim | Reduz churn |
| Online Backup | Backup | Sim | Uso do serviço |
| Device Protection | Proteção | Sim | Retenção |
| Tech Support | Suporte | Sim | Muito relevante |
| Streaming TV | Streaming TV | Sim | Comportamento |
| Streaming Movies | Streaming filmes | Sim | Comportamento |
| Contract | Tipo de contrato | Sim | Muito relevante |
| Paperless Billing | Fatura digital | Sim | Perfil cliente |
| Payment Method | Método pagamento | Sim | Impacta churn |
| Monthly Charges | Valor mensal | Sim | Influencia cancelamento |
| Total Charges | Total gasto | Não | Redundante |
| Churn Label | Label textual | Não | Redundante |
| Churn Value | Target | Sim | Variável alvo |
| Churn Score | Score churn | Não | Data leakage |
| CLTV | Lifetime value | Não | Data leakage |
| Churn Reason | Motivo churn | Não | Data leakage |