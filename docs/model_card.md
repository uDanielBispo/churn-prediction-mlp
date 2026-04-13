# Model Card — Churn Prediction (Baseline)

## 1. Visão Geral

Este modelo tem como objetivo prever o churn de clientes em uma operadora de telecomunicações, identificando quais clientes possuem maior probabilidade de cancelamento.

A solução busca apoiar estratégias de retenção, permitindo ações preventivas direcionadas aos clientes com maior risco.

---

## 2. Tipo de Modelo

* Problema: Classificação binária
* Variável alvo: `target` (0 = não cancelou, 1 = cancelou)
* Modelos atuais:

  * Regressão Logística (baseline)
  * DummyClassifier (referência mínima)

> Observação: A implementação de uma rede neural (MLP com PyTorch) está em desenvolvimento.

---

## 3. Dataset

* Nome: Telco Customer Churn
* Tipo: Dados tabulares
* Contexto: Clientes de uma empresa de telecomunicações

### Principais características:

* Informações demográficas
* Tipo de contrato
* Serviços contratados
* Tempo de relacionamento
* Faturamento e pagamento

---

## 4. Features Utilizadas

Foram utilizadas 25 features após pré-processamento e encoding.

### Variáveis numéricas:

* Tenure Months
* Churn Score
* CLTV

### Variáveis categóricas (codificadas):

* Partner_Yes, Dependents_Yes
* Internet Service_Fiber optic, Internet Service_No
* Serviços adicionais (segurança, backup, suporte, streaming)
* Contract_One year, Contract_Two year
* Paperless Billing_Yes
* Payment Method (diferentes categorias)

Essas variáveis foram selecionadas com base na análise exploratória e representam fatores relevantes para o churn.

---

## 5. Métricas de Avaliação

Os modelos são avaliados utilizando:

* AUC-ROC
* F1-Score
* Precision
* Recall

A escolha dessas métricas considera o impacto de erros no contexto de churn, especialmente:

* Falsos negativos: clientes que cancelam sem serem identificados
* Falsos positivos: clientes classificados como risco sem necessidade

---

## 6. Resultados (Baseline)

Os modelos baseline apresentam desempenho inicial satisfatório, servindo como referência para evolução do modelo.

> Observação: Os valores numéricos serão atualizados após consolidação dos experimentos e comparação com a MLP.

---

## 7. Limitações

* O modelo é treinado com dados históricos e pode não refletir mudanças futuras no comportamento dos clientes
* Possível presença de variáveis correlacionadas que impactam a interpretabilidade
* Ausência de variáveis externas (ex: concorrência, contexto econômico)
* Baselines ainda não capturam relações não lineares complexas

---

## 8. Vieses e Riscos

* Possível viés relacionado a perfis específicos de clientes (ex: tipo de contrato ou serviços contratados)
* O modelo pode favorecer padrões majoritários do dataset
* Risco de decisões automatizadas impactarem negativamente determinados grupos de clientes

---

## 9. Uso Recomendado

O modelo deve ser utilizado como ferramenta de apoio à decisão, e não como único critério.

Aplicações recomendadas:

* Identificação de clientes com risco de churn
* Priorização de ações de retenção
* Suporte a campanhas de marketing

---

## 10. Uso Não Recomendado

* Decisões automatizadas sem validação humana
* Uso em contextos diferentes do dataset original
* Aplicação direta sem monitoramento contínuo

---

## 11. Próximos Passos

* Implementar modelo MLP com PyTorch
* Comparar desempenho com baseline
* Integrar MLflow para rastreamento de experimentos
* Disponibilizar modelo via API (FastAPI)
* Atualizar este Model Card com métricas finais

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