Tech Challenge 2
TECH CHALLENGE
Atividade em grupo · Obrigatória · Avaliada.
Entrega obrigatória: Repositório GitHub + Vídeo de 5 minutos (método
STAR).
Entrega opcional: Deploy em ambiente de produção em nuvem (AWS,
Azure ou GCP).
Tema Central: Rede Neural para Previsão de Churn com Pipeline Profissional End-to-End
Contexto
Uma operadora de telecomunicações está perdendo clientes em ritmo
acelerado. A diretoria precisa de um modelo preditivo de churn que classifique
clientes com risco de cancelamento. Assim, o grupo deve construir o projeto do
zero ao modelo servido via API, aplicando todas as boas práticas de engenharia
de ML aprendidas na Fase 1.
O modelo central de entrega é uma rede neural (MLP), treinada com
PyTorch, comparada com baselines (Scikit-Learn) e rastreada com MLflow.
Requisitos Obrigatórios
Repositório GitHub
• Estrutura organizada: src/, data/, models/, tests/, notebooks/, docs/.
• README.md completo com instruções de setup, execução e
descrição do projeto.
• pyproject.toml como single source of truth (dependências, linting,
pytest).
• Histórico de commits limpo e significativo (não 1 commit gigante).
• .gitignore adequado para projetos de ML.
Tech Challenge 3
Vídeo (5 minutos — método STAR)
• Situation: Qual o problema de negócio e o contexto do dataset?
• Task: Qual a tarefa do grupo e os objetivos técnicos?
• Action: Quais decisões técnicas foram tomadas (arquitetura, features,
modelo, métricas)?
• Result: Quais os resultados obtidos e as lições aprendidas?
Bibliotecas Requeridas
• PyTorch — construção e treinamento da rede neural (MLP).
• Scikit-Learn — pipelines de pré-processamento e modelos baseline.
• MLflow — tracking de experimentos (parâmetros, métricas, artefatos).
• FastAPI — API de inferência do modelo.
Boas Práticas Obrigatórias
• Seeds fixados para reprodutibilidade.
• Validação cruzada estratificada.
• Model Card documentando limitações e vieses.
• Testes automatizados (≥ 3: smoke test, schema, API).
• Logging estruturado (sem print()).
• Linting com ruff sem erros.
Etapas de Desenvolvimento (4 Etapas)
Etapa 1 — Entendimento e Preparação (Disciplinas 01 e 02)
Foco: formulação do problema, exploração de dados e construção de
baselines.
Tarefa Referência
Preencher ML Canvas (stakeholders, métricas de negócio, SLOs).
Ciclo de Vida, Aula 01
Tech Challenge 4
EDA completa: volume, qualidade, distribuição, data readiness.
Ciclo de Vida, Aula 01
Definir métrica técnica (AUC-ROC, PR-AUC,
F1) e métrica de negócio (custo de churn evitado).
Fundamentos, Aula 05
Treinar baseline com DummyClassifier e Regressão Logística (Scikit-Learn).
Fundamentos, Aulas 01–02
Registrar experimentos no MLflow (parâmetros, métricas, dataset version).
Ciclo de Vida, Aula 02
Entregável: notebook de EDA + baselines registrados no MLflow.
Etapa 2 — Modelagem com Redes Neurais (Disciplina 02)
Foco: Construção, treinamento e avaliação de MLP com PyTorch.
Tarefa Referência
Construir MLP em PyTorch: definir arquitetura, função de ativação, loss function.
Fundamentos, Aula 04
Implementar loop de treinamento com early
stopping e batching.
Fundamentos, Aula 04
Comparar MLP vs. baselines (lineares + árvores) usando ≥ 4 métricas.
Fundamentos, Aula 05
Analisar trade-off de custo (falso positivo
vs. negativo).
Fundamentos, Aula 05
Registrar todos os experimentos (MLP e ensembles) no MLflow.
Ciclo de Vida, Aula 02
Entregável: tabela comparativa de modelos + MLP treinado + artefatos
no MLflow.
Tech Challenge 5
Etapa 3 — Engenharia e API (Disciplinas 03, 04 e 05)
Foco: refatoração profissional, API de inferência e pacote reutilizável.
Tarefa Referência
Refatorar código em módulos (src/) com estrutura limpa.
Eng. Software, Aula 01
Criar pipeline reprodutível (sklearn + transformadores custom).
Eng. Software, Aula 01; Bibliotecas, Aula 02
Escrever testes (pytest): unitários, schema
(pandera), smoke test.
Eng. Software, Aula 03
Construir API FastAPI: /predict, /health, validação Pydantic.
APIs, Aulas 01–03
Adicionar logging estruturado e middleware
de latência.
APIs, Aula 04
Configurar pyproject.toml, ruff, Makefile (lint,
test, run).
Eng. Software, Aulas 04–05
Entregável: repositório refatorado + API funcional + testes passando.
Etapa 4 — Documentação e Entrega Final (Todas as disciplinas)
Foco: consolidação, documentação e vídeo de apresentação.
Tarefa Referência
Gerar Model Card completo (performance, limitações, vieses, cenários de falha).
Ciclo de Vida, Aula 03
Documentar arquitetura de deploy escolhida
(batch vs. real-time) + justificativa.
Ciclo de Vida, Aula 04
Criar plano de monitoramento (métricas, alertas, playbook de resposta).
Ciclo de Vida, Aula 05
Finalizar README com instruções de setup +
execução + arquitetura.
Eng. Software / APIs
Tech Challenge 6
Gravar vídeo de 5 min (método STAR) demonstrando o projeto.
—
(Opcional) Deploy da API em nuvem
(AWS/Azure/GCP) com endpoint público.
—
Entregável: repositório final + vídeo STAR + (opcional) URL do deploy
em nuvem.
Critérios de Avaliação
Critério Peso Descrição
Qualidade do código e estrutura.
20% Organização, modularidade,
SOLID, linting sem erros.
Rede neural (PyTorch). 25% MLP funcional, treinamento com
early stopping, comparação
com baselines.
Pipeline e reprodutibilidade. 15% Pipeline sklearn, seeds, pyproject.toml, instala do zero.
API de inferência. 15% FastAPI funcional, Pydantic, logging, testes passando.
Documentação e Model
Card.
10% Model Card completa, README
claro, plano de monitoramento.
Vídeo STAR. 10% Clareza, cobertura dos quatro
elementos STAR, dentro de
cinco minutos.
Bônus: deploy em nuvem. 5% API acessível via URL pública.
Dataset Sugerido
Dataset público de telecomunicações com variáveis tabulares (ex.: Telco
Customer Churn — IBM). Alternativas: qualquer dataset de classificação binária
com ≥ 5.000 registros e ≥ 10 features.
Tech Challenge 7
Passo a Passo Resumido
• [Etapa 1] EDA + ML Canvas + Baselines → MLflow tracking.
• [Etapa 2] MLP PyTorch + comparação de modelos + análise de custo.
• [Etapa 3] Refatoração + API FastAPI + testes + Makefile.
• [Etapa 4] Model Card + README + vídeo STAR + (opcional) deploy
em nuvem.
Caso tenha qualquer dúvida, não deixe de nos procurar no Discord!