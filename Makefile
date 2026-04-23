# Makefile — centraliza os comandos mais usados do projeto em atalhos simples.
# Em vez de lembrar flags e caminhos longos, basta digitar 'make <alvo>'.
#
# Como usar:
#   make lint    → verifica qualidade do código com ruff
#   make test    → executa os testes automatizados com pytest
#   make run     → sobe a API FastAPI localmente
#   make train   → treina o modelo MLP

# .PHONY declara que esses alvos são comandos, não arquivos.
# Sem isso, se existir um arquivo chamado 'test' na pasta, o make o ignoraria.
.PHONY: lint test run train train-baselines

# Verifica o código em busca de erros de estilo e problemas lógicos.
# As regras são definidas no pyproject.toml (seção [tool.ruff]).
lint:
	ruff check src/ tests/

# Executa todos os testes com saída detalhada (-v = verbose).
test:
	pytest tests/ -v

# Inicia a API em modo desenvolvimento.
# --reload reinicia o servidor automaticamente ao salvar qualquer arquivo.
run:
	uvicorn src.api.main:app --reload

# Treina o modelo MLP e registra os experimentos no MLflow.
# Usa '-m' para rodar como módulo a partir da raiz do projeto,
# garantindo que os imports 'from src.X import Y' funcionem corretamente.
train:
	python -m src.train_mlp

# Treina os modelos de baseline (Regressão Logística e DummyClassifier) e salva em src/models/.
# Necessário rodar antes de subir a API pela primeira vez.
train-baselines:
	python -m src.train_baselines
