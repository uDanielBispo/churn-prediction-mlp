FROM python:3.11-slim

WORKDIR /app

# Copia apenas o pyproject.toml primeiro para instalar dependências.
# Assim, o Docker reutiliza essa camada em cache enquanto o pyproject.toml
# não mudar — mesmo que o código em src/ seja alterado.
COPY pyproject.toml ./
RUN apt-get update && apt-get install -y --no-install-recommends make \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip && pip install -e .

# Copia o código-fonte após as dependências para aproveitar o cache acima.
COPY src ./src

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
