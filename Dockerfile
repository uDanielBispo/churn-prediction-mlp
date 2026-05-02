FROM python:3.11-slim

WORKDIR /app

# Dependências do projeto
COPY pyproject.toml ./
COPY src ./src

COPY src/models ./src/models

RUN python -m pip install --upgrade pip
RUN pip install -e .

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]