# Desculpa gente, venho de Java, relembrando python pois só vi na faculdade,
# preciso comentar cada coisinha para eu lembrar depois

# test_smoke.py — smoke tests do pipeline de treinamento.
#
# Não verificamos valores exatos aqui. Apenas garantimos que as funções
# executam sem lançar exceção e retornam os tipos corretos.
# Isso é suficiente para detectar quebras de interface entre módulos.

import os

import numpy as np
import pandas as pd
import pytest
import torch.nn as nn
import torch.optim as optim

from src.model import ChurnMLP
from src.pipeline import apply_preprocessing, build_preprocessing_pipeline
from src.train_mlp import (
    build_model,
    create_dataloaders,
    evaluate_model,
    find_best_threshold,
    load_and_split_data,
)

DATA_PATH = "data/processed/telco_customer_churn_processed.csv"

# Pula todos os testes deste arquivo se o CSV processado não existir.
# Isso evita falhas em ambientes de CI que não possuem os dados.
pytestmark = pytest.mark.skipif(
    not os.path.exists(DATA_PATH),
    reason="CSV processado não encontrado — rode o pipeline de dados primeiro",
)


# ---------------------------------------------------------------------------
# Fixtures — dados compartilhados entre os testes do módulo
# ---------------------------------------------------------------------------
# scope="module" significa que a fixture é criada uma vez e reutilizada
# por todos os testes do arquivo, evitando recarregar o CSV a cada teste.

@pytest.fixture(scope="module")
def splits():
    """Carrega e divide os dados em treino, validação e teste."""
    return load_and_split_data(DATA_PATH)


@pytest.fixture(scope="module")
def preprocessed_splits(splits):
    """Aplica o pipeline de pré-processamento nos três conjuntos."""
    df_train, df_val, df_test = splits
    preprocessor = build_preprocessing_pipeline()
    return apply_preprocessing(preprocessor, df_train, df_val, df_test)


@pytest.fixture(scope="module")
def dataloaders(preprocessed_splits):
    """Cria os DataLoaders a partir dos dados pré-processados."""
    df_train, df_val, df_test = preprocessed_splits
    return create_dataloaders(df_train, df_val, df_test, batch_size=64)


# ---------------------------------------------------------------------------
# Testes de load_and_split_data
# ---------------------------------------------------------------------------

def test_split_retorna_tres_dataframes(splits):
    """Verifica que a divisão retorna exatamente 3 DataFrames."""
    assert len(splits) == 3
    for parte in splits:
        assert isinstance(parte, pd.DataFrame)


def test_split_preserva_total_de_linhas(splits):
    """Verifica que nenhuma linha é perdida ou duplicada na divisão."""
    df_original = pd.read_csv(DATA_PATH)
    total_apos_split = sum(len(parte) for parte in splits)
    assert total_apos_split == len(df_original)


def test_split_mantem_coluna_target(splits):
    """Verifica que a coluna target está presente em todos os conjuntos."""
    for parte in splits:
        assert "target" in parte.columns


# ---------------------------------------------------------------------------
# Testes do pipeline de pré-processamento
# ---------------------------------------------------------------------------

def test_preprocessing_normaliza_colunas_numericas(preprocessed_splits):
    """Verifica que o StandardScaler deixou as colunas numéricas com média ~0.

    Após o StandardScaler, cada coluna numérica deve ter média próxima de zero
    e desvio padrão próximo de 1 — essa é a definição da normalização z-score.
    Usamos abs() < 0.1 como tolerância razoável.
    """
    df_train, _, _ = preprocessed_splits
    assert abs(df_train["Tenure Months"].mean()) < 0.1
    assert abs(df_train["Monthly Charges"].mean()) < 0.1


def test_preprocessing_nao_altera_colunas_binarias(preprocessed_splits):
    """Verifica que as colunas binárias continuam com valores 0 ou 1."""
    df_train, _, _ = preprocessed_splits
    assert set(df_train["Gender_Male"].unique()).issubset({0, 1, 0.0, 1.0})


# ---------------------------------------------------------------------------
# Testes de create_dataloaders
# ---------------------------------------------------------------------------

def test_dataloaders_retorna_tres_loaders(dataloaders):
    """Verifica que create_dataloaders retorna exatamente 3 DataLoaders."""
    assert len(dataloaders) == 3


def test_dataloaders_tem_batches(dataloaders):
    """Verifica que cada DataLoader tem pelo menos um batch."""
    for loader in dataloaders:
        assert len(loader) > 0


# ---------------------------------------------------------------------------
# Testes de build_model
# ---------------------------------------------------------------------------

def test_build_model_retorna_tipos_corretos():
    """Verifica que build_model retorna modelo, critério e otimizador corretos."""
    model, criterion, optimizer = build_model(input_dim=28, learning_rate=0.001)
    assert isinstance(model, ChurnMLP)
    assert isinstance(criterion, nn.BCEWithLogitsLoss)
    assert isinstance(optimizer, optim.Adam)


# ---------------------------------------------------------------------------
# Testes de evaluate_model
# ---------------------------------------------------------------------------

def test_evaluate_model_retorna_arrays_corretos(dataloaders):
    """Verifica que evaluate_model retorna arrays com formato e valores corretos."""
    train_loader, _, test_loader = dataloaders
    input_dim = train_loader.dataset.X.shape[1]
    model, _, _ = build_model(input_dim=input_dim, learning_rate=0.001)

    probs, targets = evaluate_model(model, test_loader)

    assert isinstance(probs, np.ndarray)
    assert isinstance(targets, np.ndarray)
    assert len(probs) == len(targets)
    # Probabilidades após sigmoid devem estar em [0, 1]
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0


# ---------------------------------------------------------------------------
# Testes de find_best_threshold
# ---------------------------------------------------------------------------

def test_find_best_threshold_retorna_threshold_no_intervalo_valido():
    """Verifica que o threshold encontrado está entre 0.1 e 0.9."""
    probs = np.array([0.2, 0.8, 0.3, 0.9, 0.1, 0.7])
    targets = np.array([0, 1, 0, 1, 0, 1])

    threshold, preds = find_best_threshold(probs, targets)

    assert 0.1 <= threshold <= 0.9


def test_find_best_threshold_retorna_predicoes_binarias():
    """Verifica que as previsões retornadas contêm apenas 0 ou 1."""
    probs = np.array([0.2, 0.8, 0.3, 0.9, 0.1, 0.7])
    targets = np.array([0, 1, 0, 1, 0, 1])

    # '_' É uma convenção do Python para variável que você quer descartar — funciona como um "lixo".
    #
    #   find_best_threshold retorna dois valores: (threshold, preds). Nesse teste, só precisamos de preds para verificar se são binários — o valor do threshold em si não interessa aqui. Então usamos _ no lugar de dar um nome à variável, sinalizando para quem lê o
    #    código que o primeiro retorno foi ignorado intencionalmente.
    _, preds = find_best_threshold(probs, targets)

    assert set(preds).issubset({0, 1})
    assert len(preds) == len(targets)
