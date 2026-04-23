# test_schema.py — validação do schema do CSV processado com Pandera.
#
# Pandera é uma biblioteca de validação de DataFrames. Funciona como um
# "contrato formal" entre o pipeline de dados e o modelo:
# se alguém alterar o CSV (remover coluna, mudar tipo, inserir nulos),
# esses testes falham imediatamente — antes que o modelo quebre em produção.
#
# É o equivalente a um schema de banco de dados, mas para DataFrames pandas.

import os

import pandas as pd
import pandera.pandas as pa
import pytest
from pandera.pandas import Check, Column, DataFrameSchema

DATA_PATH = "data/processed/telco_customer_churn_processed.csv"

pytestmark = pytest.mark.skipif(
    not os.path.exists(DATA_PATH),
    reason="CSV processado não encontrado",
)

# ---------------------------------------------------------------------------
# Schema formal do dataset processado
# ---------------------------------------------------------------------------
# Cada Column define:
#   - dtype: tipo esperado da coluna (int, float, bool, etc.)
#   - Check: validação do conteúdo (valores mínimos, conjunto permitido, etc.)
#   - nullable=False (padrão): não permite valores nulos

CHURN_SCHEMA = DataFrameSchema(
    columns={
        # Colunas numéricas contínuas
        "Tenure Months": Column(int, Check.greater_than_or_equal_to(0)),
        "Monthly Charges": Column(float, Check.greater_than_or_equal_to(0.0)),
        # Variável alvo — apenas 0 ou 1
        "target": Column(int, Check.isin([0, 1])),
        # Colunas binárias (True/False no CSV)
        "Gender_Male": Column(bool),
        "Partner_Yes": Column(bool),
        "Dependents_Yes": Column(bool),
        "Phone Service_Yes": Column(bool),
        "Multiple Lines_No phone service": Column(bool),
        "Multiple Lines_Yes": Column(bool),
        "Internet Service_Fiber optic": Column(bool),
        "Internet Service_No": Column(bool),
        "Online Security_No internet service": Column(bool),
        "Online Security_Yes": Column(bool),
        "Online Backup_No internet service": Column(bool),
        "Online Backup_Yes": Column(bool),
        "Device Protection_No internet service": Column(bool),
        "Device Protection_Yes": Column(bool),
        "Tech Support_No internet service": Column(bool),
        "Tech Support_Yes": Column(bool),
        "Streaming TV_No internet service": Column(bool),
        "Streaming TV_Yes": Column(bool),
        "Streaming Movies_No internet service": Column(bool),
        "Streaming Movies_Yes": Column(bool),
        "Contract_One year": Column(bool),
        "Contract_Two year": Column(bool),
        "Paperless Billing_Yes": Column(bool),
        "Payment Method_Credit card (automatic)": Column(bool),
        "Payment Method_Electronic check": Column(bool),
        "Payment Method_Mailed check": Column(bool),
    }
)


@pytest.fixture(scope="module")
def df():
    """Carrega o CSV processado uma única vez para todos os testes do módulo."""
    return pd.read_csv(DATA_PATH)


# ---------------------------------------------------------------------------
# Testes de schema
# ---------------------------------------------------------------------------

def test_schema_valida_sem_erros(df):
    """Verifica que o CSV completo passa na validação do schema Pandera.

    Se qualquer coluna tiver tipo errado ou valor fora do esperado,
    o Pandera lança SchemaError com detalhes de qual linha/coluna falhou.
    """
    CHURN_SCHEMA.validate(df)


def test_dataset_nao_tem_valores_nulos(df):
    """Verifica que não há valores ausentes (NaN) em nenhuma coluna.

    Valores nulos causam erros silenciosos durante o treino — o modelo
    pode ignorá-los ou produzir previsões incorretas sem aviso.
    """
    total_nulos = df.isnull().sum().sum()
    assert total_nulos == 0, f"Encontrados {total_nulos} valores nulos no dataset"


def test_dataset_tem_todas_as_colunas_esperadas(df):
    """Verifica que todas as colunas do schema estão presentes no CSV."""
    colunas_esperadas = set(CHURN_SCHEMA.columns.keys())
    colunas_presentes = set(df.columns)
    colunas_ausentes = colunas_esperadas - colunas_presentes
    assert not colunas_ausentes, f"Colunas ausentes no CSV: {colunas_ausentes}"


def test_target_e_binario(df):
    """Verifica que a variável alvo contém apenas os valores 0 e 1."""
    valores_unicos = set(df["target"].unique())
    assert valores_unicos.issubset({0, 1})


def test_dataset_tem_registros_suficientes(df):
    """Verifica que o dataset tem pelo menos 1000 linhas.

    Um dataset muito pequeno pode indicar que o arquivo foi corrompido
    ou que apenas uma amostra foi salva por engano.
    """
    assert len(df) >= 1000, f"Dataset com apenas {len(df)} linhas — muito pequeno"


def test_proporcao_de_churn_e_razoavel(df):
    """Verifica que a taxa de churn está entre 10% e 50%.

    Uma taxa fora desse intervalo pode indicar problema no processamento
    ou que a variável target foi invertida acidentalmente.
    """
    taxa_churn = df["target"].mean()
    assert 0.10 <= taxa_churn <= 0.50, (
        f"Taxa de churn fora do intervalo esperado: {taxa_churn:.2%}"
    )


def test_tenure_months_dentro_do_intervalo_esperado(df):
    """Verifica que Tenure Months está entre 0 e 72 meses (6 anos)."""
    assert df["Tenure Months"].min() >= 0
    assert df["Tenure Months"].max() <= 72


def test_monthly_charges_dentro_do_intervalo_esperado(df):
    """Verifica que Monthly Charges está entre 0 e 200 dólares."""
    assert df["Monthly Charges"].min() >= 0.0
    assert df["Monthly Charges"].max() <= 200.0
