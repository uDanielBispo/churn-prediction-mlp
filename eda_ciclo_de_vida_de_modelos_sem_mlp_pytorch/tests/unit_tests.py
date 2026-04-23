# unit_tests.py - Testes unitários que validam o funcionamento das funções em utils.py,
# garantindo que o pipeline opera corretamente antes do treino e registro do modelo.
#
# Pytest é o equivalente ao JUnit do Java:
# - cada função 'test_*' é um método de teste (como @Test no JUnit).
# - 'assert X' é a assertiva (como assertTrue/assertEquals no JUnit).
# - 'with pytest.raises(Erro)' valida que uma exceção foi lançada (equivalente a
#   assertThrows(Erro.class, ...)).
# Pra rodar:
#
# pytest tests/unit_tests.py -v

import os
import sys

# Adiciona o diretório src/ ao sys.path para conseguir importar o módulo utils.
# Necessário porque o pytest roda os testes de dentro de tests/, e o Python só
# adiciona automaticamente ao sys.path o diretório do arquivo que está executando.
# Sem isso, o 'from utils import ...' abaixo não encontraria o módulo.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
import pytest

from utils import load_data, split_data, compute_metrics


# ---------------------------------------------------------------------------
# Fixtures - dados de apoio reutilizáveis entre os testes
# ---------------------------------------------------------------------------
# No pytest, usamos "fixtures": funções decoradas com @pytest.fixture. Qualquer teste que
# receba o nome da fixture como parâmetro a executa automaticamente antes do
# teste rodar.

@pytest.fixture
def sample_csv(tmp_path):
    """Cria um CSV temporário com a estrutura mínima esperada pelo projeto.

    'tmp_path' é uma fixture embutida do pytest que cria um diretório temporário
    único por teste (e limpa no final). Usamos isso para não depender do CSV
    real do projeto.
    """
    # seed(0) fixa a semente aleatória - testes precisam ser determinísticos, ou
    # seja, sempre gerarem os mesmos valores a cada execução.
    np.random.seed(0)

    # n = 200 é suficiente para testar o split 80/20 (160 treino / 40 teste)
    # sem deixar o teste lento.
    n = 200

    # Os ranges (1-72, 5-100, 0-2) imitam valores plausíveis do dataset real,
    # mas não precisam ser realistas. Qualquer número serviria - o objetivo é só
    # ter um DataFrame com a estrutura esperada (features numéricas + target 0/1).
    df = pd.DataFrame({
        'Tenure Months': np.random.randint(1, 72, n),
        'Churn Score': np.random.randint(5, 100, n),
        'CLTV': np.random.randint(2000, 6500, n),
        'Partner_Yes': np.random.randint(0, 2, n),
        # target SÓ pode ser 0 ou 1 (churn / não-churn).
        'target': np.random.randint(0, 2, n),
    })

    path = tmp_path / 'test_data.csv'
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def sample_df(sample_csv):
    """Fixture derivada: carrega o CSV temporário como DataFrame pronto pra uso.
    Fixtures podem depender de outras fixtures - basta receber o nome como parâmetro.
    """
    return load_data(sample_csv)


# ---------------------------------------------------------------------------
# Testes de load_data
# ---------------------------------------------------------------------------
# Valida que a função de carregamento lê o CSV corretamente e trata erros.

class TestLoadData:
    def test_returns_dataframe(self, sample_csv):
        # Garante que load_data retorna um DataFrame, e não outro tipo
        # (string, None, lista, etc.).
        df = load_data(sample_csv)
        assert isinstance(df, pd.DataFrame)

    def test_has_target_column(self, sample_csv):
        # A coluna 'target' é obrigatória - sem ela, o treino não funciona.
        df = load_data(sample_csv)
        assert 'target' in df.columns

    def test_row_count(self, sample_csv):
        # Criamos o CSV com 200 linhas - ele deve voltar com 200 linhas.
        # Se vier com número diferente, algo deu errado na leitura.
        df = load_data(sample_csv)
        assert len(df) == 200

    def test_file_not_found_raises(self):
        # Se o arquivo não existe, load_data deve LANÇAR FileNotFoundError.
        # Equivalente ao assertThrows(FileNotFoundException.class, ...) do JUnit.
        with pytest.raises(FileNotFoundError):
            load_data('/caminho/inexistente/arquivo.csv')


# ---------------------------------------------------------------------------
# Testes de split_data
# ---------------------------------------------------------------------------
# Valida o comportamento da função que separa features/target e divide em
# treino/teste. Usa a fixture sample_df (200 linhas de dados fake).

class TestSplitData:
    def test_returns_four_splits(self, sample_df):
        # split_data deve retornar uma tupla de 4 elementos:
        # (X_train, X_test, y_train, y_test).
        result = split_data(sample_df)
        assert len(result) == 4

    def test_target_not_in_features(self, sample_df):
        # A coluna 'target' só pode estar em y - nunca em X.
        # Se estiver em X, o modelo estaria "trapaceando" (vendo a resposta).
        X_train, X_test, _, _ = split_data(sample_df)
        assert 'target' not in X_train.columns
        assert 'target' not in X_test.columns

    def test_total_rows_preserved(self, sample_df):
        # O split não pode perder nem duplicar linhas:
        # treino + teste = total original.
        X_train, X_test, y_train, y_test = split_data(sample_df)
        assert len(X_train) + len(X_test) == len(sample_df)
        assert len(y_train) + len(y_test) == len(sample_df)

    def test_default_test_size(self, sample_df):
        # Com test_size=0.2, o teste deve ter ~40 linhas (20% de 200).
        # pytest.approx(40, abs=2) = "40 com tolerância de +-2" (pode dar 38 a 42,
        # por causa do arredondamento do train_test_split ao estratificar).
        X_train, X_test, _, _ = split_data(sample_df, test_size=0.2)
        assert len(X_test) == pytest.approx(len(sample_df) * 0.2, abs=2)

    def test_y_values_are_binary(self, sample_df):
        # y só pode conter 0 ou 1 (classificação binária).
        # .issubset({0, 1}) = "os valores únicos estão contidos no conjunto {0, 1}".
        _, _, y_train, y_test = split_data(sample_df)
        assert set(y_train.unique()).issubset({0, 1})
        assert set(y_test.unique()).issubset({0, 1})

    def test_shapes_match(self, sample_df):
        # Número de linhas de X e y precisam bater - senão há desalinhamento
        # entre features e labels, e o fit do modelo quebraria.
        X_train, X_test, y_train, y_test = split_data(sample_df)
        assert X_train.shape[0] == len(y_train)
        assert X_test.shape[0] == len(y_test)


# ---------------------------------------------------------------------------
# Testes de compute_metrics
# ---------------------------------------------------------------------------
# Aqui os arrays são MUITO PEQUENOS e construídos à mão, para validar cenários
# matemáticos específicos. Arrays pequenos permitem verificar o resultado
# esperado mentalmente - se passasse 10.000 linhas aleatórias, não dava pra
# saber o que esperar das métricas.

class TestComputeMetrics:
    def test_returns_expected_keys(self):
        # Valida apenas que o dicionário retornado contém as 6 chaves esperadas.
        # Os VALORES aqui não importam - estamos testando o "contrato" da função.
        y = [0, 1, 0, 1]
        metrics = compute_metrics(y, y, y, y)
        expected = {
            'train_accuracy', 'test_accuracy', 'test_f1_score',
            'test_precision', 'test_recall', 'overfitting',
        }
        assert set(metrics.keys()) == expected

    def test_perfect_predictions(self):
        # Cenário extremo: y_test == y_pred_test (modelo acertou TUDO).
        # Passando o mesmo array nos 4 parâmetros, o modelo acerta 100% em treino
        # e em teste. Esperado:
        # - accuracy = 1.0 (100% de acerto)
        # - f1 = 1.0 (métrica máxima)
        # - overfitting = 0 (treino e teste têm a mesma accuracy)
        y = [0, 1, 0, 1, 0, 1]
        metrics = compute_metrics(y, y, y, y)
        assert metrics['test_accuracy'] == 1.0
        assert metrics['test_f1_score'] == 1.0
        assert metrics['overfitting'] == pytest.approx(0.0)

    def test_metrics_between_zero_and_one(self):
        # Valores diferentes entre predito e real - modelo erra alguns.
        # Não calculamos o valor exato de cada métrica, só validamos que todas
        # estão no intervalo [0, 1] (que é a definição matemática de accuracy,
        # F1, precision e recall). É um "smoke test" de sanidade.
        y_true = [0, 1, 0, 1, 0, 1]
        y_pred = [0, 1, 1, 0, 0, 1]
        metrics = compute_metrics(y_true, y_pred, y_true, y_pred)
        for key in ['test_accuracy', 'test_f1_score', 'test_precision', 'test_recall']:
            assert 0.0 <= metrics[key] <= 1.0

    def test_all_wrong_predictions(self):
        # Outro cenário extremo: y_pred é o OPOSTO de y_true.
        # Modelo errou todas as previsões, então accuracy deve ser 0.0.
        y_true = [0, 0, 1, 1]
        y_pred = [1, 1, 0, 0]
        metrics = compute_metrics(y_true, y_pred, y_true, y_pred)
        assert metrics['test_accuracy'] == 0.0

    def test_overfitting_positive_when_train_better(self):
        # Simula o clássico overfitting: modelo acerta 100% no treino mas
        # erra no teste. Como overfitting = train_accuracy - test_accuracy,
        # a diferença tem que ser POSITIVA. Isso valida que a fórmula captura
        # corretamente o conceito de overfitting.
        y_train_true = [0, 1, 0, 1]
        y_train_pred = [0, 1, 0, 1]  # acertou tudo no treino
        y_test_true  = [0, 1, 0, 1]
        y_test_pred  = [0, 1, 1, 0]  # errou 2 no teste
        metrics = compute_metrics(y_test_true, y_test_pred, y_train_true, y_train_pred)
        assert metrics['overfitting'] > 0
