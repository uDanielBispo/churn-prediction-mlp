# train.py - Responsável por ler os dados e treinar o modelo de predição de churn.

import os

from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
import mlflow
import mlflow.sklearn

from utils import load_data, split_data, compute_metrics

# Constantes de caminhos (relativos à raiz do projeto)
DATA_PATH = 'data/processed/telco_customer_churn_processed.csv'
MLFLOW_DB_PATH = os.path.join(os.getcwd(), 'mlflow.db')
MLRUNS_PATH = os.path.join(os.getcwd(), 'mlruns')

# Constantes de experimentos
EXPERIMENT_LOGISTIC = 'chrun_prediction_logistic_regression'
EXPERIMENT_DUMMY = 'chrun_prediction_dummy_classifier'


def setup_mlflow():
    """Configura o tracking URI e o registry URI do MLflow."""
    mlflow.set_tracking_uri(f'sqlite:///{MLFLOW_DB_PATH}')
    mlflow.set_registry_uri(f'file://{MLRUNS_PATH}')


def get_or_create_experiment(experiment_name):
    """Cria o experimento caso ainda não exista e o ativa."""
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(
            name=experiment_name,
            artifact_location=f'file://{MLRUNS_PATH}',
        )
    mlflow.set_experiment(experiment_name)


def train_model(model, model_type, experiment_name, X_train, X_test, y_train, y_test):
    """Treina o modelo, calcula as métricas e loga tudo no MLflow."""
    get_or_create_experiment(experiment_name)

    with mlflow.start_run(run_name=experiment_name):
        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        metrics = compute_metrics(y_test, y_pred_test, y_train, y_pred_train)

        print(f'=== {model_type} ===')
        for name, value in metrics.items():
            print(f'  {name}: {value:.4f}')

        mlflow.log_param('model_type', model_type)
        for name, value in metrics.items():
            mlflow.log_metric(name, value)

        mlflow.log_artifact(DATA_PATH, artifact_path='dataset')
        mlflow.sklearn.log_model(model, artifact_path='model')
        # - name='model' (novo jeito) → MLflow cria uma "LoggedModel entity" (conceito novo do 3.x), mas na versão do MLflow que estamos usando, a coluna Models da UI não mostra isso.
        # - artifact_path='model' (jeito antigo) → Funciona e aparece na coluna Models, mas dá warning de depreciação. Vamos manter usando artifact_path

        mlflow.set_tag('stage', 'baseline')
        mlflow.set_tag('dataset', 'telco_churn_processed')


# Leitura e split dos dados
df = load_data(DATA_PATH)

X_train, X_test, y_train, y_test = split_data(df)

# Configuração do MLflow
setup_mlflow()

# Treino dos modelos
train_model(
    model=LogisticRegression(random_state=42, solver='liblinear'),
    model_type='Logistic Regression',
    experiment_name=EXPERIMENT_LOGISTIC,
    X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
)

train_model(
    model=DummyClassifier(random_state=42, strategy='most_frequent'),
    model_type='Dummy Classifier',
    experiment_name=EXPERIMENT_DUMMY,
    X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
)

# ---
# Explicação sobre o uso do parãmetro solver='liblinear' em LogisticRegression
#
# solver='liblinear' é o algoritmo de otimização usado pra encontrar os coeficientes da Regressão Logística.
#
# - Padrão (lbfgs): algoritmo quase-Newton, bom pra datasets grandes, mas exige dados escalados e às vezes
# não converge no número padrão de iterações (max_iter=100). Foi o motivo do aviso de ConvergenceWarning no notebook, pois
# foi usado o padrão lbfgs.

# - liblinear: baseado em coordinate descent, rápido em datasets pequenos/médios (como esse com 7k linhas), converge
# sem precisar de scaling, e aceita regularização L1 e L2.
#
# Ou seja: liblinear evita o warning de convergência e tende a dar um fit mais estável nesse dataset.
#
# ---
# Analisando a tabela de resultado de testes usando ou não esses parãmetros para explicar o benefício deles.
# Cada linha da tabela foi um treino rodado com seus resultados.
#
# ┌───────┬──────────────┬───────────────┬─────────┬─────────────┐
# │ Linha │    Config    │ test_accuracy │ test_f1 │ overfitting │
# ├───────┼──────────────┼───────────────┼─────────┼─────────────┤
# │ 1     │ Nenhum       │ 0.8970        │ 0.8221  │ 0.0053      │
# ├───────┼──────────────┼───────────────┼─────────┼─────────────┤
# │ 2     │ Só liblinear │ 0.9120        │ 0.8473  │ 0.0112      │
# ├───────┼──────────────┼───────────────┼─────────┼─────────────┤
# │ 3     │ Só stratify  │ 0.9084        │ 0.8300  │ 0.0012      │
# ├───────┼──────────────┼───────────────┼─────────┼─────────────┤
# │ 4     │ Ambos        │ 0.8964        │ 0.8079  │ 0.0032      │
# └───────┴──────────────┴───────────────┴─────────┴─────────────┘
#
# Observações
#
# - A linha 2 parece "a melhor" (maior F1, maior accuracy), mas também tem o maior overfitting.
# Isso é engano estatístico: como o split não foi estratificado, o teste pode ter pegado uma amostra "fácil" por sorte.
# - A linha 3 (só stratify) tem o menor overfitting (0.0012) — a distribuição do teste é idêntica à do treino, então a
# métrica é a mais honesta.
# - A linha 4 (ambos) tem métricas ligeiramente menores, mas são as mais confiáveis.
#
# Recomendação: use os dois
#
# - stratify=y é obrigatório em dataset desbalanceado. Não é opcional — é o que torna suas métricas comparáveis entre execuções.
# - solver='liblinear' elimina o warning de convergência e garante que o fit termina corretamente.
#
# A diferença de F1 entre a linha 2 (0.8473) e a linha 4 (0.8079) não significa que a linha 2 é um modelo melhor — significa
# que ela foi avaliada numa fatia de teste com proporção de classes diferente, e por isso os números saíram
# mais "bonitos". Em produção, o modelo real é o mesmo; só o "termômetro" mudou.
