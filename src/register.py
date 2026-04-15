# register.py - RESUMO
# Responsável por registrar os artefatos do modelo treinado no MLflow,
# promovendo para produção apenas se as métricas (acurácia, F1-Score, etc.)
# forem melhores que as do modelo já em produção.

# DETALHE
# O register.py faz três coisas:
#
# 1. get_best_run(experiment_name, metric)
#
# Usa o MlflowClient pra listar runs do experimento, ordenados decrescentemente pela métrica (no caso, test_f1_score). Retorna só o primeiro (melhor).
#
# 2. get_production_metric(model_name, metric)
#
# Olha no Model Registry se já existe alguma versão do modelo na stage Production. Se sim, retorna o F1 dela; se não, retorna None.
#
# 3. register_if_better(experiment_name, model_name)
#
# Compara o novo F1 com o da produção:
# - Se não tem nada em produção ou o novo é melhor → registra a nova versão no Registry e promove pra Production (arquivando a versão anterior com archive_existing_versions=True).
# - Se o que está em produção é melhor → não faz nada.
#
# No final do arquivo, chama a função pra cada modelo (Logistic e Dummy).
#
# Como rodar
#
# Primeiro garanta que você já rodou o train.py pelo menos uma vez (pra ter runs no MLflow). Depois:
#
# python src/register.py
#
# Na primeira execução: vai registrar ambos os modelos em produção (não tem nada lá antes).
#
# Nas execuções seguintes: só vai promover se o novo F1 for estritamente maior que o atual em produção.


import os

import mlflow
from mlflow.tracking import MlflowClient

# Constantes de caminhos (relativos à raiz do projeto)
MLFLOW_DB_PATH = os.path.join(os.getcwd(), 'mlflow.db')
MLRUNS_PATH = os.path.join(os.getcwd(), 'mlruns')
# O que os.path.join faz
#
# Ele junta pedaços de caminho usando o separador correto do sistema operacional:
# - No Linux/macOS: usa /
# - No Windows: usa \
#  \
# Então os.path.join(os.getcwd(), 'mlflow.db') vira:
# - Mac/Linux: /Users/rafael/.../churn-prediction-mlp/mlflow.db
# - Windows: C:\Users\daniel\...\churn-prediction-mlp\mlflow.db

# Constantes de experimentos e nomes dos modelos no Model Registry
EXPERIMENT_LOGISTIC = 'chrun_prediction_logistic_regression'
EXPERIMENT_DUMMY = 'chrun_prediction_dummy_classifier'

MODEL_NAME_LOGISTIC = 'churn-logistic-regression'
MODEL_NAME_DUMMY = 'churn-dummy-classifier'

# Métrica usada para comparar o novo modelo com o que está em produção
COMPARISON_METRIC = 'test_f1_score'


def setup_mlflow():
    """Configura o tracking URI e o registry URI do MLflow."""
    mlflow.set_tracking_uri(f'sqlite:///{MLFLOW_DB_PATH}')
    mlflow.set_registry_uri(f'file://{MLRUNS_PATH}')


def get_best_run(experiment_name, metric=COMPARISON_METRIC):
    """Busca o melhor run de um experimento, ordenado pela métrica especificada."""
    client = MlflowClient()
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"Experimento '{experiment_name}' nao encontrado.")
        return None

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f'metrics.{metric} DESC'],
        max_results=1,
    )

    if not runs:
        print(f"Nenhum run encontrado em '{experiment_name}'.")
        return None

    return runs[0]


def get_production_metric(model_name, metric=COMPARISON_METRIC):
    """Retorna o valor da métrica do modelo atualmente em Production, ou None se não houver."""
    client = MlflowClient()
    try:
        production_versions = client.get_latest_versions(model_name, stages=['Production'])
        if not production_versions:
            return None
        run = client.get_run(production_versions[0].run_id)
        return run.data.metrics.get(metric)
    except Exception:
        return None


def register_if_better(experiment_name, model_name, metric=COMPARISON_METRIC):
    """Busca o melhor run do experimento e promove para Production se for melhor que o atual."""
    print(f'\n--- {experiment_name} ---')

    best_run = get_best_run(experiment_name, metric)
    if best_run is None:
        return

    new_value = best_run.data.metrics.get(metric, 0.0)
    prod_value = get_production_metric(model_name, metric)

    print(f'  Novo modelo  -> {metric}: {new_value:.4f}')
    if prod_value is None:
        print(f'  Em producao  -> nenhum modelo registrado ainda')
    else:
        print(f'  Em producao  -> {metric}: {prod_value:.4f}')

    if prod_value is None or new_value > prod_value:
        model_uri = f'runs:/{best_run.info.run_id}/model'
        registered = mlflow.register_model(model_uri=model_uri, name=model_name)

        client = MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=registered.version,
            stage='Production',
            archive_existing_versions=True,
        )
        print(f"  Modelo '{model_name}' v{registered.version} promovido para Production.")
    else:
        print(f'  Modelo em producao continua sendo o melhor. Nada alterado.')


# Configuração do MLflow
setup_mlflow()

# Registra (ou nao) cada modelo no Model Registry
register_if_better(EXPERIMENT_LOGISTIC, MODEL_NAME_LOGISTIC)
register_if_better(EXPERIMENT_DUMMY, MODEL_NAME_DUMMY)
