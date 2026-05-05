# register_models.py — Compara os modelos recém-treinados com os que estão em
# Production no MLflow Registry e promove se as métricas forem melhores.
#
# Fluxo:
#   1. Busca o melhor run de cada experimento (maior test_f1_score).
#   2. Compara com a versão atualmente em Production no Registry.
#   3. Se o novo for melhor (ou se não houver nenhum em Production), promove.
#   4. O preprocessor sempre acompanha o MLP — eles são versionados juntos.
#
# Execute com: python -m src.register_models

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from src.utils import setup_mlflow

# Nomes dos modelos no MLflow Registry — usados pela API em modo production.
REGISTRY_MLP = "ChurnMLP"
REGISTRY_LOGISTIC = "ChurnLogistic"
REGISTRY_DUMMY = "ChurnDummy"
REGISTRY_PREPROCESSOR = "ChurnPreprocessor"

METRIC = "test_f1_score"


def get_best_run(experiment_name: str):
    """Busca o run com maior test_f1_score dentro do experimento."""
    client = MlflowClient()
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"  Experimento '{experiment_name}' não encontrado.")
        return None

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{METRIC} DESC"],
        max_results=1,
    )
    return runs[0] if runs else None


def get_production_metric(registry_name: str) -> float | None:
    """Retorna o test_f1_score do modelo atualmente em Production, ou None."""
    client = MlflowClient()
    try:
        versions = client.get_latest_versions(registry_name, stages=["Production"])
        if not versions:
            return None
        run = client.get_run(versions[0].run_id)
        return run.data.metrics.get(METRIC)
    except Exception:
        return None


def promote(run_id: str, artifact_path: str, registry_name: str) -> str:
    """Registra o artefato no Registry e promove para Production."""
    client = MlflowClient()

    # Cria uma nova versão do modelo no Registry apontando para o run.
    model_uri = f"runs:/{run_id}/{artifact_path}"
    registered = mlflow.register_model(model_uri=model_uri, name=registry_name)

    # Promove para Production e arquiva a versão anterior automaticamente.
    client.transition_model_version_stage(
        name=registry_name,
        version=registered.version,
        stage="Production",
        archive_existing_versions=True,
    )
    return registered.version


def register_if_better(experiment_name: str, registry_name: str, artifact_path: str) -> str | None:
    """Compara novo modelo com Production e promove se for melhor.

    Retorna o run_id do melhor run se houve promoção, None caso contrário.
    Isso permite que o chamador saiba qual run foi promovido (ex: para registrar
    o preprocessor a partir do mesmo run da MLP).
    """
    print(f"\n--- {registry_name} ---")

    best_run = get_best_run(experiment_name)
    if best_run is None:
        return None

    new_value = best_run.data.metrics.get(METRIC, 0.0)
    prod_value = get_production_metric(registry_name)

    print(f"  Novo modelo  → {METRIC}: {new_value:.4f}")
    if prod_value is None:
        print("  Em produção  → nenhum modelo registrado ainda")
    else:
        print(f"  Em produção  → {METRIC}: {prod_value:.4f}")

    if prod_value is None or new_value > prod_value:
        version = promote(best_run.info.run_id, artifact_path, registry_name)
        print(f"  Promovido para Production (v{version})")
        return best_run.info.run_id
    else:
        print("  Modelo em produção continua sendo o melhor. Nada alterado.")
        return None


def register_all() -> None:
    setup_mlflow()

    register_if_better("churn_prediction_logistic_regression", REGISTRY_LOGISTIC, "model")
    register_if_better("churn_prediction_dummy_classifier", REGISTRY_DUMMY, "model")

    # MLP: se for promovido, o preprocessor do mesmo run também é promovido.
    mlp_run_id = register_if_better("churn_prediction_mlp_pytorch", REGISTRY_MLP, "model")

    if mlp_run_id:
        print(f"\n--- {REGISTRY_PREPROCESSOR} ---")
        version = promote(mlp_run_id, "preprocessor", REGISTRY_PREPROCESSOR)
        print(f"  Promovido para Production junto com a MLP (v{version})")

    print("\nRegistro de modelos concluído.")


if __name__ == "__main__":
    register_all()
