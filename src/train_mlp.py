# train_mlp.py — Orquestra o pipeline completo de treinamento da rede neural MLP.
#
# Divisão dos dados em três partes (diferente do baseline que usava duas):
#   - Treino   (64%) — o modelo aprende com esses dados
#   - Validação(16%) — monitorado pelo early stopping a cada época
#   - Teste    (20%) — avaliação final, igual ao conjunto usado no baseline
import os

os.environ["LOKY_MAX_CPU_COUNT"] = "1"  # silencia avisos do joblib/loky

import joblib
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.dataset import ChurnDataset
from src.early_stopping import EarlyStopping
from src.model import ChurnMLP
from src.pipeline import apply_preprocessing, build_preprocessing_pipeline
from src.utils import find_best_threshold, set_seed

# Raiz do projeto — calculada uma única vez e reutilizada por todas as funções.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def setup_mlflow() -> None:
    """Configura o MLflow para gravar experimentos no banco SQLite local.

    O tracking URI aponta para mlflow.db na raiz do projeto.
    O registry URI aponta para a pasta mlruns/, onde os artefatos são salvos.
    """
    mlflow.set_tracking_uri(f"sqlite:///{os.path.join(ROOT_DIR, 'mlflow.db')}")
    mlflow.set_registry_uri(f"file://{os.path.join(ROOT_DIR, 'mlruns')}")


def load_and_split_data(data_path: str):
    """Carrega o CSV processado e divide em três conjuntos: treino, validação e teste.

    A divisão é feita em dois passos para preservar a proporção de classes (stratify)
    em cada um dos três conjuntos, evitando viés na avaliação.

    Retorna:
        df_train: 64% dos dados — usado para atualizar os pesos da rede.
        df_val:   16% dos dados — usado pelo early stopping para detectar overfitting.
        df_test:  20% dos dados — reservado para a avaliação final.
    """
    df = pd.read_csv(data_path)

    # Passo 1: separa os 20% de teste do restante
    df_train_val, df_test = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['target']
    )
    # Passo 2: dos 80% restantes, separa mais 20% para validação (= 16% do total)
    df_train, df_val = train_test_split(
        df_train_val, test_size=0.2, random_state=42, stratify=df_train_val['target']
    )
    return df_train, df_val, df_test


def create_dataloaders(df_train, df_val, df_test, batch_size: int):
    """Converte os DataFrames em DataLoaders do PyTorch.

    O DataLoader divide os dados em mini-lotes (batches) e os entrega à rede
    um lote por vez durante o treino, o que é mais eficiente do que passar
    todos os dados de uma vez. O shuffle=True embaralha a ordem dos lotes a
    cada época para o modelo não memorizar a sequência dos exemplos.

    Retorna:
        train_loader, val_loader, test_loader: iteradores de batches para cada conjunto.
    """
    train_dataset = ChurnDataset(df_train)
    val_dataset = ChurnDataset(df_val)
    test_dataset = ChurnDataset(df_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def build_model(input_dim: int, learning_rate: float):
    """Instancia o modelo MLP, a função de perda e o otimizador.

    - ChurnMLP: a rede neural definida em src/model.py.
    - BCEWithLogitsLoss: função de perda para classificação binária. Aplica a
      Sigmoid internamente, o que é mais estável numericamente do que aplicar
      Sigmoid na saída da rede e depois calcular a perda separadamente.
    - Adam: otimizador que adapta a taxa de aprendizado para cada peso
      individualmente, convergindo mais rápido que o gradiente descendente simples.

    Retorna:
        model, criterion, optimizer
    """
    model = ChurnMLP(input_dim=input_dim, hidden_dim_1=64, hidden_dim_2=32, dropout_rate=0.3)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return model, criterion, optimizer


def run_training_loop(model, train_loader, val_loader, criterion, optimizer,
                      early_stopping, num_epochs: int):
    """Executa o loop de treinamento época por época com monitoramento de early stopping.

    A cada época:
      1. Passa todos os batches de treino pela rede (forward + backward + step).
      2. Avalia a loss no conjunto de validação sem atualizar os pesos.
      3. Verifica se a loss de validação melhorou; se não melhorar por 'patience'
         épocas seguidas, interrompe o treino e restaura o melhor modelo salvo.

    Retorna:
        model com os pesos do melhor checkpoint (menor val_loss).
    """
    print("Iniciando Treinamento...")

    for epoch in range(num_epochs):
        # --- Fase de treino ---
        # model.train() ativa o Dropout e o BatchNorm no modo correto para treino.
        model.train()
        train_loss = 0.0

        for batch_X, batch_y in train_loader:
            # zero_grad: limpa os gradientes do passo anterior — sem isso eles se acumulam.
            optimizer.zero_grad()
            # forward: a rede gera previsões (logits) para o batch.
            outputs = model(batch_X)
            # loss: mede o quão erradas foram as previsões.
            loss = criterion(outputs, batch_y)
            # backward: calcula a contribuição de cada peso para o erro (gradientes).
            loss.backward()
            # step: ajusta os pesos na direção que reduz o erro.
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)

        train_loss /= len(train_loader.dataset)

        # --- Fase de validação ---
        # model.eval() desativa o Dropout e coloca BatchNorm em modo de inferência.
        model.eval()
        val_loss = 0.0

        with torch.no_grad():  # desativa o cálculo de gradientes para economizar memória
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_X.size(0)

        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)

        # Verifica se deve parar e salva o melhor modelo se houve melhora.
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"Early stopping na época {epoch + 1}")
            break

    # Restaura os pesos do melhor momento do treino antes de retornar.
    return early_stopping.load_best_model(model)


def evaluate_model(model, test_loader):
    """Coleta probabilidades preditas e rótulos reais no conjunto de teste.

    Usa torch.sigmoid para converter os logits da rede em probabilidades [0, 1].
    Não aplica threshold aqui — isso é responsabilidade de find_best_threshold().

    Retorna:
        all_probs:   array com a probabilidade de churn para cada cliente.
        all_targets: array com o rótulo real (0 ou 1) de cada cliente.
    """
    model.eval()
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            probs = torch.sigmoid(model(batch_X))
            all_probs.extend(probs.numpy())
            all_targets.extend(batch_y.numpy())

    return np.array(all_probs), np.array(all_targets)


def log_metrics(all_targets, all_preds, all_probs) -> None:
    """Calcula, imprime e registra as métricas finais de avaliação no MLflow.

    Métricas calculadas:
      - Accuracy:  proporção de previsões corretas.
      - F1-Score:  média harmônica entre precision e recall (principal métrica aqui).
      - Precision: dos que o modelo disse que vão dar churn, quantos realmente deram.
      - Recall:    dos que realmente deram churn, quantos o modelo identificou.
      - AUC-ROC:   capacidade geral do modelo de separar as classes (não depende do threshold).
    """
    test_accuracy = accuracy_score(all_targets, all_preds)
    test_f1 = f1_score(all_targets, all_preds)
    test_precision = precision_score(all_targets, all_preds, zero_division=0)
    test_recall = recall_score(all_targets, all_preds)
    test_auc = roc_auc_score(all_targets, all_probs)

    print(f"Test Accuracy:  {test_accuracy:.4f}")
    print(f"Test F1 Score:  {test_f1:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall:    {test_recall:.4f}")
    print(f"Test AUC-ROC:   {test_auc:.4f}")

    mlflow.log_metric("test_accuracy", test_accuracy)
    mlflow.log_metric("test_f1_score", test_f1)
    mlflow.log_metric("test_precision", test_precision)
    mlflow.log_metric("test_recall", test_recall)
    mlflow.log_metric("test_roc_auc", test_auc)


def train_model() -> None:
    """Orquestra o pipeline completo de treinamento da MLP.

    Esta função não contém lógica própria — ela chama as funções especializadas
    na ordem correta e passa os resultados de uma para a outra. Esse é o padrão
    Extract Function aplicado: cada etapa tem nome, responsabilidade e pode ser
    testada de forma independente.
    """
    set_seed(42)
    setup_mlflow()

    # Hiperparâmetros centralizados aqui para fácil ajuste
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 100
    patience = 10

    data_path = os.path.join(ROOT_DIR, 'data', 'processed', 'telco_customer_churn_processed.csv')

    df_train, df_val, df_test = load_and_split_data(data_path)

    # Aplica o pipeline de pré-processamento: normaliza numéricas, mantém binárias.
    # O preprocessor é salvo em disco para ser reutilizado pela API na inferência.
    preprocessor = build_preprocessing_pipeline()
    df_train, df_val, df_test = apply_preprocessing(preprocessor, df_train, df_val, df_test)

    preprocessor_path = os.path.join(ROOT_DIR, 'src', 'models', 'preprocessor.pkl')
    joblib.dump(preprocessor, preprocessor_path)

    train_loader, val_loader, test_loader = create_dataloaders(df_train, df_val, df_test, batch_size)

    # input_dim é o número de features — lido direto do dataset para não hardcodar
    input_dim = train_loader.dataset.X.shape[1]
    model, criterion, optimizer = build_model(input_dim, learning_rate)

    early_stopping = EarlyStopping(
        patience=patience,
        save_path=os.path.join(ROOT_DIR, "best_mlp_model.pth")
    )

    # Configura o experimento no MLflow
    experiment_name = "churn_prediction_mlp_pytorch"
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(name=experiment_name)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="MLP_Base_Architecture"):
        mlflow.log_params({
            "model_type": "PyTorch MLP",
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_epochs": num_epochs,
            "early_stop_patience": patience,
            "hidden_dim_1": 64,
            "hidden_dim_2": 32,
            "dropout_rate": 0.3,
            "preprocessing": "StandardScaler em colunas numéricas",
        })

        model = run_training_loop(
            model, train_loader, val_loader, criterion, optimizer, early_stopping, num_epochs
        )

        print("\nAvaliação com as métricas do Baseline (Test Set)")
        all_probs, all_targets = evaluate_model(model, test_loader)
        _, all_preds = find_best_threshold(all_probs, all_targets)
        log_metrics(all_targets, all_preds, all_probs)

        mlflow.set_tag("stage", "pytorch_mlp")
        mlflow.set_tag("dataset", "telco_churn_processed")
        mlflow.pytorch.log_model(model, "model")

        model_path = os.path.join(ROOT_DIR, 'src', 'models', f'{experiment_name}_model.pkl')
        joblib.dump(model, model_path)

    print("\nTreinamento finalizado e gravado no MLflow.")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    train_model()
