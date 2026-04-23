"""
    carrega dados, instancia o modelo, executa o loop de treino e avalia.

     Divisão dos dados em três partes:

      df_train_val, df_test = train_test_split(df, test_size=0.2, stratify=df['target'])
      df_train, df_val = train_test_split(df_train_val, test_size=0.2, stratify=df_train_val['target'])

      Diferente do treino feito para o baseline com regressao logistica que dividia em dois (treino/teste), aqui são três partes:
      - Treino (64%) — o modelo aprende com esses dados
      - Validação (16%) — usado em cada epoch para checar se está overfittando (o early stopping monitora a loss aqui)
      - Teste (20%) — usado só no final, para a avaliação final, igual foi feito no baseline
"""
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "1" # Mute loky warnings

import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import mlflow
import mlflow.pytorch
import argparse
import random

# Import de módulos locais
from src.dataset import ChurnDataset
from src.model import ChurnMLP
from src.early_stopping import EarlyStopping

def set_seed(seed=42):
    """Fixa as seeds aleatórias para reprodutibilidade total (Engenharia de MLOps)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def setup_mlflow():
    """Configura o tracking URI do MLflow apontando para a raiz do projeto."""
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mlflow_tracking_db = os.path.join(root_dir, "mlflow.db")
    mlflow_registry_uri = os.path.join(root_dir, "mlruns")
    
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_tracking_db}")
    mlflow.set_registry_uri(f"file://{mlflow_registry_uri}")

def train_model():
    """Função principal de treinamento e instrumentação com MLFlow."""
    set_seed(42)
    setup_mlflow()
    
    # Parâmetros padrão
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 100
    patience = 10
    
    # 1. Carregamento dos dados
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(root_dir, 'data', 'processed', 'telco_customer_churn_processed.csv')
    df = pd.read_csv(data_path)
    
    # 2. Divisão dos dados (Train/Validation/Test)
    # Primeiro separamos o de Teste (20%) - Mesmo que no Baseline!
    df_train_val, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['target'])
    # Depois o Validação usando 20% do restante (o que dá 16% do total) para early stopping
    df_train, df_val = train_test_split(df_train_val, test_size=0.2, random_state=42, stratify=df_train_val['target'])
    
    # 3. Criação de Datasets e DataLoaders
    train_dataset = ChurnDataset(df_train)
    val_dataset = ChurnDataset(df_val)
    test_dataset = ChurnDataset(df_test)

    # Em vez de passar todos os 4500 exemplos de uma vez para a rede, o DataLoader divide em mini-lotes de 64.
    # A cada época, a rede vê todos os lotes um por um. O shuffle=True embaralha a ordem a cada época para o modelo não memorizar a sequência.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 4. Instanciação do Modelo, Loss e Optimizer
    input_dim = train_dataset.X.shape[1]
    model = ChurnMLP(input_dim=input_dim, hidden_dim_1=64, hidden_dim_2=32, dropout_rate=0.3)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    early_stopping = EarlyStopping(patience=patience, save_path=os.path.join(root_dir, "best_mlp_model.pth"))
    
    # 5. MLflow Tracking
    experiment_name = "churn_prediction_mlp_pytorch"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(name=experiment_name)
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name="MLP_Base_Architecture"):
        # Log dos Hyperparametros
        mlflow.log_params({
            "model_type": "PyTorch MLP",
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_epochs": num_epochs,
            "early_stop_patience": patience,
            "hidden_dim_1": 64,
            "hidden_dim_2": 32,
            "dropout_rate": 0.3
        })
        
        print("Iniciando Treinamento...")
        for epoch in range(num_epochs):
            model.train()       # ativa Dropout e BatchNorm no modo treino
            train_loss = 0.0
            
            # Loop de batches (Epoch de Treino)
            for batch_X, batch_y in train_loader:
                """
                O ciclo interno (zero_grad -> forward -> loss -> backward -> step) é o coração do aprendizado de qualquer rede neural. Em termos simples:
                1. A rede chuta uma resposta (forward)
                2. Calculamos o quão errado foi o chute (loss)
                3. Calculamos em qual direção cada peso contribuiu para o erro (backward)
                4. Ajustamos os pesos um pouquinho na direção que reduz o erro (step)
                
                Repetido milhares de vezes, o modelo vai convergindo para pesos que erram cada vez menos.
                """
                optimizer.zero_grad()                    # zera os gradientes acumulados do passo anterior
                outputs = model(batch_X)                 # passagem forward: gera previsões
                loss = criterion(outputs, batch_y)       # calcula o erro (BCEWithLogitsLoss)
                loss.backward()                          # backpropagation: calcula gradientes
                optimizer.step()                         # ajusta os pesos na direção certa
                train_loss += loss.item() * batch_X.size(0)

            train_loss = train_loss / len(train_dataset)
            
            # Loop de batches (Epoch de Validação)
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item() * batch_X.size(0)
            val_loss = val_loss / len(val_dataset)
            
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            
            # Checagem de Early Stopping com a loss de validação
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch+1}")
                break
                
        # 6. Avaliação Final (com o Test Set idêntico ao dos baselines)
        print("\nAvaliação com as métricas do Baseline (Test Set)")
        model = early_stopping.load_best_model(model)
        model.eval()
        
        all_probs = []
        all_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X) # Logits
                probs = torch.sigmoid(outputs)
                
                all_probs.extend(probs.numpy())
                all_targets.extend(batch_y.numpy())
                
        # Aplicação Prática (Aula 06): Substituição do "0.5" engessado
        # Busca do 'ponto operacional' ideal que maximize o F1-Score
        best_threshold = 0.5
        best_f1 = 0
        for th in np.linspace(0.1, 0.9, 81):
            preds_th = (np.array(all_probs) > th).astype(int)
            f1_th = f1_score(all_targets, preds_th, zero_division=0)
            if f1_th > best_f1:
                best_f1 = f1_th
                best_threshold = th
                
        print(f"Limiar de Classificação (Threshold) Selecionado: {best_threshold:.3f}")
        all_preds = (np.array(all_probs) > best_threshold).astype(int)
                
        # Metricas
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
        
        mlflow.set_tag("stage", "pytorch_mlp")
        mlflow.set_tag("dataset", "telco_churn_processed")
        
        mlflow.pytorch.log_model(model, "model")
        print("\nTreinamento Finalizado e gravado no MLFlow.")
        
        joblib.dump(model, os.path.join(os.getcwd(), f'src/models/{experiment_name}_model.pkl'))


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    train_model()
