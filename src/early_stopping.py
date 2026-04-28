import os

import numpy as np
import torch


class EarlyStopping:
    """
    Interrompe o treinamento antecipadamente se a loss de validação não melhorar após um determinado número de épocas.
    Também cuida de salvar o melhor modelo.
    Este arquivo resolve um problema clássico em redes neurais chamado overfitting, onde o modelo continua treinando,
    decora os dados de treino, e começa a piorar nos dados de validação. O early stopping para o treino no momento certo.
    """
    def __init__(self, patience: int = 7, min_delta: float = 0, save_path: str = "best_model.pth"):
        """
        Args:
            patience (int): Quantas épocas esperar desde a última melhora para parar o treino.
            min_delta (float): Mudança mínima na métrica monitorada para ser considerada uma melhora (evita flutuações pequenas).
            save_path (str): Caminho para salvar os pesos do melhor modelo.
        """
        # patience=7 significa: "se a loss de validação não melhorar por 7 épocas seguidas, pare". No train_mlp.py foi configurado com patience=10.

        self.patience = patience      # quantas epochs antes de parar
        self.min_delta = min_delta    # melhora mínima para contar como progresso
        self.save_path = save_path

        self.best_loss = np.inf       # começa com "infinito" para qualquer valor ser melhor
        self.counter = 0              # contador de epochs sem melhora
        self.early_stop = False       # flag que sinaliza para o loop de treino parar

    '''
     O __call__ faz com que o objeto seja chamável como uma função — early_stopping(val_loss, model).
     A cada epoch, verifica se a loss melhorou. Se sim, salva o modelo e zera o contador. Se não, incrementa o contador. Quando
     o contador atinge o limite de paciência, levanta a flag early_stop.
    '''
    def __call__(self, val_loss: float, model: torch.nn.Module):
        if val_loss < (self.best_loss - self.min_delta):
            # Tivemos uma melhora
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model: torch.nn.Module):
        """Salva o modelo quando há uma diminuição na loss de validação."""
        # Se um caminho for provido, salvar
        if self.save_path:
            # state_dict() retorna um dicionário com todos os pesos da rede neural.
            # Só salva quando a loss melhora, então no final você tem os pesos do melhor momento do treino, não do último.
            torch.save(model.state_dict(), self.save_path)

    def load_best_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Carrega os pesos do melhor modelo no arquivo configurado.
        Depois que o treino para, este método carrega de volta os pesos do melhor checkpoint.
        Sem isso, ficaria com os pesos da última época — que podem ser piores.
        """
        if os.path.exists(self.save_path):
            model.load_state_dict(torch.load(self.save_path))
        return model
