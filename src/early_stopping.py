import numpy as np
import torch
import os

class EarlyStopping:
    """
    Interrompe o treinamento antecipadamente se a loss de validação não melhorar após um determinado número de épocas.
    Também cuida de salvar o melhor modelo.
    """
    def __init__(self, patience: int = 7, min_delta: float = 0, save_path: str = "best_model.pth"):
        """
        Args:
            patience (int): Quantas épocas esperar desde a última melhora para parar o treino.
            min_delta (float): Mudança mínima na métrica monitorada para ser considerada uma melhora (evita flutuações pequenas).
            save_path (str): Caminho para salvar os pesos do melhor modelo.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False

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
            torch.save(model.state_dict(), self.save_path)
            
    def load_best_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Carrega os pesos do melhor modelo no arquivo configurado."""
        if os.path.exists(self.save_path):
            model.load_state_dict(torch.load(self.save_path))
        return model
