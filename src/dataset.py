import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class ChurnDataset(Dataset):
    """
    Custom PyTorch Dataset para o modelo de previsão de Churn.
    """
    def __init__(self, data: pd.DataFrame, target_col: str = 'target'):
        """
        Inicializa o dataset.
        
        Args:
            data (pd.DataFrame): DataFrame Pandas processado.
            target_col (str): Nome da coluna alvo (variável resposta).
        """
        self.target_col = target_col
        
        # Extrai features e alvo
        self.X = data.drop(columns=[self.target_col]).values.astype(np.float32)
        
        # Manter como float32 é necessário para a BCEWithLogitsLoss
        if self.target_col in data.columns:
            self.y = data[self.target_col].values.astype(np.float32)
        else:
            self.y = np.zeros(len(data), dtype=np.float32)

    def __len__(self):
        """Retorna o número de amostras no dataset."""
        return len(self.X)

    def __getitem__(self, idx):
        """Retorna um dicionário com os tensores de features e alvo para um dado índice."""
        features = torch.tensor(self.X[idx], dtype=torch.float32)
        target = torch.tensor(self.y[idx], dtype=torch.float32)
        
        return features, target
