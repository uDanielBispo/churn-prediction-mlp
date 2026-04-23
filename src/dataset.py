import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class ChurnDataset(Dataset):
    """
    Custom PyTorch Dataset para o modelo de previsão de Churn.
    Aqui o construtor separa o DataFrame em features (X) e target (y), só separando colunas. O '.values' converte o DataFrame pandas para um array NumPy puro. O .astype(np.float32)
    converte para número de ponto flutuante de 32 bits porque a GPU e o PyTorch trabalham com esse tipo — usar float64 (o padrão do pandas) seria desnecessariamente pesado.
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
        """
        Retorna um dicionário com os tensores de features e alvo para um dado índice.
        Dado um índice, retorna aquela linha como tensores PyTorch. Um tensor é basicamente um array multidimensional otimizado para operações matemáticas em GPU/CPU.
        Por que essa classe existe? Porque o PyTorch usa um DataLoader que fica chamando __getitem__ para montar mini-lotes (batches) de forma automática, embaralhando os dados a cada epoch.
        Você entrega o ChurnDataset para o DataLoader e ele cuida do resto.
        """
        features = torch.tensor(self.X[idx], dtype=torch.float32)
        target = torch.tensor(self.y[idx], dtype=torch.float32)
        
        return features, target
