import torch
import torch.nn as nn

class ChurnMLP(nn.Module):
    """
    Multilayer Perceptron (MLP) para classificação binária de Churn.
    """
    def __init__(self, input_dim: int, hidden_dim_1: int = 64, hidden_dim_2: int = 32, dropout_rate: float = 0.3):
        """
        Inicializa as camadas do modelo.
        
        Args:
            input_dim (int): Número de features no dataset de entrada.
            hidden_dim_1 (int): Número de neurônios na 1ª camada oculta.
            hidden_dim_2 (int): Número de neurônios na 2ª camada oculta.
            dropout_rate (float): Taxa de dropout para regularização.
        """
        super(ChurnMLP, self).__init__()
        
        self.network = nn.Sequential(
            # Camada 1
            nn.Linear(input_dim, hidden_dim_1),
            nn.BatchNorm1d(hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Camada 2
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.BatchNorm1d(hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Camada de Saída (Logits)
            # A função de ativação Sigmóide não é inserida aqui, pois será aplicada
            # em conjunto com a loss BCEWithLogitsLoss no loop de treinamento, 
            # o que traz mais estabilidade numérica.
            nn.Linear(hidden_dim_2, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passagem forward da rede neural."""
        return self.network(x).squeeze(1) # Squeeze remove a dimensão extra do output [batch_size, 1] -> [batch_size]
