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

        #  nn.Sequential é uma lista ordenada de camadas — os dados entram no topo e passam por cada camada em sequência, como um pipeline. #  Cada bloco segue o mesmo padrão: Linear → BatchNorm → ReLU → Dropout
        self.network = nn.Sequential(
            # obs. nn.Module é a classe base de qualquer rede neural no PyTorch — equivalente a extends no Java. Toda rede precisa herdar dela.

            # Camada 1
            nn.Linear(input_dim, hidden_dim_1), # - nn.Linear(entrada, saída) — a camada principal. Multiplica cada feature por um peso e soma um bias.
            nn.BatchNorm1d(hidden_dim_1),       # - nn.BatchNorm1d — normaliza os valores que saem da camada Linear para que fiquem numa escala parecida. Sem isso, os valores podem explodir ou zerar durante o treino, dificultando o aprendizado. Pense nisso como o StandardScaler do sklearn, mas aplicado dentro da própria rede a cada batch.
            nn.ReLU(),                          # - nn.ReLU() — função de ativação. Simplesmente faz: se o valor é negativo, vira zero; se é positivo, permanece igual. Parece simples, mas é isso que dá à rede a capacidade de aprender padrões não-lineares. Sem ela, empilhar camadas lineares não adiantaria nada — seria equivalente a uma só camada.
            nn.Dropout(dropout_rate),           # - nn.Dropout(0.3) — durante o treino, desliga aleatoriamente 30% dos neurônios a cada passagem. Isso força a rede a não depender de nenhum neurônio específico, reduzindo overfitting. É desligado automaticamente na avaliação (quando você chamamodel.eval()).

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
