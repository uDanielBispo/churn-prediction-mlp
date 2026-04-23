# utils.py — Funções utilitárias reutilizáveis por outros módulos do projeto.
#
# Padrão Extract Module: funções que não pertencem a nenhum módulo específico
# (não são de treino, nem de pipeline, nem de API) vivem aqui.

import random

import numpy as np
import torch
from sklearn.metrics import f1_score


def set_seed(seed: int = 42) -> None:
    """Fixa as seeds de todos os geradores aleatórios para garantir reprodutibilidade.

    Sem isso, cada execução do treino produziria pesos iniciais diferentes,
    tornando impossível comparar resultados entre runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_best_threshold(all_probs, all_targets):
    """Busca o limiar de classificação que maximiza o F1-Score no conjunto de teste.

    Em vez de usar 0.5 fixo, testa 81 valores entre 0.1 e 0.9 e escolhe o que
    gera o melhor F1. Isso é importante em datasets desbalanceados: um threshold
    menor (ex: 0.36) significa que o modelo prefere identificar mais casos de churn
    mesmo arriscando mais falsos positivos — o que muitas vezes faz sentido no negócio.

    Retorna:
        best_threshold: float com o melhor valor encontrado.
        all_preds:      array de previsões binárias (0 ou 1) usando esse threshold.
    """
    best_threshold = 0.5
    best_f1 = 0.0

    for th in np.linspace(0.1, 0.9, 81):
        preds = (all_probs > th).astype(int)
        current_f1 = f1_score(all_targets, preds, zero_division=0)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = th

    print(f"Limiar de Classificação (Threshold) Selecionado: {best_threshold:.3f}")
    return best_threshold, (all_probs > best_threshold).astype(int)
