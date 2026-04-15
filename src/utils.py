# utils.py - Funções utilitárias compartilhadas entre os módulos do projeto.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def load_data(path):
    """Carrega o dataset processado a partir do caminho especificado."""
    return pd.read_csv(path)


def split_data(df, target_col='target', test_size=0.2, random_state=42):
    """Separa features e target, e divide em conjuntos de treino e teste."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # DIVISÃO EM TREINO E TESTE
    # Esta função divide as linhas: 80% para o modelo estudar e 20% para a prova final.
    # X_train: Perguntas de estudo | y_train: Respostas de estudo
    # X_test: Perguntas da prova  | y_test: Gabarito da prova (guardado conosco)

    # Resumo da diferença:
    # Variável	O que contém?	                            Para que serve?
    # X_train	Características (Idade, Colesterol, etc.)	É o material de estudo do modelo.
    # y_train	Resposta (Teve doença? Sim/Não)	            É o gabarito que o modelo usa para aprender durante o estudo.
    # X_test	Características (Novos pacientes)	        É a prova. O modelo deve prever baseado apenas nisso.
    # y_test	Resposta (Gabarito real)    	            Serve para você (desenvolvedor) conferir se o modelo passou na prova.
    #
    # Essa função é como uma "guilhotina" que corta seus dados em quatro pedaços. Imagine que você tem 100 linhas de dados:
    # X_train (80 linhas): As perguntas que o modelo vai usar para estudar. (Ex: idade, peso, pressão).
    # y_train (80 linhas): As respostas correspondentes às perguntas do X_train. O modelo olha para o X_train, chuta um resultado e confere com o y_train para ver se acertou e ajustar seus pesos.
    # X_test (20 linhas): Perguntas que o modelo nunca viu. Você as guarda para fazer a "prova final".
    # y_test (20 linhas): O gabarito da "prova final". Você usa isso para calcular a acurácia, comparando o que o modelo chutou para o X_test com a realidade que está no y_test.
    #
    # Explicando o uso do parãmetro stratify=y na função:
    #
    # É um parâmetro do train_test_split. Ele garante que a proporção das classes no target seja preservada nos conjuntos de treino e teste.
    #
    # No seu dataset: ~73% de não-churn (0) e ~27% de churn (1).
    # - Sem stratify: o split é aleatório. Por sorte/azar, o teste pode ficar com 30% de churn ou 22%, distorcendo as métricas.
    # - Com stratify=y: treino e teste ficam ambos com ~73/27.
    #
    # Isso é especialmente importante em datasets desbalanceados (como esse). Sem estratificação, cada execução com random_state diferente geraria F1/recall muito diferentes só por acaso do sorteio.

    return X_train, X_test, y_train, y_test


def compute_metrics(y_test, y_pred_test, y_train, y_pred_train):
    """Calcula as métricas de avaliação do modelo."""
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test, zero_division=0)
    test_precision = precision_score(y_test, y_pred_test, zero_division=0)
    test_recall = recall_score(y_test, y_pred_test, zero_division=0)
    overfitting = train_accuracy - test_accuracy

    return {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'test_f1_score': test_f1,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'overfitting': overfitting,
    }
