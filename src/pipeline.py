# pipeline.py — define o pipeline de pré-processamento sklearn reprodutível.
# o scaler deve ser fitado APENAS nos dados de treino
# e depois aplicado nos de validação e teste — nunca o contrário.
# Se você calcular a média/desvio-padrão incluindo dados de validação ou teste,
# o modelo terá "visto o futuro" durante o treino. Isso se chama data leakage
# (vazamento de dados) e produz métricas otimistas que não se repetem em produção.
#
# O ColumnTransformer do sklearn formaliza esse processo:
#   preprocessor.fit(X_train)        -> aprende média/desvio só com treino
#   preprocessor.transform(X_val)    -> aplica o mesmo scaler no val (sem reaprender)
#   preprocessor.transform(X_test)   -> aplica o mesmo scaler no teste (sem reaprender)

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

# Colunas numéricas contínuas — valores podem variar muito em magnitude.
# Ex: Tenure Months vai de 1 a 72, Monthly Charges de ~20 a ~120.
# O StandardScaler transforma cada coluna para média 0 e desvio padrão 1,
# colocando as duas features na mesma escala e facilitando a convergência da rede.
NUMERICAL_COLS = ['Tenure Months', 'Monthly Charges']

# Colunas booleanas (0 ou 1) — já estão na mesma escala, não precisam de normalização.
BINARY_COLS = [
    'Gender_Male',
    'Partner_Yes',
    'Dependents_Yes',
    'Phone Service_Yes',
    'Multiple Lines_No phone service',
    'Multiple Lines_Yes',
    'Internet Service_Fiber optic',
    'Internet Service_No',
    'Online Security_No internet service',
    'Online Security_Yes',
    'Online Backup_No internet service',
    'Online Backup_Yes',
    'Device Protection_No internet service',
    'Device Protection_Yes',
    'Tech Support_No internet service',
    'Tech Support_Yes',
    'Streaming TV_No internet service',
    'Streaming TV_Yes',
    'Streaming Movies_No internet service',
    'Streaming Movies_Yes',
    'Contract_One year',
    'Contract_Two year',
    'Paperless Billing_Yes',
    'Payment Method_Credit card (automatic)',
    'Payment Method_Electronic check',
    'Payment Method_Mailed check',
]

# Ordem das colunas de saída após a transformação:
# o ColumnTransformer coloca primeiro as colunas do primeiro transformador,
# depois as do segundo — então numéricas primeiro, binárias depois.
OUTPUT_COLS = NUMERICAL_COLS + BINARY_COLS


def build_preprocessing_pipeline() -> ColumnTransformer:
    """Constrói o pipeline de pré-processamento sem fitá-lo.

    Usa ColumnTransformer para aplicar transformações diferentes por tipo de coluna:
    - StandardScaler nas numéricas: normaliza para média 0 e desvio padrão 1.
    - 'passthrough' nas binárias: mantém os valores 0/1 sem alterar.
    - remainder='drop': descarta qualquer coluna não listada (ex: a coluna 'target').

    O pipeline retornado ainda não foi fitado — ele só aprende os parâmetros
    (média, desvio) quando apply_preprocessing() chamar preprocessor.fit().
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('scaler', StandardScaler(), NUMERICAL_COLS),
            ('binary', 'passthrough', BINARY_COLS),
        ],
        remainder='drop',
    )
    return preprocessor


def apply_preprocessing(preprocessor: ColumnTransformer, df_train, df_val, df_test):
    """Fita o pipeline nos dados de treino e transforma os três conjuntos.

    Regra aplicada aqui:
      - fit() apenas no df_train -> o scaler aprende somente com dados do passado.
      - transform() nos três conjuntos -> aplica a mesma transformação de forma consistente.

    Isso garante que validação e teste sejam avaliados nas mesmas condições
    que o modelo encontrará em produção — sem qualquer vazamento de informação.

    Retorna:
        df_train, df_val, df_test transformados como DataFrames,
        com a coluna 'target' reinserida ao final.
    """
    # Remove a coluna target antes de passar pelo scaler
    X_train = df_train.drop(columns=['target'])
    X_val = df_val.drop(columns=['target'])
    X_test = df_test.drop(columns=['target'])

    # Fita SOMENTE no treino — aqui o scaler aprende média e desvio padrão
    preprocessor.fit(X_train)

    # Transforma os três conjuntos com os parâmetros aprendidos no treino
    df_train_proc = pd.DataFrame(preprocessor.transform(X_train), columns=OUTPUT_COLS)
    df_val_proc = pd.DataFrame(preprocessor.transform(X_val), columns=OUTPUT_COLS)
    df_test_proc = pd.DataFrame(preprocessor.transform(X_test), columns=OUTPUT_COLS)

    # Reinsere o target — necessário para o ChurnDataset separar features e rótulo
    df_train_proc['target'] = df_train['target'].values
    df_val_proc['target'] = df_val['target'].values
    df_test_proc['target'] = df_test['target'].values

    return df_train_proc, df_val_proc, df_test_proc
