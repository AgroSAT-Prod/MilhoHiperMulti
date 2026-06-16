import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ================= CONFIGURAÇÕES =================
PASTA_DO_SCRIPT = os.path.dirname(os.path.abspath(__file__))
ARQUIVO_DADOS = os.path.join(PASTA_DO_SCRIPT, 'DATASET_IA_PROCESSADO.csv')

SEED = 42 

# ================= FUNÇÕES DE AVALIAÇÃO E PLOTAGEM =================

def plotar_correlacao_pearson(dataframe, colunas_bandas, coluna_alvo, titulo):
    """
    Calcula e plota a Correlação de Pearson (r) entre cada banda e o alvo.
    """
    print(f"\nCalculando Correlação de Pearson para: {titulo}...")
    
    correlacoes = []
    wls = []
    
    # Garante que o alvo seja numérico
    y_target = pd.to_numeric(dataframe[coluna_alvo], errors='coerce')
    
    if y_target.isnull().all():
        print("ERRO: A coluna alvo não é numérica. Pearson requer números.")
        return

    # Itera sobre as bandas
    for col in colunas_bandas:
        try:
            # Extrai o comprimento de onda do nome da coluna
            wl = float(col.replace('d1_Band_', '').replace('Band_', '').replace('nm', ''))
            
            # Calcula Pearson
            r = dataframe[col].corr(y_target)
            
            correlacoes.append(r)
            wls.append(wl)
        except Exception as e:
            continue

    # Cria DataFrame temporário para facilitar a plotagem ordenada
    df_corr = pd.DataFrame({'Wavelength': wls, 'Pearson_r': correlacoes})
    df_corr = df_corr.sort_values(by='Wavelength')

    # --- PLOTAGEM ---
    plt.figure(figsize=(12, 6))
    
    # Plota a linha de correlação
    plt.plot(df_corr['Wavelength'], df_corr['Pearson_r'], color='#1f77b4', linewidth=2, label='Correlação (r)')
    
    # Preenche a área para destacar a magnitude
    plt.fill_between(df_corr['Wavelength'], df_corr['Pearson_r'], 0, alpha=0.2, color='#1f77b4')
    
    # Linhas de referência
    plt.axhline(0, color='black', linewidth=1, linestyle='--')
    plt.axhline(0.5, color='red', linewidth=0.5, linestyle=':', label='Correlação Forte (+/- 0.5)')
    plt.axhline(-0.5, color='red', linewidth=0.5, linestyle=':')
    
    # Estilização
    plt.title(f"Correlação de Pearson por Comprimento de Onda - {titulo}", fontsize=14)
    plt.xlabel("Comprimento de Onda (nm)")
    plt.ylabel("Coeficiente de Pearson (r)")
    plt.ylim(-1.1, 1.1)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    
    # Destaque Região Red Edge (Onde a clorofila atua mais)
    plt.axvspan(690, 750, color='green', alpha=0.1, label='Red Edge')
    
    plt.tight_layout()
    plt.show()

# (Mantenha as outras funções: avaliar_modelo, plotar_importancia, etc. aqui...)
def avaliar_modelo(y_teste, y_pred, titulo, labels_nomes=None):
    print(f"\n{'='*20} {titulo} {'='*20}")
    acc = accuracy_score(y_teste, y_pred)
    print(f"Acurácia Global: {acc:.2%}")
    try:
        print(classification_report(y_teste, y_pred, target_names=labels_nomes))
    except:
        print(classification_report(y_teste, y_pred))
    cm = confusion_matrix(y_teste, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels_nomes if labels_nomes else "auto",
                yticklabels=labels_nomes if labels_nomes else "auto")
    plt.title(f"Matriz de Confusão: {titulo}")
    plt.tight_layout()
    plt.show()

def plotar_importancia(modelo, colunas_X, titulo):
    importancias = modelo.feature_importances_
    indices = np.argsort(importancias)[::-1]
    top_n = 15
    indices_top = indices[:top_n]
    plt.figure(figsize=(10, 6))
    plt.title(f"Top {top_n} Features (Random Forest) - {titulo}")
    plt.bar(range(top_n), importancias[indices_top], align="center", color='#2ca02c')
    plt.xticks(range(top_n), [colunas_X[i].replace('d1_Band_', '').replace('nm', '') + 'nm' for i in indices_top], rotation=45)
    plt.tight_layout()
    plt.show()

# ================= EXECUÇÃO PRINCIPAL =================

if __name__ == "__main__":
    if not os.path.exists(ARQUIVO_DADOS):
        sys.exit("ERRO: Arquivo processado não encontrado.")
        
    df = pd.read_csv(ARQUIVO_DADOS, sep=';', decimal=',')
    print(f"Dataset carregado: {len(df)} amostras.")

    # Prepara Features Numéricas
    cols_features = [c for c in df.columns if c.startswith('d1_Band_') or c.startswith('Band_')]
    df[cols_features] = df[cols_features].apply(pd.to_numeric, errors='coerce').fillna(0)
    X = df[cols_features]

    # ---------------------------------------------------------
    # ANÁLISE 1: DOSES DE NITROGÊNIO
    # ---------------------------------------------------------
    print("\n--- [1] Análise por DOSES ---")
    
    # 1.1 Gráfico de Pearson para Doses (Usa a dose numérica)
    # Certifique-se de que a coluna Dose_N é numérica para o cálculo
    df['Dose_Num'] = pd.to_numeric(df['Dose_N'], errors='coerce').fillna(0)
    plotar_correlacao_pearson(df, cols_features, 'Dose_Num', "Dose de Nitrogênio (Kg/ha)")

    # 1.2 Modelo Random Forest
    y_dose = df['Dose_N'].astype(str)
    # (Adicione aqui sua lógica de split augmented se tiver, ou split simples abaixo)
    X_train, X_test, y_train, y_test = train_test_split(X, y_dose, test_size=0.3, random_state=SEED, stratify=y_dose)
    
    rf_dose = RandomForestClassifier(n_estimators=100, random_state=SEED)
    rf_dose.fit(X_train, y_train)
    plotar_importancia(rf_dose, cols_features, "Doses N")

    # ---------------------------------------------------------
    # ANÁLISE 2: BINÁRIO (COM vs SEM)
    # ---------------------------------------------------------
    print("\n--- [2] Análise BINÁRIA ---")
    
    # Cria coluna Numérica para Pearson (0 e 1)
    # 0 = Sem Nitrogênio, 1 = Com Nitrogênio
    doses_float = pd.to_numeric(df['Dose_N'], errors='coerce').fillna(0)
    df['Binario_Num'] = np.where(doses_float > 0, 1, 0)
    
    # 2.1 Gráfico de Pearson Binário (Point-Biserial Correlation)
    plotar_correlacao_pearson(df, cols_features, 'Binario_Num', "Status Binário (0=Sem, 1=Com)")

    # 2.2 Modelo Random Forest Binário
    y_binario = np.where(doses_float > 0, "Com N", "Sem N")
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X, y_binario, test_size=0.3, random_state=SEED, stratify=y_binario)
    
    rf_bin = RandomForestClassifier(n_estimators=100, random_state=SEED)
    rf_bin.fit(X_train_b, y_train_b)
    plotar_importancia(rf_bin, cols_features, "Binário")
    
    print("\n✅ Concluído.")