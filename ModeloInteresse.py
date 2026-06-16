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

# Configuração do Modelo
SEED = 42

# ### NOVO: DEFINIÇÃO DE INTERVALOS ###
# Defina aqui as faixas de nanômetros que você quer usar.
# Exemplo: Focar no Verde (500-600) e Red-Edge/NIR (700-850)
# Se deixar a lista vazia [], ele usa TODAS as bandas disponíveis.
INTERVALOS_DE_INTERESSE = [
    (520, 580),  # Pico do Verde (Refletância relacionada a clorofila)
    (690, 760)   # Red Edge (Região mais sensível a Nitrogênio)
] 
# Dica: Para usar tudo, deixe assim: INTERVALOS_DE_INTERESSE = []

# ================= FUNÇÕES AUXILIARES =================

def extrair_valor_onda(nome_coluna):
    """Transforma 'd1_Band_720.5nm' em float 720.5"""
    try:
        # Remove prefixos e sufixos comuns
        limpo = nome_coluna.replace('d1_', '').replace('Band_', '').replace('nm', '')
        return float(limpo)
    except:
        return None

def filtrar_bandas(colunas, intervalos):
    """Retorna apenas as colunas que caem dentro dos intervalos definidos."""
    if not intervalos:
        print(" > Nenhum intervalo definido. Usando espectro completo.")
        return colunas

    cols_selecionadas = []
    print(f"\nFiltrando bandas nos intervalos: {intervalos}")
    
    for col in colunas:
        wl = extrair_valor_onda(col)
        if wl is not None:
            # Verifica se o WL está dentro de ALGUM dos intervalos
            for (inicio, fim) in intervalos:
                if inicio <= wl <= fim:
                    cols_selecionadas.append(col)
                    break # Já achou, vai pra próxima coluna
    
    return cols_selecionadas

# ================= FUNÇÕES DE AVALIAÇÃO =================

def avaliar_modelo(y_teste, y_pred, titulo, labels_nomes=None):
    """Gera matriz de confusão e relatório de métricas."""
    print(f"\n{'='*20} {titulo} {'='*20}")
    
    # Acurácia
    acc = accuracy_score(y_teste, y_pred)
    print(f"Acurácia Global: {acc:.2%}")
    
    # Relatório Detalhado
    print("\nRelatório de Classificação:")
    print(classification_report(y_teste, y_pred, target_names=labels_nomes))
    
    # Matriz de Confusão Visual
    cm = confusion_matrix(y_teste, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels_nomes if labels_nomes else "auto",
                yticklabels=labels_nomes if labels_nomes else "auto")
    plt.title(f"Matriz de Confusão: {titulo}")
    plt.xlabel("Predição da IA")
    plt.ylabel("Real (Ground Truth)")
    plt.tight_layout()
    plt.show()

def plotar_importancia(modelo, colunas_X, titulo):
    """Mostra quais bandas foram mais importantes para a decisão."""
    importancias = modelo.feature_importances_
    indices = np.argsort(importancias)[::-1]
    
    # Pega as TOP 15 bandas (ou menos se tiver poucas)
    top_n = min(15, len(colunas_X))
    indices_top = indices[:top_n]
    
    plt.figure(figsize=(10, 6))
    plt.title(f"Top {top_n} Bandas Mais Importantes - {titulo}")
    plt.bar(range(top_n), importancias[indices_top], align="center", color='#2ca02c')
    plt.xticks(range(top_n), [colunas_X[i].replace('d1_Band_', '').replace('nm', '') + 'nm' for i in indices_top], rotation=45)
    plt.xlabel("Comprimento de Onda")
    plt.ylabel("Importância (Gini)")
    plt.tight_layout()
    plt.show()

def plotar_assinatura_media_por_classe(dataframe, colunas_bandas, coluna_alvo, titulo):
    plt.figure(figsize=(12, 6))
    
    wls = [extrair_valor_onda(c) for c in colunas_bandas]
    
    grupos = dataframe.groupby(coluna_alvo)[colunas_bandas].mean()
    
    if len(grupos) == 2:
        cores = ['red', 'green']
    else:
        cores = plt.cm.viridis(np.linspace(0, 1, len(grupos)))

    for i, (classe, linha_media) in enumerate(grupos.iterrows()):
        plt.plot(wls, linha_media, label=f"{coluna_alvo}: {classe}", color=cores[i], linewidth=2)

    plt.title(f"Assinatura Espectral Média por {titulo}", fontsize=14)
    plt.xlabel("Comprimento de Onda (nm)")
    plt.ylabel("Intensidade")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Marcação visual dos intervalos escolhidos no gráfico
    for (inicio, fim) in INTERVALOS_DE_INTERESSE:
        plt.axvspan(inicio, fim, color='orange', alpha=0.1)

    plt.tight_layout()
    plt.show()

# ================= EXECUÇÃO PRINCIPAL =================

if __name__ == "__main__":
    # 1. Carregar Dados
    if not os.path.exists(ARQUIVO_DADOS):
        print("ERRO: Arquivo processado não encontrado.")
        sys.exit()
        
    df = pd.read_csv(ARQUIVO_DADOS, sep=';', decimal=',')
    print(f"Dataset carregado: {len(df)} amostras.")

    # 2. Preparar Features (X)
    # Primeiro identifica TODAS as bandas disponíveis
    cols_todas = [c for c in df.columns if c.startswith('d1_Band_') or c.startswith('Band_')]
    
    # ### NOVO: APLICA O FILTRO DE INTERVALOS ###
    cols_features = filtrar_bandas(cols_todas, INTERVALOS_DE_INTERESSE)
    
    if len(cols_features) == 0:
        sys.exit("ERRO CRÍTICO: Nenhuma banda foi encontrada dentro dos intervalos definidos.")
        
    print(f" > Bandas Selecionadas: {len(cols_features)} de {len(cols_todas)} originais.")

    X = df[cols_features]
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # ---------------------------------------------------------
    # MODELO 1: Classificação por DOSE
    # ---------------------------------------------------------
    y_dose = df['Dose_N'].astype(str)
    
    # Verifica validade das classes
    if df['Dose_N'].value_counts().min() < 2:
        print("\nALERTA: Doses com amostra única detectadas. Pulando modelo multiclasse para evitar crash.")
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y_dose, test_size=0.3, random_state=SEED, stratify=y_dose)
        
        print("\nTreinando Modelo 1 (Doses Específicas)...")
        rf_dose = RandomForestClassifier(n_estimators=100, random_state=SEED)
        rf_dose.fit(X_train, y_train)
        
        y_pred_dose = rf_dose.predict(X_test)
        classes_dose = sorted(rf_dose.classes_, key=lambda x: float(x))
        
        avaliar_modelo(y_test, y_pred_dose, "Modelo Multiclasse (Doses N)", labels_nomes=classes_dose)
        plotar_importancia(rf_dose, cols_features, "Doses N")
        plotar_assinatura_media_por_classe(df, cols_features, 'Dose_N', "Doses de Nitrogênio")

    # ---------------------------------------------------------
    # MODELO 2: Classificação BINÁRIA
    # ---------------------------------------------------------
    print("\nPreparando Modelo 2 (Binário)...")
    doses_numericas = pd.to_numeric(df['Dose_N'], errors='coerce').fillna(0)
    y_binario = np.where(doses_numericas > 0, "Com Nitrogenio", "Sem Nitrogenio")
    
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X, y_binario, test_size=0.3, random_state=SEED, stratify=y_binario)
    
    rf_bin = RandomForestClassifier(n_estimators=100, random_state=SEED)
    rf_bin.fit(X_train_b, y_train_b)
    
    y_pred_bin = rf_bin.predict(X_test_b)
    avaliar_modelo(y_test_b, y_pred_bin, "Modelo Binário (Com vs Sem N)")
    plotar_importancia(rf_bin, cols_features, "Binário")
    
    # Prepara coluna para plot
    df['Status_N'] = np.where(doses_numericas > 0, "Com N", "Sem N")
    plotar_assinatura_media_por_classe(df, cols_features, 'Status_N', "Status Nitrogênio")
    
    print("\nProcesso concluído.")