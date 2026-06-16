import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Importações atualizadas para Leave-One-Out
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ================= CONFIGURAÇÕES =================
PASTA_DO_SCRIPT = os.path.dirname(os.path.abspath(__file__))
ARQUIVO_DADOS = os.path.join(PASTA_DO_SCRIPT, 'DATASET_IA_PROCESSADO.csv')

# Configuração do Modelo
SEED = 42

# Intervalos de interesse (Igual ao anterior)
INTERVALOS_DE_INTERESSE = [
    (520, 580),  # Pico do Verde
    (690, 760)   # Red Edge
] 
# Para usar tudo: INTERVALOS_DE_INTERESSE = []

# ================= FUNÇÕES AUXILIARES =================

def extrair_valor_onda(nome_coluna):
    """Transforma 'd1_Band_720.5nm' em float 720.5"""
    try:
        limpo = nome_coluna.replace('d1_', '').replace('Band_', '').replace('nm', '')
        return float(limpo)
    except:
        return None

def filtrar_bandas(colunas, intervalos):
    """Retorna apenas as colunas que caem dentro dos intervalos definidos."""
    if not intervalos:
        return colunas
    
    cols_selecionadas = []
    print(f"\nFiltrando bandas nos intervalos: {intervalos}")
    for col in colunas:
        wl = extrair_valor_onda(col)
        if wl is not None:
            for (inicio, fim) in intervalos:
                if inicio <= wl <= fim:
                    cols_selecionadas.append(col)
                    break
    return cols_selecionadas

# ================= FUNÇÕES DE AVALIAÇÃO =================

def avaliar_modelo(y_true, y_pred, titulo, labels_nomes=None):
    """Gera matriz de confusão e relatório de métricas."""
    print(f"\n{'='*20} {titulo} {'='*20}")
    
    acc = accuracy_score(y_true, y_pred)
    print(f"Acurácia (Leave-One-Out): {acc:.2%}")
    
    print("\nRelatório de Classificação:")
    # target_names ajuda a deixar o relatório legível
    print(classification_report(y_true, y_pred, target_names=labels_nomes))
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels_nomes if labels_nomes else "auto",
                yticklabels=labels_nomes if labels_nomes else "auto")
    plt.title(f"Matriz de Confusão: {titulo}\n(Validado via Leave-One-Out)")
    plt.xlabel("Predição da IA")
    plt.ylabel("Real (Ground Truth)")
    plt.tight_layout()
    plt.show()

def plotar_importancia(modelo, colunas_X, titulo):
    """Mostra feature importance (do modelo treinado com todos os dados)."""
    importancias = modelo.feature_importances_
    indices = np.argsort(importancias)[::-1]
    
    top_n = min(15, len(colunas_X))
    indices_top = indices[:top_n]
    
    plt.figure(figsize=(10, 6))
    plt.title(f"Bandas Mais Importantes - {titulo}")
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
    
    for (inicio, fim) in INTERVALOS_DE_INTERESSE:
        plt.axvspan(inicio, fim, color='orange', alpha=0.1)

    plt.tight_layout()
    plt.show()

# ================= EXECUÇÃO PRINCIPAL =================

if __name__ == "__main__":
    # 1. Carregar Dados
    if not os.path.exists(ARQUIVO_DADOS):
        sys.exit("ERRO: Arquivo processado não encontrado.")
        
    df = pd.read_csv(ARQUIVO_DADOS, sep=';', decimal=',')
    print(f"Dataset carregado: {len(df)} amostras.")

    # 2. Preparar Features (X)
    cols_todas = [c for c in df.columns if c.startswith('d1_Band_') or c.startswith('Band_')]
    cols_features = filtrar_bandas(cols_todas, INTERVALOS_DE_INTERESSE)
    
    if len(cols_features) == 0:
        sys.exit("ERRO: Nenhuma banda nos intervalos selecionados.")
        
    print(f" > Bandas Selecionadas: {len(cols_features)}")

    X = df[cols_features]
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # Define o método de validação
    # LeaveOneOut não precisa de parâmetros, ele pega N splits automaticamente
    loo = LeaveOneOut()

    # ---------------------------------------------------------
    # MODELO 1: Classificação por DOSE
    # ---------------------------------------------------------
    y_dose = df['Dose_N'].astype(str)
    
    # Verifica se há classes com 1 amostra apenas (aviso apenas)
    if df['Dose_N'].value_counts().min() < 2:
        print("\nAVISO: Existem doses com apenas 1 amostra. No LOOCV, essa amostra será sempre classificada errada (pois não existe no treino).")

    print(f"\nRodando Leave-One-Out para Doses (Total de iterações: {len(X)})... aguarde.")
    
    # Instancia o modelo base
    rf_dose = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    
    # cross_val_predict executa o LOOCV completo e retorna as predições
    y_pred_dose_loo = cross_val_predict(rf_dose, X, y_dose, cv=loo, n_jobs=-1)
    
    classes_dose = sorted(y_dose.unique(), key=lambda x: float(x))
    avaliar_modelo(y_dose, y_pred_dose_loo, "Modelo Doses (LOOCV)", labels_nomes=classes_dose)
    
    # Para Feature Importance: Treina o modelo COM TUDO uma última vez
    rf_dose.fit(X, y_dose)
    plotar_importancia(rf_dose, cols_features, "Doses N")
    
    plotar_assinatura_media_por_classe(df, cols_features, 'Dose_N', "Doses de Nitrogênio")

    # ---------------------------------------------------------
    # MODELO 2: Classificação BINÁRIA
    # ---------------------------------------------------------
    print("\nRodando Leave-One-Out Binário...")
    doses_numericas = pd.to_numeric(df['Dose_N'], errors='coerce').fillna(0)
    y_binario = np.where(doses_numericas > 0, "Com Nitrogenio", "Sem Nitrogenio")
    
    rf_bin = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    
    # Executa LOOCV
    y_pred_bin_loo = cross_val_predict(rf_bin, X, y_binario, cv=loo, n_jobs=-1)
    
    avaliar_modelo(y_binario, y_pred_bin_loo, "Modelo Binário (LOOCV)")
    
    # Treina final para importância
    rf_bin.fit(X, y_binario)
    plotar_importancia(rf_bin, cols_features, "Binário")
    
    df['Status_N'] = np.where(doses_numericas > 0, "Com N", "Sem N")
    plotar_assinatura_media_por_classe(df, cols_features, 'Status_N', "Status Nitrogênio")
    
    print("\nProcesso concluído com sucesso!")