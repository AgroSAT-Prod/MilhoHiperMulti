import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Importações para SVM e Processamento
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ================= CONFIGURAÇÕES =================
PASTA_DO_SCRIPT = os.path.dirname(os.path.abspath(__file__))
ARQUIVO_DADOS = os.path.join(PASTA_DO_SCRIPT, 'DATASET_IA_PROCESSADO.csv')

SEED = 42

# --- CONFIGURAÇÃO DE INTERVALOS ---
# Filtrar apenas regiões úteis (Verde e Red-Edge)
INTERVALOS_DE_INTERESSE = [
    #(520, 580),  # Pico do Verde
    #(690, 760)   # Red Edge
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
    print(f"Acurácia Global (Leave-One-Out): {acc:.2%}")
    
    print("\nRelatório de Classificação:")
    print(classification_report(y_true, y_pred, target_names=labels_nomes))
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels_nomes if labels_nomes else "auto",
                yticklabels=labels_nomes if labels_nomes else "auto")
    plt.title(f"Matriz de Confusão: {titulo}\n(SVM Linear)")
    plt.xlabel("Predição da IA")
    plt.ylabel("Real (Ground Truth)")
    plt.tight_layout()
    plt.show()

def plotar_coeficientes_svm(modelo_treinado, colunas_X, titulo):
    """
    Mostra os pesos (coeficientes) do SVM Linear.
    Nota: Isso só funciona para kernel='linear'.
    """
    # Extrai o classificador de dentro do pipeline
    svc = modelo_treinado.named_steps['svc']
    
    if svc.kernel != 'linear':
        print("Aviso: Gráfico de importância só está disponível para SVM com kernel linear.")
        return

    # Se for multiclasse, o SVM gera vários coeficientes (um para cada classe vs resto).
    # Se for binário, gera apenas um array.
    coefs = svc.coef_
    
    # Se for binário (1 linha), pegamos ela. Se for multi, tiramos a média absoluta para ver relevância geral.
    if coefs.ndim > 1:
        importancia = np.mean(np.abs(coefs), axis=0)
    else:
        importancia = np.abs(coefs.flatten())

    indices = np.argsort(importancia)[::-1]
    top_n = min(15, len(colunas_X))
    indices_top = indices[:top_n]
    
    plt.figure(figsize=(10, 6))
    plt.title(f"Bandas Mais Importantes (Pesos SVM) - {titulo}")
    plt.bar(range(top_n), importancia[indices_top], align="center", color='#8b0000') # Vermelho escuro para SVM
    plt.xticks(range(top_n), [colunas_X[i].replace('d1_Band_', '').replace('nm', '') + 'nm' for i in indices_top], rotation=45)
    plt.xlabel("Comprimento de Onda")
    plt.ylabel("Peso Absoluto do Coeficiente")
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

    plt.title(f"Assinatura Espectral Média - {titulo}", fontsize=14)
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

    X = df[cols_features]
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # Define Validação Leave-One-Out
    loo = LeaveOneOut()

    # ---------------------------------------------------------
    # MODELO 1: Classificação por DOSE (Multiclasse SVM)
    # ---------------------------------------------------------
    y_dose = df['Dose_N'].astype(str)
    
    # Validação de classes mínimas
    if df['Dose_N'].value_counts().min() < 2:
        print("\nAVISO: Classes com apenas 1 amostra detectadas. O LOOCV falhará para essas amostras.")

    print(f"\nRodando SVM (Linear) para Doses com Leave-One-Out...")
    
    # CRIAÇÃO DO PIPELINE SVM
    # 1. StandardScaler: Normaliza os dados (Média 0, Desvio 1) - CRUCIAL para SVM
    # 2. SVC: O classificador SVM com kernel linear (para podermos ver a importância)
    pipeline_dose = make_pipeline(StandardScaler(), SVC(kernel='linear', C=1.0, random_state=SEED))
    
    # Executa LOOCV
    y_pred_dose_loo = cross_val_predict(pipeline_dose, X, y_dose, cv=loo, n_jobs=-1)
    
    classes_dose = sorted(y_dose.unique(), key=lambda x: float(x))
    avaliar_modelo(y_dose, y_pred_dose_loo, "SVM Multiclasse (Doses N)", labels_nomes=classes_dose)
    
    # Treina com tudo para gerar gráfico de importância
    pipeline_dose.fit(X, y_dose)
    plotar_coeficientes_svm(pipeline_dose, cols_features, "Doses N")
    
    plotar_assinatura_media_por_classe(df, cols_features, 'Dose_N', "Doses de Nitrogênio")

    # ---------------------------------------------------------
    # MODELO 2: Classificação BINÁRIA (SVM)
    # ---------------------------------------------------------
    print("\nRodando SVM Binário (Com N vs Sem N)...")
    doses_numericas = pd.to_numeric(df['Dose_N'], errors='coerce').fillna(0)
    y_binario = np.where(doses_numericas > 0, "Com Nitrogenio", "Sem Nitrogenio")
    
    # Pipeline Binário
    pipeline_bin = make_pipeline(StandardScaler(), SVC(kernel='linear', C=1.0, random_state=SEED))
    
    # Executa LOOCV
    y_pred_bin_loo = cross_val_predict(pipeline_bin, X, y_binario, cv=loo, n_jobs=-1)
    
    avaliar_modelo(y_binario, y_pred_bin_loo, "SVM Binário")
    
    # Gráficos Finais
    pipeline_bin.fit(X, y_binario)
    plotar_coeficientes_svm(pipeline_bin, cols_features, "Binário")
    
    df['Status_N'] = np.where(doses_numericas > 0, "Com N", "Sem N")
    plotar_assinatura_media_por_classe(df, cols_features, 'Status_N', "Status Nitrogênio")
    
    print("\nProcesso concluído com sucesso!")