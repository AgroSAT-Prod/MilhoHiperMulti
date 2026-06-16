import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Importações para estatística
from scipy.stats import spearmanr

# Importações para Machine Learning
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ================= CONFIGURAÇÕES =================
PASTA_DO_SCRIPT = os.path.dirname(os.path.abspath(__file__))
ARQUIVO_DADOS = os.path.join(PASTA_DO_SCRIPT, 'DATASET_IA_PROCESSADO.csv')

# Configurações de Seleção
SEED = 42
NUM_VARIAVEIS_DESEJADAS = 20
DISTANCIA_MINIMA_NM = 10.0  # <--- NOVA TRAVA: Bandas devem estar a pelo menos 10nm de distância

# Intervalos de interesse
INTERVALOS_DE_INTERESSE = [
    (520, 580),   # Pico do Verde
    (690, 760)    # Red Edge
]

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

# ================= NOVA FUNÇÃO: SELEÇÃO COM INIBIÇÃO FÍSICA =================

def selecionar_bandas_por_distancia(X, y_target, n_selecionadas=20, min_dist_nm=10.0):
    """
    Seleciona as bandas mais correlacionadas, proibindo a seleção 
    de bandas vizinhas (dentro de um raio de min_dist_nm).
    """
    print(f"\n{'='*20} SELEÇÃO COM INIBIÇÃO DE VIZINHANÇA {'='*20}")
    print(f"Buscando {n_selecionadas} bandas com separação mínima de {min_dist_nm}nm...")

    # 1. Calcular correlação de TODAS as bandas com o alvo
    relevancia = []
    for col in X.columns:
        # Usa valor absoluto da correlação de Spearman (captura relações não lineares)
        score = abs(spearmanr(X[col], y_target)[0])
        wl = extrair_valor_onda(col)
        if wl is not None:
            relevancia.append({'coluna': col, 'score': score, 'wl': wl})
    
    # Cria DataFrame e ordena da MELHOR para a PIOR correlação
    df_rel = pd.DataFrame(relevancia).sort_values('score', ascending=False)
    
    bandas_escolhidas = []
    comprimentos_escolhidos = []

    # 2. Loop Guloso (Greedy) com Bloqueio
    for idx, row in df_rel.iterrows():
        candidata_col = row['coluna']
        candidata_wl = row['wl']
        
        # Verifica se esta candidata está muito perto de alguma já escolhida
        esta_perto = False
        for wl_escolhido in comprimentos_escolhidos:
            if abs(candidata_wl - wl_escolhido) < min_dist_nm:
                esta_perto = True
                break
        
        # Se não estiver perto de ninguém, seleciona
        if not esta_perto:
            bandas_escolhidas.append(candidata_col)
            comprimentos_escolhidos.append(candidata_wl)
            # print(f" > Selecionada: {candidata_wl}nm (Score: {row['score']:.4f})")
        
        # Se já atingiu o número desejado, para
        if len(bandas_escolhidas) >= n_selecionadas:
            break
    
    # Reordena as escolhidas pelo comprimento de onda para ficar bonito no print/plot
    # (O zip cria pares (wl, nome), o sorted ordena pelo wl, e depois descompactamos)
    final_sorted = sorted(zip(comprimentos_escolhidos, bandas_escolhidas))
    bandas_finais = [b for _, b in final_sorted]
    
    return bandas_finais

# ================= FUNÇÕES DE VISUALIZAÇÃO =================

def avaliar_modelo(y_true, y_pred, titulo, labels_nomes=None):
    acc = accuracy_score(y_true, y_pred)
    print(f"\n[{titulo}] Acurácia (LOOCV): {acc:.2%}")
    # Ocultando relatório completo para economizar espaço no terminal se desejar
    # print(classification_report(y_true, y_pred, target_names=labels_nomes))
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels_nomes if labels_nomes else "auto",
                yticklabels=labels_nomes if labels_nomes else "auto")
    plt.title(f"Matriz de Confusão: {titulo}")
    plt.xlabel("Predição")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.show()

def plotar_importancia(modelo, colunas_X, titulo):
    importancias = modelo.feature_importances_
    indices = np.argsort(importancias)[::-1]
    top_n = min(15, len(colunas_X))
    indices_top = indices[:top_n]
    
    plt.figure(figsize=(10, 5))
    plt.title(f"Bandas Mais Importantes - {titulo}")
    nomes_limpos = [colunas_X[i].replace('d1_Band_', '').replace('nm', '') + 'nm' for i in indices_top]
    plt.bar(range(top_n), importancias[indices_top], align="center", color='#2ca02c')
    plt.xticks(range(top_n), nomes_limpos, rotation=45)
    plt.tight_layout()
    plt.show()

def plotar_assinatura_escolhida(dataframe, colunas_bandas, coluna_alvo, titulo):
    # Garante ordenação
    colunas_ordenadas = sorted(colunas_bandas, key=lambda x: extrair_valor_onda(x))
    wls = [extrair_valor_onda(c) for c in colunas_ordenadas]
    
    grupos = dataframe.groupby(coluna_alvo)[colunas_ordenadas].mean()
    
    plt.figure(figsize=(12, 6))
    if len(grupos) == 2:
        cores = ['red', 'green']
    else:
        cores = plt.cm.viridis(np.linspace(0, 1, len(grupos)))

    for i, (classe, linha_media) in enumerate(grupos.iterrows()):
        plt.plot(wls, linha_media, label=f"{classe}", color=cores[i], linewidth=2, marker='o')

    # Adiciona linhas verticais para mostrar onde as bandas caíram
    for wl in wls:
        plt.axvline(x=wl, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)

    plt.title(f"Bandas Selecionadas e Assinatura Média - {titulo}", fontsize=14)
    plt.xlabel("Comprimento de Onda (nm)")
    plt.ylabel("Intensidade")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# ================= EXECUÇÃO PRINCIPAL =================

if __name__ == "__main__":
    if not os.path.exists(ARQUIVO_DADOS):
        sys.exit("ERRO: Ficheiro de dados não encontrado.")
        
    df = pd.read_csv(ARQUIVO_DADOS, sep=';', decimal=',')
    print(f"Dados carregados: {len(df)} amostras.")

    # 1. Preparação Inicial
    cols_todas = [c for c in df.columns if c.startswith('d1_Band_') or c.startswith('Band_')]
    cols_candidatas = filtrar_bandas(cols_todas, INTERVALOS_DE_INTERESSE)
    
    X_full = df[cols_candidatas].apply(pd.to_numeric, errors='coerce').fillna(0)
    y_target_selection = pd.to_numeric(df['Dose_N'], errors='coerce').fillna(0)

    # 2. APLICA A NOVA SELEÇÃO
    cols_finais = selecionar_bandas_por_distancia(
        X_full, 
        y_target_selection, 
        n_selecionadas=NUM_VARIAVEIS_DESEJADAS, 
        min_dist_nm=DISTANCIA_MINIMA_NM
    )
    
    print(f"\nBandas Selecionadas ({len(cols_finais)}):")
    wl_finais = [extrair_valor_onda(c) for c in cols_finais]
    print(wl_finais)

    X = X_full[cols_finais]
    loo = LeaveOneOut()

    # 3. Classificação por DOSE
    y_dose = df['Dose_N'].astype(str)
    print("\n[1/2] Classificando por Doses...")
    rf_dose = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    y_pred_dose = cross_val_predict(rf_dose, X, y_dose, cv=loo, n_jobs=-1)
    
    classes_dose = sorted(y_dose.unique(), key=lambda x: float(x))
    avaliar_modelo(y_dose, y_pred_dose, "Doses (LOOCV)", labels_nomes=classes_dose)
    
    rf_dose.fit(X, y_dose)
    plotar_assinatura_escolhida(df, cols_finais, 'Dose_N', "Doses")

    # 4. Classificação BINÁRIA
    print("\n[2/2] Classificando Binário (Com N vs Sem N)...")
    y_binario = np.where(y_target_selection > 0, "Com Nitrogenio", "Sem Nitrogenio")
    rf_bin = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    y_pred_bin = cross_val_predict(rf_bin, X, y_binario, cv=loo, n_jobs=-1)
    
    avaliar_modelo(y_binario, y_pred_bin, "Binário (LOOCV)")
    
    rf_bin.fit(X, y_binario)
    plotar_importancia(rf_bin, cols_finais, "Binário")
    
    print("\nProcesso terminado.")