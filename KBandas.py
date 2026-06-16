import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import shutil
import imageio.v2 as imageio 

# Importações para Machine Learning
from sklearn.ensemble import RandomForestClassifier

# ================= CONFIGURAÇÕES =================
PASTA_DO_SCRIPT = os.path.dirname(os.path.abspath(__file__))
ARQUIVO_DADOS = os.path.join(PASTA_DO_SCRIPT, 'DATASET_IA_PROCESSADO.csv')

# MUDANÇA 1: Nome do arquivo diferente para evitar CACHE do Windows/Navegador
NOME_GIF = os.path.join(PASTA_DO_SCRIPT, 'gif_lento_v2.gif') 
PASTA_FRAMES = os.path.join(PASTA_DO_SCRIPT, 'temp_frames')

SEED = 42

# MUDANÇA 2: Usar FPS baixo (Frames Por Segundo)
# 1 FPS = 1 segundo por imagem (Bem lento)
# 0.5 FPS = 2 segundos por imagem (Muito lento)
FPS_DESEJADO = 1 

INTERVALOS_DE_INTERESSE = [
    (520, 580),   # Verde
    (690, 760),   # Red-edge
]

# ================= FUNÇÕES AUXILIARES =================

def extrair_valor_onda(nome_coluna):
    try:
        limpo = nome_coluna.replace('d1_', '').replace('Band_', '').replace('nm', '')
        return float(limpo)
    except:
        return None

def filtrar_bandas(colunas, intervalos):
    if not intervalos:
        return colunas
    cols_selecionadas = []
    for col in colunas:
        wl = extrair_valor_onda(col)
        if wl is not None:
            for (inicio, fim) in intervalos:
                if inicio <= wl <= fim:
                    cols_selecionadas.append(col)
                    break
    return cols_selecionadas

def greedy_selecao_distante(importancias_dict, k, min_distancia):
    sorted_bands = sorted(importancias_dict.items(), key=lambda x: x[1], reverse=True)
    selecionadas = []
    wl_selecionadas = []
    
    for nome, imp in sorted_bands:
        if len(selecionadas) >= k:
            break
        wl = extrair_valor_onda(nome)
        if wl is None: continue
            
        pode_adicionar = True
        for wl_antiga in wl_selecionadas:
            if abs(wl - wl_antiga) < min_distancia:
                pode_adicionar = False
                break
        
        if pode_adicionar:
            selecionadas.append(nome)
            wl_selecionadas.append(wl)
            
    return selecionadas

def plotar_frame_para_gif(dataframe, cols_todas, cols_selecionadas, coluna_alvo, min_dist, k_meta, filename):
    plt.figure(figsize=(12, 6))
    
    cols_todas_sorted = sorted(cols_todas, key=extrair_valor_onda)
    wls_full = [extrair_valor_onda(c) for c in cols_todas_sorted]
    grupos_full = dataframe.groupby(coluna_alvo)[cols_todas_sorted].mean()

    if cols_selecionadas:
        cols_sel_sorted = sorted(cols_selecionadas, key=extrair_valor_onda)
        wls_sel = [extrair_valor_onda(c) for c in cols_sel_sorted]
        grupos_sel = dataframe.groupby(coluna_alvo)[cols_sel_sorted].mean()
    else:
        grupos_sel = pd.DataFrame()

    if len(grupos_full) == 2:
        cores = ['red', 'green']
    else:
        cores = plt.cm.viridis(np.linspace(0, 1, len(grupos_full)))

    for i, (classe, linha_media) in enumerate(grupos_full.iterrows()):
        cor_atual = cores[i] if i < len(cores) else 'blue'
        plt.plot(wls_full, linha_media, label=f"{classe}", color=cor_atual, linewidth=1.5, alpha=0.6)
        
        if not grupos_sel.empty and classe in grupos_sel.index:
            pontos_media = grupos_sel.loc[classe]
            plt.scatter(wls_sel, pontos_media, color=cor_atual, s=90, edgecolors='black', marker='o', zorder=5)

    plt.title(f"Seleção: Distância Mínima = {min_dist}nm | Bandas = {k_meta}", fontsize=15, fontweight='bold')
    plt.xlabel("Comprimento de Onda (nm)")
    plt.ylabel("Intensidade")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.ylim(dataframe[cols_todas].min().min() * 1.1, dataframe[cols_todas].max().max() * 1.1)
    
    for (inicio, fim) in INTERVALOS_DE_INTERESSE:
        plt.axvspan(inicio, fim, color='orange', alpha=0.1)

    plt.tight_layout()
    plt.savefig(filename, dpi=100)
    plt.close()

# ================= EXECUÇÃO PRINCIPAL =================

if __name__ == "__main__":
    if not os.path.exists(ARQUIVO_DADOS):
        sys.exit("ERRO: Arquivo processado não encontrado.")
        
    df = pd.read_csv(ARQUIVO_DADOS, sep=';', decimal=',')
    print(f"Dataset carregado: {len(df)} amostras.")

    cols_todas = [c for c in df.columns if c.startswith('d1_Band_') or c.startswith('Band_')]
    cols_features = filtrar_bandas(cols_todas, INTERVALOS_DE_INTERESSE)
    X = df[cols_features].apply(pd.to_numeric, errors='coerce').fillna(0)
    y_dose = df['Dose_N'].astype(str)

    print("Calculando importâncias...")
    rf_full = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    rf_full.fit(X, y_dose)
    importancias = dict(zip(cols_features, rf_full.feature_importances_))

    # Vamos de 80nm até 4nm (seleção inversa)
    distancias_teste = range(80, 4, -2) 
    CONSTANTE_INVERSA = 200 
    
    if os.path.exists(PASTA_FRAMES):
        shutil.rmtree(PASTA_FRAMES)
    os.makedirs(PASTA_FRAMES)

    print(f"Gerando frames...")
    arquivos_frames = []

    for i, dist in enumerate(distancias_teste):
        k_calculado = int(CONSTANTE_INVERSA / dist)
        k_calculado = max(1, min(k_calculado, 30))
        
        cols_sel = greedy_selecao_distante(importancias, k_calculado, dist)
        
        nome_frame = os.path.join(PASTA_FRAMES, f"frame_{i:03d}.png")
        plotar_frame_para_gif(df, cols_todas, cols_sel, 'Dose_N', dist, k_calculado, nome_frame)
        arquivos_frames.append(nome_frame)
        
        sys.stdout.write(f"\rProcessando: Distância {dist}nm -> K={k_calculado}   ")
        sys.stdout.flush()

    print("\n\nMontando GIF LENTO (Usando FPS)...")
    
    # Carregar imagens
    imagens_gif = []
    for filename in arquivos_frames:
        imagens_gif.append(imageio.imread(filename))
        
    # Pausa no final (adiciona 3 cópias do último frame)
    ultimo_frame = imagens_gif[-1]
    for _ in range(3):
        imagens_gif.append(ultimo_frame)

    # MUDANÇA PRINCIPAL: Usando fps=FPS_DESEJADO em vez de duration
    # Isso força 1 quadro por segundo
    imageio.mimsave(NOME_GIF, imagens_gif, fps=FPS_DESEJADO, loop=0)

    print(f"GIF salvo com sucesso em: {NOME_GIF}")
    print(f"Verifique se o arquivo '{os.path.basename(NOME_GIF)}' foi criado.")
    
    try:
        shutil.rmtree(PASTA_FRAMES)
        print("Limpeza concluída.")
    except:
        pass