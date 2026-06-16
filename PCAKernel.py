import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Importação necessária para 3D
from mpl_toolkits.mplot3d import Axes3D 
import seaborn as sns
import os
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics import silhouette_score

# ================= CONFIGURAÇÕES =================
ARQUIVO_DADOS = 'DATASET_IA_PROCESSADO.csv'
SEED = 53

# Níveis de Dose para Clusterização
DOSES_ALVO = [0, 90, 180, 360]

# Bandas específicas
BANDAS_ESPECIFICAS = [430, 450, 460, 500, 520, 550, 600, 625, 640, 660, 685, 720, 722, 740, 990, 995, 970]

# Configurações do Kernel PCA
KERNEL_TYPE = 'rbf'
GAMMA = None
N_COMPONENTS = 3

# ================= FUNÇÕES AUXILIARES =================

def extrair_valor_onda(nome_coluna):
    try:
        limpo = nome_coluna.replace('d1_', '').replace('Band_', '').replace('nm', '')
        return float(limpo)
    except:
        return None

def filtrar_por_doses_especificas(df, coluna_dose, doses):
    df_temp = df.copy()
    df_temp[coluna_dose] = pd.to_numeric(df_temp[coluna_dose], errors='coerce')
    mask = df_temp[coluna_dose].round(0).isin(doses)
    return df_temp[mask].copy()

def encontrar_colunas_por_bandas(colunas, bandas_alvo, tolerancia=1.0):
    cols_encontradas = []
    bandas_encontradas = []
    for banda_alvo in bandas_alvo:
        for col in colunas:
            wl = extrair_valor_onda(col)
            if wl is not None and abs(wl - banda_alvo) <= tolerancia:
                cols_encontradas.append(col)
                bandas_encontradas.append(banda_alvo)
                break
    return cols_encontradas, bandas_encontradas

def aplicar_analise_hibrida_3d(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 1. PCA Linear (Referência Estatística)
    pca_linear = PCA(n_components=min(len(X.columns), 10)) 
    pca_linear.fit(X_scaled)
    
    # 2. Kernel PCA (Projeção 3D)
    kpca = KernelPCA(n_components=N_COMPONENTS, kernel=KERNEL_TYPE, gamma=GAMMA, random_state=SEED)
    X_kpca = kpca.fit_transform(X_scaled)
    
    return {
        'scaler': scaler,
        'X_scaled': X_scaled,
        'pca_linear': pca_linear,
        'explained_variance': pca_linear.explained_variance_ratio_,
        'cumsum_variance': np.cumsum(pca_linear.explained_variance_ratio_),
        'X_kpca': X_kpca,
        'kpca_model': kpca
    }

def analisar_importancia_linear(pca_data, feature_names):
    pca = pca_data['pca_linear']
    loadings = pca.components_[:3, :].T 
    feature_contributions = np.sum(loadings**2, axis=1)
    feature_contributions_norm = feature_contributions / feature_contributions.max() * 100
    ordem = np.argsort(feature_contributions_norm)[::-1]
    return feature_contributions_norm, ordem

# ================= FUNÇÕES DE PLOTAGEM ATUALIZADAS =================

def plotar_variancia_explicada_linear(pca_data):
    cumsum = pca_data['cumsum_variance']
    expl_var = pca_data['explained_variance']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    n_show = min(len(expl_var), 10)
    ax1.bar(range(1, n_show + 1), expl_var[:n_show] * 100, color='#34495e')
    ax1.set_title('Variância por Componente (Ref. Linear)')
    ax2.plot(range(1, n_show + 1), cumsum[:n_show] * 100, 'bo-', linewidth=2)
    ax2.axhline(y=95, color='r', linestyle='--', label='95% variância')
    ax2.set_title('Variância Acumulada (Ref. Linear)')
    plt.tight_layout()
    plt.show()

def plotar_clusters_kpca_3d(pca_data, y_doses):
    """
    Visualiza Clusters + Centróides em 3D.
    """
    X_kpca = pca_data['X_kpca']
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    cores_dose = {0: '#440154', 90: '#31688e', 180: '#27ad81', 360: '#fde724'}
    doses_uniques = sorted(y_doses.unique())
    
    # Lista para legendas manuais (para evitar duplicatas no plot)
    legend_elements = []

    for dose in doses_uniques:
        mask = y_doses == dose
        points_dose = X_kpca[mask, :3]
        
        # 1. Plotar os pontos normais
        scatter = ax.scatter(
            points_dose[:, 0], points_dose[:, 1], points_dose[:, 2],
            label=f'{int(dose)} kg/ha',
            s=60, alpha=0.6, edgecolor='w', linewidth=0.2,
            color=cores_dose[dose]
        )
        
        # 2. Calcular e Plotar o CENTROIDE
        if len(points_dose) > 0:
            centroid = np.mean(points_dose, axis=0)
            
            # Marcador do centroide (X grande)
            ax.scatter(
                centroid[0], centroid[1], centroid[2],
                marker='X', s=300,  # Tamanho bem maior
                color=cores_dose[dose], 
                edgecolor='black', linewidth=2, # Borda preta para destaque
                alpha=1.0,
                label=f'Centro {int(dose)}'
            )
            
            # Texto indicativo flutuando um pouco acima do centroide
            ax.text(
                centroid[0], centroid[1], centroid[2] + (np.max(X_kpca)*0.05), 
                f"  {int(dose)}", 
                color='black', fontsize=10, fontweight='bold'
            )

    ax.set_title(f'Clusters 3D com Centroides (Kernel {KERNEL_TYPE.upper()})', fontsize=15, fontweight='bold', pad=20)
    ax.set_xlabel('Comp. 1', fontweight='bold')
    ax.set_ylabel('Comp. 2', fontweight='bold')
    ax.set_zlabel('Comp. 3', fontweight='bold')
    
    ax.view_init(elev=25, azim=-60)
    
    # Criar legenda apenas para as classes (filtrando duplicatas dos centroides)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # Filtramos para mostrar apenas as legendas das doses (não duplicar com "Centro")
    clean_dict = {k: v for k, v in by_label.items() if 'Centro' not in k}
    ax.legend(clean_dict.values(), clean_dict.keys(), title="Dose N", loc='best')
    
    plt.tight_layout()
    print("A janela 3D é interativa. Gire para ver a posição dos centroides.")
    plt.show()

def plotar_contribuicao_bandas_linear(pca_data, feature_names, top_n=15):
    contribs, ordem = analisar_importancia_linear(pca_data, feature_names)
    top_indices = ordem[:top_n]
    wavelengths = [extrair_valor_onda(feature_names[i]) for i in top_indices]
    contributions = contribs[top_indices]
    
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.4, 0.9, top_n))
    plt.barh(range(top_n), contributions[::-1], color=colors)
    plt.yticks(range(top_n), [f'{wl:.1f}nm' for wl in wavelengths[::-1]])
    plt.title(f'Top {top_n} Bandas (Referência Linear)')
    plt.tight_layout()
    plt.show()
    return ordem, contribs

# ================= EXECUÇÃO PRINCIPAL =================

if __name__ == "__main__":
    if not os.path.exists(ARQUIVO_DADOS):
        print("AVISO: Criando dataset dummy...")
        np.random.seed(SEED)
        N_SAMPLES = 200
        data = {'Dose_N': np.random.choice(DOSES_ALVO, N_SAMPLES)}
        base_signal = np.linspace(0, 1, len(BANDAS_ESPECIFICAS))
        for i, b in enumerate(BANDAS_ESPECIFICAS):
            # Criação de dados sintéticos que forçam uma separação 3D
            dose_factor = data['Dose_N'] / 360.0
            noise = np.random.normal(0, 0.15, N_SAMPLES)
            # Sinal complexo para justificar Kernel RBF
            signal = base_signal[i] * np.sin(dose_factor * np.pi + i) + dose_factor + noise
            data[f'd1_Band_{b}nm'] = signal
        df = pd.DataFrame(data)
    else:
        df_raw = pd.read_csv(ARQUIVO_DADOS, sep=';', decimal=',')
        df = filtrar_por_doses_especificas(df_raw, 'Dose_N', DOSES_ALVO)

    print(f"Processando Kernel PCA ({KERNEL_TYPE})...")

    # Seleção
    cols_derivada = [c for c in df.columns if 'Band' in c or 'nm' in c]
    cols_selecionadas, _ = encontrar_colunas_por_bandas(cols_derivada, BANDAS_ESPECIFICAS)
    
    X = df[cols_selecionadas].apply(pd.to_numeric, errors='coerce').fillna(0)
    y = df['Dose_N'].round(0)

    # Análise
    resultados = aplicar_analise_hibrida_3d(X)
    
    # Métricas
    s_score_3d = silhouette_score(resultados['X_kpca'], y)
    print(f"Silhouette Score (3D): {s_score_3d:.4f}")
    
    # Gráficos
    plotar_variancia_explicada_linear(resultados)
    
    # *** AQUI ESTÁ A VISUALIZAÇÃO COM CENTROIDES ***
    plotar_clusters_kpca_3d(resultados, y)
    
    plotar_contribuicao_bandas_linear(resultados, list(X.columns))