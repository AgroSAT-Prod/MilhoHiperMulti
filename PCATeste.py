import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# ================= CONFIGURAÇÕES =================
# O arquivo deve estar na mesma pasta do script
ARQUIVO_DADOS = 'spectral_indices_rois.csv'
SEED = 42

# Níveis de Dose para Clusterização
DOSES_ALVO = [0, 90, 180, 360]

# Bandas específicas para análise (em nm)
BANDAS_ESPECIFICAS = [430, 450, 460, 500, 520, 550, 600, 625, 640, 660, 685, 720, 722, 740, 990, 995, 970]

# ================= FUNÇÕES AUXILIARES =================

def extrair_valor_onda(nome_coluna):
    """Extrai o valor numérico do comprimento de onda do nome da coluna."""
    try:
        limpo = nome_coluna.replace('d1_', '').replace('Band_', '').replace('nm', '')
        return float(limpo)
    except:
        return None

def filtrar_por_doses_especificas(df, coluna_dose, doses):
    """Filtra o dataframe apenas para as doses de interesse (0, 90, 180, 360)."""
    df_temp = df.copy()
    df_temp[coluna_dose] = pd.to_numeric(df_temp[coluna_dose], errors='coerce')
    # Filtra considerando uma pequena tolerância para arredondamento
    mask = df_temp[coluna_dose].round(0).isin(doses)
    return df_temp[mask].copy()

def encontrar_colunas_por_bandas(colunas, bandas_alvo, tolerancia=1.0):
    """
    Encontra as colunas que correspondem às bandas específicas.
    Usa tolerância para permitir pequenas variações no valor do comprimento de onda.
    """
    cols_encontradas = []
    bandas_encontradas = []
    bandas_nao_encontradas = []
    
    for banda_alvo in bandas_alvo:
        encontrou = False
        for col in colunas:
            wl = extrair_valor_onda(col)
            if wl is not None and abs(wl - banda_alvo) <= tolerancia:
                cols_encontradas.append(col)
                bandas_encontradas.append(banda_alvo)
                encontrou = True
                break
        if not encontrou:
            bandas_nao_encontradas.append(banda_alvo)
    
    return cols_encontradas, bandas_encontradas, bandas_nao_encontradas

def aplicar_pca(X):
    """Aplica PCA e retorna variância explicada e componentes."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    return {
        'pca': pca,
        'X_scaled': X_scaled,
        'X_pca': X_pca,
        'scaler': scaler,
        'explained_variance': pca.explained_variance_ratio_,
        'cumsum_variance': np.cumsum(pca.explained_variance_ratio_)
    }

def selecionar_bandas_por_pca(X, variancia_explicada=0.95):
    """Identifica bandas com maior contribuição para a variância."""
    pca_result = aplicar_pca(X)
    pca = pca_result['pca']
    
    cumsum = pca_result['cumsum_variance']
    n_componentes = np.argmax(cumsum >= variancia_explicada) + 1
    
    # Loadings (contribuição de cada banda nas componentes selecionadas)
    loadings = pca.components_[:n_componentes, :].T
    feature_names = X.columns.tolist()
    
    # Contribuição total: soma dos quadrados dos loadings
    feature_contributions = np.sum(loadings**2, axis=1)
    feature_contributions_norm = feature_contributions / feature_contributions.max() * 100
    ordem = np.argsort(feature_contributions_norm)[::-1]
    
    return {
        'pca_result': pca_result,
        'n_componentes': n_componentes,
        'loadings': loadings,
        'feature_names': feature_names,
        'feature_contributions': feature_contributions_norm,
        'feature_order': ordem
    }

# ================= FUNÇÕES DE PLOTAGEM =================

def plotar_variancia_explicada(pca_result):
    """Plota a variância explicada acumulada pelos componentes."""
    cumsum = pca_result['pca_result']['cumsum_variance']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Variância individual
    ax1.bar(range(1, len(pca_result['pca_result']['explained_variance']) + 1), 
            pca_result['pca_result']['explained_variance'] * 100)
    ax1.set_xlabel('Componente Principal')
    ax1.set_ylabel('Variância Explicada (%)')
    ax1.set_title('Variância Explicada por Componente')
    ax1.grid(True, alpha=0.3)
    
    # Variância acumulada
    ax2.plot(range(1, len(cumsum) + 1), cumsum * 100, 'bo-', linewidth=2, markersize=8)
    ax2.axhline(y=95, color='r', linestyle='--', label='95% variância')
    ax2.set_xlabel('Número de Componentes')
    ax2.set_ylabel('Variância Acumulada (%)')
    ax2.set_title('Variância Explicada Acumulada')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pca_variancia_explicada.png', dpi=300)
    plt.show()

def plotar_clusters_pca(pca_result, y_doses):
    """Visualiza como as amostras se agrupam nos 4 níveis de dose com áreas de domínio."""
    from scipy.spatial import ConvexHull
    from matplotlib.patches import Polygon
    
    X_pca = pca_result['pca_result']['X_pca']
    
    plt.figure(figsize=(14, 9))
    
    # Cores para cada dose
    cores_dose = {0: '#440154', 90: '#31688e', 180: '#35b779', 360: '#fde724'}
    doses_uniques = sorted(y_doses.unique())
    
    # Desenhar áreas de domínio (convex hull) para cada dose
    for dose in doses_uniques:
        mask = y_doses == dose
        X_dose = X_pca[mask, :2]
        
        if len(X_dose) >= 3:
            try:
                hull = ConvexHull(X_dose)
                hull_points = X_dose[hull.vertices]
                polygon = Polygon(hull_points, alpha=0.15, color=cores_dose[dose], edgecolor=cores_dose[dose], linewidth=2)
                plt.gca().add_patch(polygon)
            except:
                pass
    
    # Plotar os pontos
    for dose in doses_uniques:
        mask = y_doses == dose
        plt.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            label=f'{int(dose)} kg/ha',
            s=100, alpha=0.8, edgecolor='white', linewidth=0.5,
            color=cores_dose[dose]
        )
    
    plt.title('Clusters de Doses de Nitrogênio (PC1 vs PC2) com Áreas de Domínio', fontsize=14, fontweight='bold')
    plt.xlabel(f'PC1 ({pca_result["pca_result"]["explained_variance"][0]*100:.1f}%)')
    plt.ylabel(f'PC2 ({pca_result["pca_result"]["explained_variance"][1]*100:.1f}%)')
    plt.legend(title="Dose N (kg/ha)", loc='best', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('pca_clusters_doses.png', dpi=300)
    plt.show()

def plotar_biplot_clusters(pca_result, y_doses, top_features=10):
    """Biplot mostrando a direção das bandas em relação aos clusters."""
    X_pca = pca_result['pca_result']['X_pca']
    loadings = pca_result['loadings']
    feature_names = pca_result['feature_names']
    
    plt.figure(figsize=(12, 9))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_doses.astype(str), palette="viridis", alpha=0.3, s=60)
    
    scale_factor = np.abs(X_pca).max() * 0.8
    top_indices = pca_result['feature_order'][:top_features]
    
    for i in top_indices:
        plt.arrow(0, 0, loadings[i, 0]*scale_factor, loadings[i, 1]*scale_factor,
                 color='red', alpha=0.6, head_width=scale_factor*0.02)
        wl = extrair_valor_onda(feature_names[i])
        plt.text(loadings[i, 0]*scale_factor*1.1, loadings[i, 1]*scale_factor*1.1, 
                 f'{wl:.1f}nm', color='darkred', fontweight='bold', fontsize=9)

    plt.title(f'Biplot PCA: Top {top_features} Bandas vs Clusters de Dose', fontsize=14)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig('pca_biplot_clusters.png', dpi=300)
    plt.show()

def plotar_contribuicao_bandas(pca_result, top_n=15):
    """Plota as bandas com maior contribuição para o PCA."""
    feature_names = pca_result['feature_names']
    feature_contribs = pca_result['feature_contributions']
    feature_order = pca_result['feature_order']
    
    top_indices = feature_order[:top_n]
    wavelengths = [extrair_valor_onda(feature_names[i]) for i in top_indices]
    contributions = feature_contribs[top_indices]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(top_n), contributions[::-1])
    plt.yticks(range(top_n), [f'{wl:.1f}nm' for wl in wavelengths[::-1]])
    plt.xlabel('Importância PCA (%)')
    plt.title(f'Top {top_n} Bandas Espectrais para Diferenciação de Doses')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig('pca_contribuicao_bandas.png', dpi=300)
    plt.show()

# ================= EXECUÇÃO PRINCIPAL =================

if __name__ == "__main__":
    if not os.path.exists(ARQUIVO_DADOS):
        sys.exit(f"ERRO: Arquivo {ARQUIVO_DADOS} não encontrado.")
    
    # 1. Carregamento e Filtragem por Doses Clusterizadas
    df_raw = pd.read_csv(ARQUIVO_DADOS, sep=';', decimal=',')
    df = filtrar_por_doses_especificas(df_raw, 'Dose_N', DOSES_ALVO)
    
    print(f"Dataset carregado e filtrado para as doses: {DOSES_ALVO}")
    print(f"Total de amostras: {len(df)}")

    # 2. Encontrar colunas que correspondem às bandas específicas
    cols_derivada = [c for c in df.columns if c.startswith('d1_Band_')]
    cols_selecionadas, bandas_encontradas, bandas_nao_encontradas = encontrar_colunas_por_bandas(
        cols_derivada, BANDAS_ESPECIFICAS, tolerancia=1.0
    )
    
    if bandas_nao_encontradas:
        print(f"\n⚠ AVISO: As seguintes bandas não foram encontradas no dataset:")
        print(f"  {bandas_nao_encontradas}")
    
    print(f"\n✓ {len(cols_selecionadas)} bandas encontradas no dataset:")
    print(f"  {sorted(bandas_encontradas)}")

    # 3. Preparar dados para PCA
    X = df[cols_selecionadas].apply(pd.to_numeric, errors='coerce').fillna(0)
    y = df['Dose_N'].round(0)

    # 4. Análise PCA
    print("\nCalculando PCA para bandas especificadas...")
    pca_result = selecionar_bandas_por_pca(X, variancia_explicada=0.90)
    
    # 5. Cálculo da Qualidade da Clusterização (Silhouette Score)
    s_score = silhouette_score(pca_result['pca_result']['X_pca'][:, :2], y)
    print(f"Silhouette Score (PC1 e PC2): {s_score:.4f} (Melhor se próximo de 1)")
    
    # 6. Variância explicada pelos 2 primeiros componentes
    var_pc1 = pca_result['pca_result']['explained_variance'][0] * 100
    var_pc2 = pca_result['pca_result']['explained_variance'][1] * 100
    print(f"Variância explicada por PC1: {var_pc1:.2f}%")
    print(f"Variância explicada por PC2: {var_pc2:.2f}%")
    print(f"Variância acumulada (PC1 + PC2): {var_pc1 + var_pc2:.2f}%")

    # 7. Visualizações
    print("\nGerando visualizações...")
    plotar_variancia_explicada(pca_result)
    plotar_clusters_pca(pca_result, y)
    plotar_biplot_clusters(pca_result, y, top_features=min(12, len(cols_selecionadas)))
    plotar_contribuicao_bandas(pca_result, top_n=min(15, len(cols_selecionadas)))

    # 8. Relatório Final de Bandas Chave para os Clusters
    print(f"\n{'='*70}")
    print(f" RANKING DAS BANDAS PARA DIFERENCIAR DOSES {DOSES_ALVO}")
    print(f"{'='*70}")
    
    feature_order = pca_result['feature_order']
    feature_names = pca_result['feature_names']
    feature_contribs = pca_result['feature_contributions']
    
    print(f"{'Rank':<5} {'Comprimento (nm)':<20} {'Importância PCA (%)':<20}")
    print("-" * 70)
    
    for rank, idx in enumerate(feature_order, 1):
        wl = extrair_valor_onda(feature_names[idx])
        importance = feature_contribs[idx]
        print(f"{rank:<5} {wl:<20.1f} {importance:.2f}%")

    print(f"\n{'='*70}")
    print(" ✓ ANÁLISE DE CLUSTERS CONCLUÍDA")
    print(f"{'='*70}")