import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import os
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics import silhouette_score

# ================= CONFIGURAÇÕES =================
ARQUIVO_DADOS = 'spectral_indices_rois.csv'
ARQUIVO_AGRONOMICO = os.path.join('Dataset', 'PlanilhaFiltrada.xlsx')
SEED = 53

# Níveis de Dose para Clusterização
DOSES_ALVO = [0, 90, 180, 360]

# Bandas específicas para análise
BANDAS_ESPECIFICAS = [430, 450, 460, 500, 520, 550, 600, 625, 640, 660, 685, 720, 722, 740, 970, 990, 995]

# Configurações do Kernel PCA
KERNEL_TYPE = 'rbf'
GAMMA = None
N_COMPONENTS = 3

# ================= FUNÇÕES AUXILIARES =================

def extrair_valor_onda(nome_coluna):
    """Extrai o valor numérico do comprimento de onda."""
    try:
        limpo = nome_coluna.replace('d1_', '').replace('Band_', '').replace('nm', '')
        return float(limpo)
    except:
        return None

def cruzar_com_excel(df_espectral, arquivo_excel):
    """Cruza dados espectrais com agronômicos do Excel."""
    try:
        df_agro = pd.read_excel(arquivo_excel, header=2)
        df_agro.columns = [str(c).strip().lower() for c in df_agro.columns]
        
        print("\n✓ Dados agronômicos carregados do Excel")
        
    except Exception as e:
        print(f"⚠️  Erro ao ler Excel: {e}")
        df_espectral['Dose_N'] = 0
        return df_espectral
    
    # Mapear colunas
    col_id = next((c for c in df_agro.columns if 'id' in c and 'parcela' in c), 'id parcela')
    col_dose = next((c for c in df_agro.columns if 'dose' in c), None)
    col_clor_m = next((c for c in df_agro.columns if 'médio' in c and 'clorofila' in c), None)
    col_clor_s = next((c for c in df_agro.columns if 'superior' in c and 'clorofila' in c), None)
    
    # Cruzar dados
    df_espectral['ID_Numeric'] = df_espectral['ID_Amostra'].astype(int)
    df_merge = df_espectral.merge(
        df_agro[[col_id, col_dose, col_clor_m, col_clor_s]],
        left_on='ID_Numeric',
        right_on=col_id,
        how='left'
    )
    
    # Selecionar clorofila (M ou S)
    df_merge['Y_Clorofila'] = np.where(
        df_merge['Parte'] == 'M',
        df_merge[col_clor_m],
        df_merge[col_clor_s]
    )
    
    df_merge.rename(columns={col_dose: 'Dose_N'}, inplace=True)
    
    return df_merge[list(df_espectral.columns) + ['Dose_N', 'Y_Clorofila']]

def filtrar_por_doses_especificas(df, coluna_dose, doses):
    """Filtra para as doses de interesse."""
    df_temp = df.copy()
    df_temp[coluna_dose] = pd.to_numeric(df_temp[coluna_dose], errors='coerce')
    mask = df_temp[coluna_dose].round(0).isin(doses)
    return df_temp[mask].copy()

def encontrar_colunas_por_bandas(colunas, bandas_alvo, tolerancia=1.0):
    """Encontra colunas que correspondem às bandas específicas."""
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
    """Aplica PCA linear + Kernel PCA 3D."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA Linear (referência)
    pca_linear = PCA(n_components=min(len(X.columns), 10))
    pca_linear.fit(X_scaled)
    
    # Kernel PCA (projeção 3D)
    kpca = KernelPCA(n_components=N_COMPONENTS, kernel=KERNEL_TYPE, 
                     gamma=GAMMA, random_state=SEED, n_jobs=-1)
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
    """Analisa importância das features no PCA linear."""
    pca = pca_data['pca_linear']
    loadings = pca.components_[:min(3, pca.n_components_), :].T
    feature_contributions = np.sum(loadings**2, axis=1)
    feature_contributions_norm = feature_contributions / feature_contributions.max() * 100
    ordem = np.argsort(feature_contributions_norm)[::-1]
    return feature_contributions_norm, ordem

# ================= FUNÇÕES DE PLOTAGEM =================

def plotar_variancia_explicada_linear(pca_data):
    """Plota variância explicada pelo PCA linear."""
    cumsum = pca_data['cumsum_variance']
    expl_var = pca_data['explained_variance']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    n_show = min(len(expl_var), 10)
    ax1.bar(range(1, n_show + 1), expl_var[:n_show] * 100, color='#34495e')
    ax1.set_xlabel('Componente Principal')
    ax1.set_ylabel('Variância Explicada (%)')
    ax1.set_title('Variância por Componente (Ref. Linear)')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(range(1, n_show + 1), cumsum[:n_show] * 100, 'bo-', linewidth=2, markersize=8)
    ax2.axhline(y=95, color='r', linestyle='--', label='95% variância')
    ax2.set_xlabel('Número de Componentes')
    ax2.set_ylabel('Variância Acumulada (%)')
    ax2.set_title('Variância Acumulada (Ref. Linear)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('kpca_variancia_linear.png', dpi=300)
    print("✓ Salvo: kpca_variancia_linear.png")
    plt.show()

def plotar_clusters_kpca_3d(pca_data, y_doses):
    """Visualiza clusters 3D com centroides e checkboxes interativos."""
    from matplotlib.widgets import CheckButtons
    X_kpca = pca_data['X_kpca']
    
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(121, projection='3d')
    
    cores_dose = {0: '#440154', 90: '#31688e', 180: '#35b779', 360: '#fde724'}
    doses_uniques = sorted(y_doses.unique())
    
    # Dicionários para armazenar artists
    scatter_artists = {}
    centroid_artists = {}
    text_artists = {}
    
    # Plotar pontos e centroides para cada dose
    for dose in doses_uniques:
        mask = y_doses == dose
        points_dose = X_kpca[mask, :3]
        
        # Pontos da classe
        scatter = ax.scatter(
            points_dose[:, 0], points_dose[:, 1], points_dose[:, 2],
            label=f'{int(dose)} kg/ha',
            s=80, alpha=0.7, edgecolor='white', linewidth=0.5,
            color=cores_dose.get(dose, '#808080')
        )
        scatter_artists[dose] = scatter
        
        # Centroide
        if len(points_dose) > 0:
            centroid = np.mean(points_dose, axis=0)
            
            centroid_scatter = ax.scatter(
                centroid[0], centroid[1], centroid[2],
                marker='X', s=400,
                color=cores_dose.get(dose, '#808080'),
                edgecolor='black', linewidth=2,
                alpha=1.0,
                zorder=1000
            )
            centroid_artists[dose] = centroid_scatter
            
            offset = np.max(np.abs(X_kpca)) * 0.05
            text = ax.text(
                centroid[0] + offset, centroid[1] + offset, centroid[2] + offset,
                f"{int(dose)}", color='black', fontsize=11, fontweight='bold'
            )
            text_artists[dose] = text
    
    ax.set_title(f'Clusters 3D com Centroides (Kernel {KERNEL_TYPE.upper()})', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Componente 1', fontweight='bold')
    ax.set_ylabel('Componente 2', fontweight='bold')
    ax.set_zlabel('Componente 3', fontweight='bold')
    ax.view_init(elev=25, azim=-60)
    ax.legend(title="Dose N (kg/ha)", loc='upper left', fontsize=10)
    
    # Criar área para checkboxes
    rax = fig.add_axes([0.75, 0.3, 0.15, 0.35])
    labels = [f'{int(dose)} kg/ha' for dose in doses_uniques]
    visibility = [True] * len(doses_uniques)
    check = CheckButtons(rax, labels, [True] * len(labels))
    
    # Função para alternar visibilidade
    def toggle_visibility(label_text):
        idx = labels.index(label_text)
        dose = doses_uniques[idx]
        visibility[idx] = not visibility[idx]
        
        # Mudar opacidade
        new_alpha = 0.7 if visibility[idx] else 0.05
        scatter_artists[dose].set_alpha(new_alpha)
        
        # Centroide
        if dose in centroid_artists:
            centroid_artists[dose].set_alpha(1.0 if visibility[idx] else 0.1)
            text_artists[dose].set_alpha(1.0 if visibility[idx] else 0.1)
        
        fig.canvas.draw_idle()
    
    check.on_clicked(toggle_visibility)
    
    # Adicionar texto de instrução
    fig.text(0.75, 0.68, 'Toggle Classes:', fontsize=11, fontweight='bold')
    fig.text(0.5, 0.02, '← Marque/desmarque para ocultar/mostrar classes | Gire o gráfico com mouse →', 
             fontsize=10, ha='center', style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.savefig('kpca_clusters_3d.png', dpi=300, bbox_inches='tight')
    print("✓ Salvo: kpca_clusters_3d.png")
    print("  (Marque/desmarque os checkboxes para ocultar/mostrar classes)")
    print("  (Clique e arraste no gráfico para rotacionar)")
    plt.show()

def plotar_contribuicao_bandas_linear(pca_data, feature_names, top_n=15):
    """Plota as bandas mais importantes."""
    contribs, ordem = analisar_importancia_linear(pca_data, feature_names)
    top_indices = ordem[:top_n]
    wavelengths = [extrair_valor_onda(feature_names[i]) for i in top_indices]
    contributions = contribs[top_indices]
    
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.4, 0.9, top_n))
    bars = plt.barh(range(top_n), contributions[::-1], color=colors)
    plt.yticks(range(top_n), [f'{wl:.1f}nm' for wl in wavelengths[::-1]])
    plt.xlabel('Importância PCA (%)')
    plt.title(f'Top {top_n} Bandas Espectrais (Referência Linear)')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig('kpca_bandas_top.png', dpi=300)
    print("✓ Salvo: kpca_bandas_top.png")
    plt.show()

# ================= EXECUÇÃO PRINCIPAL =================

if __name__ == "__main__":
    print("="*70)
    print(" ANÁLISE KERNEL PCA 3D COM CENTROIDES")
    print("="*70)
    
    if not os.path.exists(ARQUIVO_DADOS):
        sys.exit(f"ERRO: {ARQUIVO_DADOS} não encontrado.")
    
    # 1. Carregar dados espectrais
    print("\n1. Carregando dados espectrais...")
    df_raw = pd.read_csv(ARQUIVO_DADOS, sep=';', decimal=',')
    print(f"   Amostras carregadas: {len(df_raw)}")
    
    # 2. Cruzar com Excel
    print("\n2. Cruzando com dados agronômicos...")
    if os.path.exists(ARQUIVO_AGRONOMICO):
        df = cruzar_com_excel(df_raw, ARQUIVO_AGRONOMICO)
    else:
        print(f"   ⚠️  Excel não encontrado")
        df = df_raw.copy()
        df['Dose_N'] = 0
    
    # 3. Filtrar por doses
    print(f"\n3. Filtrando para doses: {DOSES_ALVO}")
    if 'Dose_N' in df.columns:
        df_filt = filtrar_por_doses_especificas(df, 'Dose_N', DOSES_ALVO)
        print(f"   Amostras após filtro: {len(df_filt)}")
    else:
        df_filt = df.copy()
        df_filt['Dose_N'] = 0
    
    # 4. Selecionar bandas
    print(f"\n4. Selecionando bandas espectrais...")
    cols_derivada = [c for c in df_filt.columns if c.startswith('d1_Band_')]
    cols_selecionadas, bandas_encontradas = encontrar_colunas_por_bandas(
        cols_derivada, BANDAS_ESPECIFICAS, tolerancia=1.0
    )
    
    print(f"   Bandas selecionadas: {len(cols_selecionadas)}")
    print(f"   Valores: {sorted([f'{b:.0f}nm' for b in bandas_encontradas])}")
    
    if len(cols_selecionadas) < 3:
        sys.exit("ERRO: Menos de 3 bandas encontradas.")
    
    # 5. Preparar dados para análise
    print(f"\n5. Preparando dados...")
    X = df_filt[cols_selecionadas].apply(pd.to_numeric, errors='coerce').fillna(0)
    y = df_filt['Dose_N'].round(0)
    print(f"   Matriz X: {X.shape} (amostras × bandas)")
    
    # 6. Análise Kernel PCA
    print(f"\n6. Aplicando Kernel PCA ({KERNEL_TYPE})...")
    resultados = aplicar_analise_hibrida_3d(X)
    
    # 7. Métricas de qualidade
    print(f"\n7. Métricas de Qualidade:")
    if len(np.unique(y)) > 1:
        s_score_3d = silhouette_score(resultados['X_kpca'], y)
        print(f"   Silhouette Score (3D): {s_score_3d:.4f}")
    
    var_pc1 = resultados['explained_variance'][0] * 100
    var_pc2 = resultados['explained_variance'][1] * 100 if len(resultados['explained_variance']) > 1 else 0
    print(f"   Variância PC1: {var_pc1:.2f}%")
    print(f"   Variância PC2: {var_pc2:.2f}%")
    print(f"   Variância Total (PC1+PC2): {var_pc1 + var_pc2:.2f}%")
    
    # 8. Visualizações
    print(f"\n8. Gerando visualizações...")
    plotar_variancia_explicada_linear(resultados)
    plotar_clusters_kpca_3d(resultados, y)
    plotar_contribuicao_bandas_linear(resultados, list(X.columns), top_n=min(15, len(cols_selecionadas)))
    
    # 9. Relatório de bandas
    print(f"\n{'='*70}")
    print(f" TOP 10 BANDAS PARA DIFERENCIAÇÃO (Kernel PCA)")
    print(f"{'='*70}")
    
    contribs, ordem = analisar_importancia_linear(resultados, list(X.columns))
    print(f"{'Rank':<5} {'Comprimento (nm)':<20} {'Importância (%)':<15}")
    print("-" * 70)
    
    for rank, idx in enumerate(ordem[:10], 1):
        wl = extrair_valor_onda(list(X.columns)[idx])
        importance = contribs[idx]
        print(f"{rank:<5} {wl:<20.1f} {importance:.2f}%")
    
    print(f"\n{'='*70}\n✓ ANÁLISE KERNEL PCA CONCLUÍDA\n{'='*70}")