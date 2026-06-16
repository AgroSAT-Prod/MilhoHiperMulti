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
ARQUIVO_DADOS = 'spectral_indices_rois.csv'
ARQUIVO_AGRONOMICO = os.path.join('Dataset', 'PlanilhaFiltrada.xlsx')
SEED = 53

# Níveis de Dose para Clusterização
DOSES_ALVO = [0, 90, 180, 360]

# Bandas específicas para análise (em nm)
BANDAS_ESPECIFICAS = [430, 450, 460, 500, 520, 550, 600, 625, 640, 660, 685, 720, 722, 740, 970, 990, 995]

# ================= FUNÇÕES AUXILIARES =================

def extrair_valor_onda(nome_coluna):
    """Extrai o valor numérico do comprimento de onda do nome da coluna."""
    try:
        limpo = nome_coluna.replace('d1_', '').replace('Band_', '').replace('nm', '')
        return float(limpo)
    except:
        return None

def cruzar_com_excel(df_espectral, arquivo_excel):
    """
    Cruza os dados espectrais com os dados agronômicos do Excel.
    Retorna dataframe unificado com as colunas de dose e clorofila.
    """
    try:
        # Lê a partir da linha 3 (header=2)
        df_agro = pd.read_excel(arquivo_excel, header=2)
        df_agro.columns = [str(c).strip().lower() for c in df_agro.columns]
        
        print("\n✓ Colunas encontradas no Excel:")
        print(f"  {df_agro.columns.tolist()}\n")
        
    except Exception as e:
        print(f"⚠️  Não foi possível ler o Excel: {e}")
        print("  Usando dose padrão 0 para todas as amostras.\n")
        df_espectral['Dose_N'] = 0
        return df_espectral
    
    # Mapear colunas
    col_id = next((c for c in df_agro.columns if 'id' in c and 'parcela' in c), 'id parcela')
    col_dose = next((c for c in df_agro.columns if 'dose' in c), None)
    col_clor_m = next((c for c in df_agro.columns if 'médio' in c and 'clorofila' in c), None)
    col_clor_s = next((c for c in df_agro.columns if 'superior' in c and 'clorofila' in c), None)
    
    print(f"Mapeamento de colunas encontrado:")
    print(f"  ID: '{col_id}'")
    print(f"  Dose: '{col_dose}'")
    print(f"  Clorofila Médio: '{col_clor_m}'")
    print(f"  Clorofila Superior: '{col_clor_s}'")
    
    # Cruzar dados
    df_espectral['ID_Numeric'] = df_espectral['ID_Amostra'].astype(int)
    df_merge = df_espectral.merge(
        df_agro[[col_id, col_dose, col_clor_m, col_clor_s]],
        left_on='ID_Numeric',
        right_on=col_id,
        how='left'
    )
    
    # Selecionar clorofila baseado na parte (M ou S)
    df_merge['Y_Clorofila'] = np.where(
        df_merge['Parte'] == 'M',
        df_merge[col_clor_m],
        df_merge[col_clor_s]
    )
    
    # Renomear coluna de dose
    df_merge.rename(columns={col_dose: 'Dose_N'}, inplace=True)
    
    return df_merge[list(df_espectral.columns) + ['Dose_N', 'Y_Clorofila']]

def filtrar_por_doses_especificas(df, coluna_dose, doses):
    """Filtra o dataframe apenas para as doses de interesse."""
    df_temp = df.copy()
    df_temp[coluna_dose] = pd.to_numeric(df_temp[coluna_dose], errors='coerce')
    mask = df_temp[coluna_dose].round(0).isin(doses)
    return df_temp[mask].copy()

def encontrar_colunas_por_bandas(colunas, bandas_alvo, tolerancia=1.0):
    """Encontra as colunas que correspondem às bandas específicas."""
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
    
    loadings = pca.components_[:n_componentes, :].T
    feature_names = X.columns.tolist()
    
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
    
    ax1.bar(range(1, min(len(pca_result['pca_result']['explained_variance']) + 1, 21)), 
            pca_result['pca_result']['explained_variance'][:20] * 100)
    ax1.set_xlabel('Componente Principal')
    ax1.set_ylabel('Variância Explicada (%)')
    ax1.set_title('Variância Explicada por Componente')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(range(1, len(cumsum) + 1), cumsum * 100, 'bo-', linewidth=2, markersize=8)
    ax2.axhline(y=95, color='r', linestyle='--', label='95% variância')
    ax2.set_xlabel('Número de Componentes')
    ax2.set_ylabel('Variância Acumulada (%)')
    ax2.set_title('Variância Explicada Acumulada')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pca_variancia_explicada.png', dpi=300)
    print("✓ Salvo: pca_variancia_explicada.png")
    plt.show()

def plotar_clusters_pca(pca_result, y_doses):
    """Visualiza clusters nos 2 primeiros componentes principais."""
    from scipy.spatial import ConvexHull
    from matplotlib.patches import Polygon
    
    X_pca = pca_result['pca_result']['X_pca']
    
    plt.figure(figsize=(14, 9))
    
    cores_dose = {0: '#440154', 90: '#31688e', 180: '#35b779', 360: '#fde724'}
    doses_uniques = sorted(y_doses.unique())
    
    # Áreas de domínio (convex hull)
    for dose in doses_uniques:
        mask = y_doses == dose
        X_dose = X_pca[mask, :2]
        
        if len(X_dose) >= 3:
            try:
                hull = ConvexHull(X_dose)
                hull_points = X_dose[hull.vertices]
                polygon = Polygon(hull_points, alpha=0.15, color=cores_dose.get(dose, '#808080'), 
                                edgecolor=cores_dose.get(dose, '#808080'), linewidth=2)
                plt.gca().add_patch(polygon)
            except:
                pass
    
    # Plotar pontos
    for dose in doses_uniques:
        mask = y_doses == dose
        plt.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            label=f'{int(dose)} kg/ha',
            s=100, alpha=0.8, edgecolor='white', linewidth=0.5,
            color=cores_dose.get(dose, '#808080')
        )
    
    plt.title('Clusters de Doses de Nitrogênio (PC1 vs PC2)', fontsize=14, fontweight='bold')
    plt.xlabel(f'PC1 ({pca_result["pca_result"]["explained_variance"][0]*100:.1f}%)')
    plt.ylabel(f'PC2 ({pca_result["pca_result"]["explained_variance"][1]*100:.1f}%)')
    plt.legend(title="Dose N (kg/ha)", loc='best', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('pca_clusters_doses.png', dpi=300)
    print("✓ Salvo: pca_clusters_doses.png")
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
    print("✓ Salvo: pca_contribuicao_bandas.png")
    plt.show()

# ================= EXECUÇÃO PRINCIPAL =================

if __name__ == "__main__":
    if not os.path.exists(ARQUIVO_DADOS):
        sys.exit(f"ERRO: Arquivo {ARQUIVO_DADOS} não encontrado.")
    
    print("="*70)
    print(" ANÁLISE PCA DE DADOS ESPECTRAIS")
    print("="*70)
    
    # 1. Carregamento
    df_raw = pd.read_csv(ARQUIVO_DADOS, sep=';', decimal=',')
    print(f"\n1. Dataset carregado: {len(df_raw)} amostras")
    print(f"   Colunas: {df_raw.columns.tolist()[:5]}... (+{len(df_raw.columns)-5} mais)")
    
    # 2. Cruzar com Excel (adicionar Dose_N e Y_Clorofila)
    print("\n2. Cruzando com dados agronômicos...")
    if os.path.exists(ARQUIVO_AGRONOMICO):
        df = cruzar_com_excel(df_raw, ARQUIVO_AGRONOMICO)
    else:
        print(f"   ⚠️  Excel não encontrado em {ARQUIVO_AGRONOMICO}")
        df = df_raw.copy()
        df['Dose_N'] = 0
    
    # 3. Filtrar por doses
    print(f"\n3. Filtrando para doses: {DOSES_ALVO}")
    if 'Dose_N' in df.columns:
        df_filt = filtrar_por_doses_especificas(df, 'Dose_N', DOSES_ALVO)
        print(f"   Amostras após filtro: {len(df_filt)}")
    else:
        df_filt = df.copy()
        print("   ⚠️  Coluna 'Dose_N' não encontrada, usando todas as amostras")
    
    # 4. Encontrar colunas de bandas
    print(f"\n4. Localizando bandas espectrais...")
    cols_derivada = [c for c in df_filt.columns if c.startswith('d1_Band_')]
    cols_selecionadas, bandas_encontradas, bandas_nao_encontradas = encontrar_colunas_por_bandas(
        cols_derivada, BANDAS_ESPECIFICAS, tolerancia=1.0
    )
    
    if bandas_nao_encontradas:
        print(f"   ⚠️  Bandas não encontradas: {bandas_nao_encontradas}")
    
    print(f"   ✓ {len(cols_selecionadas)} bandas selecionadas")
    print(f"     {sorted([f'{b:.0f}nm' for b in bandas_encontradas])}")
    
    if len(cols_selecionadas) < 3:
        sys.exit("ERRO: Menos de 3 bandas encontradas. Verifique os nomes das colunas.")
    
    # 5. Preparar dados para PCA
    print(f"\n5. Preparando dados para PCA...")
    X = df_filt[cols_selecionadas].apply(pd.to_numeric, errors='coerce').fillna(0)
    
    if 'Dose_N' in df_filt.columns:
        y = df_filt['Dose_N'].round(0)
    else:
        y = np.zeros(len(X))
    
    print(f"   Matriz X: {X.shape} (amostras x bandas)")
    
    # 6. Análise PCA
    print(f"\n6. Calculando PCA...")
    pca_result = selecionar_bandas_por_pca(X, variancia_explicada=0.90)
    
    # 7. Qualidade da clusterização
    if len(np.unique(y)) > 1:
        s_score = silhouette_score(pca_result['pca_result']['X_pca'][:, :2], y)
        print(f"   Silhouette Score: {s_score:.4f}")
    
    var_pc1 = pca_result['pca_result']['explained_variance'][0] * 100
    var_pc2 = pca_result['pca_result']['explained_variance'][1] * 100
    print(f"   Variância PC1: {var_pc1:.2f}%")
    print(f"   Variância PC2: {var_pc2:.2f}%")
    print(f"   Variância Total: {var_pc1 + var_pc2:.2f}%")
    
    # 8. Visualizações
    print(f"\n7. Gerando visualizações...")
    plotar_variancia_explicada(pca_result)
    plotar_clusters_pca(pca_result, y)
    plotar_contribuicao_bandas(pca_result, top_n=min(15, len(cols_selecionadas)))
    
    # 9. Relatório final
    print(f"\n{'='*70}")
    print(f" TOP 10 BANDAS PARA DIFERENCIAR DOSES")
    print(f"{'='*70}")
    
    feature_order = pca_result['feature_order']
    feature_names = pca_result['feature_names']
    feature_contribs = pca_result['feature_contributions']
    
    print(f"{'Rank':<5} {'Comprimento (nm)':<20} {'Importância (%)':<15}")
    print("-" * 70)
    
    for rank, idx in enumerate(feature_order[:10], 1):
        wl = extrair_valor_onda(feature_names[idx])
        importance = feature_contribs[idx]
        print(f"{rank:<5} {wl:<20.1f} {importance:.2f}%")
    
    print(f"\n{'='*70}\n✓ ANÁLISE CONCLUÍDA\n{'='*70}")