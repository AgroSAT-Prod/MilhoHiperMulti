import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ================= CONFIGURAÇÕES =================
ARQUIVO_DADOS = r"C:\Users\muril\OneDrive\Documentos\Experiments-Prod\ProjetoMilho\FINALFINALFINAL.csv"  # Mesmo arquivo anterior

# Índices vegetativos (features)
INDICES_INTERESSE = [
    'NGRDI', 'MSAVI', 'MGRVI', 'RGBIndex', 'VARI', 'TGI', 'GLI', 
    'ExGRaw', 'ExG', 'ExRM', 'ExGR', 'NDVI', 'SAVI', 'RDVI', 
    'NLI', 'CVI', 'GNDVI', 'PanNDVI', 'NDRE', 'NDVIRE', 'SFDVI'
]

def biplot(score, coeff, labels=None, y_vals=None):
    """
    Gera um gráfico Biplot (Scores + Loadings)
    """
    plt.figure(figsize=(12, 10))
    xs = score[:, 0]
    ys = score[:, 1]
    n = coeff.shape[0]

    # Escalar os vetores para ficarem visíveis no gráfico junto com os pontos
    scalex = 1.0 / (xs.max() - xs.min())
    scaley = 1.0 / (ys.max() - ys.min())
    
    # Plot dos pontos (Scores)
    # A cor (c) varia conforme a biomassa (y_vals)
    scatter = plt.scatter(xs * scalex, ys * scaley, c=y_vals, cmap='viridis', s=100, edgecolors='k', alpha=0.8)
    plt.colorbar(scatter, label='Biomassa (kg/ha)')

    # Plot das setas (Loadings)
    # Usamos um fator de escala para as setas ficarem visíveis
    arrow_scale = 0.8 
    
    for i in range(n):
        plt.arrow(0, 0, coeff[i, 0] * arrow_scale, coeff[i, 1] * arrow_scale, 
                  color='r', alpha=0.5, head_width=0.02)
        
        # Texto das variáveis
        plt.text(coeff[i, 0] * arrow_scale * 1.15, coeff[i, 1] * arrow_scale * 1.15, 
                 labels[i], color='darkred', ha='center', va='center', fontsize=9, weight='bold')

    plt.xlabel(f"PC1")
    plt.ylabel(f"PC2")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.axhline(0, color='black', linewidth=0.8)
    plt.axvline(0, color='black', linewidth=0.8)
    plt.title("PCA Biplot: Índices Espectrais vs Biomassa")

if __name__ == "__main__":
    print("=== ANÁLISE DE COMPONENTES PRINCIPAIS (PCA) ===\n")

    # 1. Carregar Dados
    if not os.path.exists(ARQUIVO_DADOS):
        sys.exit(f"ERRO: Arquivo '{ARQUIVO_DADOS}' não encontrado.")

    try:
        df = pd.read_csv(ARQUIVO_DADOS, sep=';', decimal=',')
    except Exception as e:
        sys.exit(f"Erro leitura: {e}")

    # 2. Identificar Colunas
    colunas_biomassa = [c for c in df.columns if 'kg/ha' in str(c)]
    if not colunas_biomassa: sys.exit("Erro: Coluna kg/ha não encontrada.")
    col_alvo = colunas_biomassa[0]

    # Verifica colunas disponíveis
    feats = [c for c in INDICES_INTERESSE if c in df.columns]
    
    # Limpeza
    df_clean = df[feats + [col_alvo]].dropna().reset_index(drop=True).astype(float)
    
    X = df_clean[feats].values
    y = df_clean[col_alvo].values

    # 3. Padronização (StandardScaler) - CRUCIAL PARA PCA
    # Transforma os dados para Média=0 e Desvio Padrão=1
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # 4. Calcular PCA
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_std)

    # Variância Explicada
    var_exp = pca.explained_variance_ratio_
    print(f"Variância Explicada:")
    print(f"PC1: {var_exp[0]*100:.2f}% (Geralmente vigor geral)")
    print(f"PC2: {var_exp[1]*100:.2f}% (Variação secundária)")
    print(f"Total Acumulado (PC1+PC2): {(var_exp[0]+var_exp[1])*100:.2f}%")

    # 5. Visualização (Biplot)
    # Passamos os vetores de carga (pca.components_) transpostos
    biplot(X_pca, np.transpose(pca.components_[0:2, :]), labels=feats, y_vals=y)
    
    plt.tight_layout()
    plt.show()
    
    # 6. Análise de Correlação das Variáveis com PC1 (Loadings)
    print("\nInfluência no PC1 (Ordenada):")
    loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2', 'PC3'], index=feats)
    print(loadings['PC1'].abs().sort_values(ascending=False).head(10))