import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import math
from scipy.stats import linregress

# ================= CONFIGURAÇÕES =================

ARQUIVO_DADOS = r"C:\Users\muril\OneDrive\Documentos\Experiments-Prod\ProjetoMilho\FINAL12.csv"

INDICES_INTERESSE = [
    'NGRDI', 'MSAVI', 'MGRVI', 'RGBIndex', 'VARI', 'TGI', 'GLI', 
    'ExGRaw', 'ExG', 'ExRM', 'ExGR', 'NDVI', 'SAVI', 'RDVI', 
    'NLI', 'CVI', 'GNDVI', 'PanNDVI', 'NDRE', 'NDVIRE', 'SFDVI'
]

COLS_GRID = 4 
FIG_WIDTH = 20
FIG_HEIGHT_PER_ROW = 4

# ================= FUNÇÕES =================

def processar_e_plotar(caminho):
    if not os.path.exists(caminho):
        print(f"ERRO: Arquivo não encontrado: {caminho}")
        sys.exit(1)

    # 1. Carregamento permissivo (lê tudo, trata '-' como NaN)
    print("-> Carregando arquivo...")
    df = pd.read_csv(
        caminho, 
        sep=';', 
        decimal=',', 
        thousands='.', 
        na_values=['-', 'nan', 'NaN', ' ', '']
    )
    df.columns = df.columns.str.strip() # Remove espaços extras dos nomes
    
    # 2. Identificar Biomassa
    colunas_biomassa = [c for c in df.columns if 'kg/ha' in str(c)]
    if not colunas_biomassa:
        print("ERRO: Coluna '(kg/ha)' não encontrada.")
        sys.exit(1)
    col_alvo = colunas_biomassa[0]

    # 3. Limpeza da Biomassa (Manual para garantir)
    # Remove pontos de milhar e troca vírgula por ponto
    if df[col_alvo].dtype == 'object':
        df[col_alvo] = (
            df[col_alvo].astype(str)
            .str.replace('.', '', regex=False)
            .str.replace(',', '.', regex=False)
        )
    df[col_alvo] = pd.to_numeric(df[col_alvo], errors='coerce')
    
    # Remove apenas linhas onde a BIOMASSA é inválida (essencial para qualquer plot)
    df = df.dropna(subset=[col_alvo])
    df = df[df[col_alvo] > 0]
    
    print(f"-> Amostras com biomassa válida: {len(df)}")

    # 4. Preparar Features (Converter para numérico)
    cols_para_plotar = []
    for col in INDICES_INTERESSE:
        if col in df.columns:
            # Força conversão para float, transformando erros/strings em NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Só adiciona na lista se tiver pelo menos 1 valor válido
            if df[col].notna().sum() > 2:
                cols_para_plotar.append(col)
            else:
                print(f"   AVISO: Índice '{col}' vazio ou com poucos dados. Ignorado.")

    # 5. Gerar Gráficos (Limpeza Par a Par)
    num_plots = len(cols_para_plotar)
    if num_plots == 0:
        print("ERRO: Nenhum índice com dados suficientes para plotar.")
        sys.exit(1)
        
    num_rows = math.ceil(num_plots / COLS_GRID)
    fig, axes = plt.subplots(num_rows, COLS_GRID, figsize=(FIG_WIDTH, num_rows * FIG_HEIGHT_PER_ROW))
    axes = axes.flatten()

    lista_r2 = []

    for i, col_x in enumerate(cols_para_plotar):
        ax = axes[i]
        
        # Pega apenas as linhas onde ESTE índice e a Biomassa existem
        df_temp = df[[col_x, col_alvo]].dropna()
        
        X = df_temp[col_x]
        Y = df_temp[col_alvo]
        
        n_amostras = len(X)
        
        # Plot Scatter
        ax.scatter(X, Y, color='black', alpha=0.6, s=20)
        
        # Regressão (se houver pontos suficientes)
        if n_amostras > 2:
            slope, intercept, r_value, p_value, std_err = linregress(X, Y)
            r2 = r_value ** 2
            lista_r2.append({'Indice': col_x, 'R2': r2, 'N': n_amostras})
            
            x_line = np.linspace(X.min(), X.max(), 100)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, color='red', lw=2)
            
            # Texto
            texto = f"$R^2 = {r2:.3f}$\nN = {n_amostras}"
        else:
            texto = "Dados insuficientes"

        # Estilo
        ax.set_title(col_x, fontweight='bold')
        ax.set_xlabel(col_x)
        ax.set_ylabel("Biomassa (kg/ha)")
        
        ax.annotate(texto, xy=(0.05, 0.95), xycoords='axes fraction',
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))
        ax.grid(True, alpha=0.3)

    # Limpar eixos vazios
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(f"Correlação Bruta (Pairwise Deletion) - {len(df)} amostras totais", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    output_dir = os.path.dirname(ARQUIVO_DADOS)
    save_path = os.path.join(output_dir, 'correlacao_bruta_final.png')
    plt.savefig(save_path, dpi=150)
    print(f"\n-> Gráfico salvo em: {save_path}")
    plt.show()

    # Ranking
    if lista_r2:
        df_res = pd.DataFrame(lista_r2).sort_values('R2', ascending=False)
        print("\n=== RANKING R² (DADOS BRUTOS) ===")
        print(df_res.to_string(index=False))

if __name__ == "__main__":
    processar_e_plotar(ARQUIVO_DADOS)