import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import warnings

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ================= CONFIGURAÇÕES =================

ARQUIVO_DADOS = "Biomassa12_12.csv"
NOME_ALVO_DESEJADO = 'Kilos'

# Variáveis de Entrada (Preditores)
PREDITORES_FIXOS = [
    'NGRDI', 'MSAVI', 'MGRVI', 'RGBIndex', 'VARI', 'TGI', 'GLI',
    'ExGRaw', 'ExG', 'ExRM', 'ExGR', 'NDVI', 'SAVI', 'RDVI',
    'NLI', 'CVI', 'GNDVI', 'PanNDVI', 'NDRE', 'NDVIRE', 'SFDVI'
]

# ================= FUNÇÕES =================

def detectar_separador(caminho):
    """Detecta automaticamente o separador do CSV."""
    with open(caminho, 'r', encoding='utf-8-sig') as f:
        linha = f.readline()
    for sep in [';', ',', '\t', '|']:
        if sep in linha:
            return sep
    return ','

# ================= EXECUÇÃO PRINCIPAL (BASELINE) =================

if __name__ == "__main__":
    print(f"=== MODELO BASELINE RFR: '{NOME_ALVO_DESEJADO}' ===\n")
    
    # 1. Leitura dos Dados
    if not os.path.exists(ARQUIVO_DADOS): 
        sys.exit(f"ERRO: Arquivo '{ARQUIVO_DADOS}' não encontrado.")
        
    sep = detectar_separador(ARQUIVO_DADOS)
    
    try:
        df = pd.read_csv(ARQUIVO_DADOS, sep=sep, decimal=',', thousands='.', 
                         na_values=['-', ' -', '- ', 'nan', 'NaN'], encoding='utf-8-sig')
        df.columns = df.columns.str.strip()
    except Exception as e: 
        sys.exit(f"Erro ao ler CSV: {e}")

    # 2. Preparação da Variável Alvo
    cols_alvo = [c for c in df.columns if NOME_ALVO_DESEJADO.lower() in c.lower()]
    if not cols_alvo: sys.exit("ERRO: Alvo não encontrado.")
    ALVO = cols_alvo[0]
    
    df = df.dropna(subset=[ALVO])
    if df[ALVO].dtype == 'object': 
        df[ALVO] = pd.to_numeric(df[ALVO].astype(str).str.replace('.', '').str.replace(',', '.'), errors='coerce')
    df = df.dropna(subset=[ALVO])
    df = df[df[ALVO] > 0.0001].reset_index(drop=True)

    # 3. Preparação das Variáveis de Entrada
    cols_candidatas = [c for c in PREDITORES_FIXOS if c in df.columns]
    for c in cols_candidatas:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        
    df_full = df[cols_candidatas + [ALVO]].copy()
    
    # Remove colunas que são 100% nulas antes de preencher com a média
    colunas_validas = df_full.columns[df_full.notna().any()].tolist()
    df_full = df_full[colunas_validas]
    
    # Atualiza a lista de preditores apenas com as colunas que sobreviveram
    cols_candidatas = [c for c in cols_candidatas if c in colunas_validas]
    
    # Preenche NaNs com a média da coluna
    df_full[cols_candidatas] = df_full[cols_candidatas].fillna(df_full[cols_candidatas].mean())
    df_full = df_full.dropna().reset_index(drop=True)
    
    if len(df_full) < 3: 
        print("AVISO CRÍTICO: Menos de 3 amostras restaram. A execução falhará no Cross-Validation.")
        sys.exit()

    X = df_full[cols_candidatas]
    y = df_full[ALVO].values

    print(f"-> Variáveis de entrada válidas: {len(cols_candidatas)} índices detectados.")
    print(f"-> Amostras válidas para treino: {len(X)}")

    # 4. Inicializar modelo RFR
    # Random Forest não exige normalização de escala nos dados
    modelo_rfr = RandomForestRegressor(random_state=42)
    
    print("-> Executando validação cruzada (Leave-One-Out) com RFR...")
    # Predição com validação cruzada rigorosa (Leave-One-Out)
    y_pred_cv = cross_val_predict(modelo_rfr, X, y, cv=LeaveOneOut(), n_jobs=-1).ravel()

    # 5. Avaliação do Modelo
    r2 = r2_score(y, y_pred_cv)
    rmse = np.sqrt(mean_squared_error(y, y_pred_cv))

    print(f"\n=== RESULTADO BASELINE RFR ===")
    print(f" R²   = {r2:.4f}")
    print(f" RMSE = {rmse:.2f}")

    # 6. Gráficos e Exportação
    plt.figure(figsize=(8, 6))
    
    # Mudança de cor para diferenciar do SVR anterior (agora laranja)
    plt.scatter(y, y_pred_cv, c='#e67e22', edgecolors='k', s=60, alpha=0.7, label='Predição CV (LOO)')
    
    # Linha de tendência ideal (1:1)
    min_v = min(y.min(), y_pred_cv.min()) * 0.9
    max_v = max(y.max(), y_pred_cv.max()) * 1.1
    plt.plot([min_v, max_v], [min_v, max_v], 'r--', lw=2, label='Linha Ideal')
    
    plt.xlabel('Observado (Reais)')
    plt.ylabel('Predito (Cross-Validation)')
    plt.title(f'Baseline RFR (Random Forest)\nR² = {r2:.3f} | RMSE = {rmse:.2f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig('resultado_baseline_rfr.png', dpi=150)
    print("\n-> Gráfico Salvo: resultado_baseline_rfr.png")
    
    df_full['Predito_CV'] = y_pred_cv
    df_full['Residuo'] = y - y_pred_cv
    df_full.to_csv('resultado_baseline_rfr.csv', sep=';', decimal=',', index=False)
    print("-> Resultados Salvos: resultado_baseline_rfr.csv")
    
    plt.show()