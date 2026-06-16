import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import warnings

# Importações de Machine Learning
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score

# Ignorar avisos
warnings.filterwarnings("ignore")

# ================= CONFIGURAÇÕES =================
ARQUIVO_DADOS = r"C:\Users\muril\OneDrive\Documentos\Experiments-Prod\ProjetoMilho\FINALFINALFINAL.csv" 

# Índices vegetativos
INDICES_INTERESSE = [
    'NGRDI', 'MSAVI', 'MGRVI', 'RGBIndex', 'VARI', 'TGI', 'GLI', 
    'ExGRaw', 'ExG', 'ExRM', 'ExGR', 'NDVI', 'SAVI', 'RDVI', 
    'NLI', 'CVI', 'GNDVI', 'PanNDVI', 'NDRE', 'NDVIRE', 'SFDVI'
]

# ================= FUNÇÕES AUXILIARES =================

def avaliar_rf_loocv(X, y):
    """
    Treina Random Forest com validação cruzada Leave-One-Out
    """
    try:
        print(f" -> Treinando Random Forest (LOOCV)...")
        
        # Configuração do Random Forest
        # n_estimators: número de árvores
        # max_depth: profundidade máxima (None = cresce até o fim)
        # random_state: semente para reprodutibilidade
        model = RandomForestRegressor(n_estimators=100, 
                                      max_depth=None, 
                                      random_state=42, 
                                      n_jobs=-1)
        
        loo = LeaveOneOut()
        
        # Validação cruzada (Leave-One-Out)
        # Para cada amostra, o modelo treina nas outras n-1 e prevê ela.
        y_cv_pred = cross_val_predict(model, X, y, cv=loo, n_jobs=-1).flatten()
        
        # Métricas
        r2 = r2_score(y, y_cv_pred)
        rmse = np.sqrt(mean_squared_error(y, y_cv_pred))
        
        # Treina o modelo final com todos os dados para pegar a importância das features
        model.fit(X, y)
        
        return r2, rmse, model, y_cv_pred
    except Exception as e:
        print(f"Erro na regressão RF: {e}")
        return -np.inf, np.inf, None, None

# ================= EXECUÇÃO PRINCIPAL =================

if __name__ == "__main__":
    print("=== PREDIÇÃO DE BIOMASSA (RANDOM FOREST + LOOCV) ===\n")

    # 1. Carregar CSV
    if not os.path.exists(ARQUIVO_DADOS):
        print(f"ERRO: Arquivo '{ARQUIVO_DADOS}' não encontrado.")
        sys.exit(1)

    try:
        df = pd.read_csv(ARQUIVO_DADOS, sep=';', decimal=',')
        print(f"✓ Arquivo carregado: {len(df)} linhas.")
    except Exception as e:
        sys.exit(f"Erro ao ler CSV: {e}")

    # 2. Localizar Coluna de Biomassa
    colunas_biomassa = [c for c in df.columns if 'kg/ha' in str(c)]
    
    if len(colunas_biomassa) == 0:
        print("⚠ ERRO: Não encontrei a coluna alvo (kg/ha). Verifique o CSV.")
        sys.exit(1)
        
    NOME_COLUNA_ALVO = colunas_biomassa[0]
    print(f"-> Coluna Alvo detectada: '{NOME_COLUNA_ALVO}'")

    # 3. Filtragem e Limpeza
    colunas_presentes = [col for col in INDICES_INTERESSE if col in df.columns]
    
    if not colunas_presentes:
        print("ERRO: Nenhum índice espectral encontrado nas colunas do CSV.")
        sys.exit(1)

    df_work = df[colunas_presentes + [NOME_COLUNA_ALVO]].copy()
    df_clean = df_work.dropna().reset_index(drop=True).astype(float)

    print(f"Amostras válidas após limpeza: {len(df_clean)}")
    
    if len(df_clean) < 3:
        print("ERRO: Poucas amostras para realizar regressão.")
        sys.exit(1)

    X = df_clean[colunas_presentes]
    y = df_clean[NOME_COLUNA_ALVO].values
    
    print(f"Média da Biomassa: {y.mean():.2f} kg/ha")

    # 4. Regressão Random Forest
    r2, rmse, model, y_pred = avaliar_rf_loocv(X, y)

    # 5. Resultados e Gráficos
    print(f"\n{'='*60}")
    print(f"RESULTADO FINAL (RANDOM FOREST)")
    print(f"{'='*60}")
    print(f"R² (CV): {r2:.4f}")
    print(f"RMSE: {rmse:.2f} kg/ha")
    print(f"{'='*60}\n")
    
    if r2 > -10: # Plota mesmo se for negativo, desde que não seja catastrófico
        plt.figure(figsize=(12, 5))
        
        # Gráfico 1: Predito vs Real
        plt.subplot(1, 2, 1)
        plt.scatter(y, y_pred, color='royalblue', alpha=0.7, s=80, edgecolors='k')
        
        min_v, max_v = min(y.min(), y_pred.min()), max(y.max(), y_pred.max())
        plt.plot([min_v, max_v], [min_v, max_v], 'r--', linewidth=2, label='1:1')
        
        plt.xlabel('Observado (kg/ha)')
        plt.ylabel('Predito (kg/ha)')
        plt.title(f'Predição Random Forest\nR²={r2:.3f} | RMSE={rmse:.1f}')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Gráfico 2: Importância das Variáveis (Feature Importance)
        if model is not None:
            plt.subplot(1, 2, 2)
            
            # Extrair importâncias
            importances = model.feature_importances_
            feat_imp_df = pd.DataFrame({'Índice': X.columns, 'Importancia': importances})
            feat_imp_df = feat_imp_df.sort_values('Importancia', ascending=True)
            
            # Plot
            plt.barh(feat_imp_df['Índice'], feat_imp_df['Importancia'], color='forestgreen')
            plt.xlabel('Importância Relativa (0 a 1)')
            plt.title('Importância das Variáveis (Random Forest)')
        
        plt.tight_layout()
        plt.show()