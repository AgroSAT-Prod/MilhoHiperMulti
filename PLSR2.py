import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Importações de Modelagem e Pré-processamento
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
from scipy.signal import savgol_filter

# ================= CONFIGURAÇÕES =================
PASTA_DO_SCRIPT = os.path.dirname(os.path.abspath(__file__))
# Ajuste o nome do arquivo se necessário
ARQUIVO_DADOS = os.path.join(PASTA_DO_SCRIPT, 'DATASET_IA_PROCESSADO.csv')

# --- CONFIGURAÇÃO DO MODELO ---
N_COMPONENTES_PLSR = 6  # Ajustado para datasets pequenos/médios

# --- INTERVALOS ---
# Deixe VAZIO [] para usar TODO o espectro (Recomendado para corrigir seu erro)
# O Nitrogênio afeta muito a região NIR (760nm a 900nm+), não corte isso!
INTERVALOS_DE_INTERESSE = [] 

# Configuração do Filtro de Suavização (Savitzky-Golay)
USAR_SUAVIZACAO = True
WINDOW_LENGTH = 11  # Janela de suavização (deve ser ímpar)
POLY_ORDER = 2      # Grau do polinômio

# ================= FUNÇÕES AUXILIARES =================

def extrair_valor_onda(nome_coluna):
    """Transforma 'd1_Band_720.5nm' ou 'Band_450nm' em float 720.5"""
    try:
        # Remove prefixos comuns e sulfixos
        limpo = nome_coluna.replace('d1_', '').replace('Band_', '').replace('nm', '')
        return float(limpo)
    except:
        return None

def preparar_dados_espectrais(df, intervalos):
    """
    Seleciona colunas, ORDENA por comprimento de onda (vital para física/suavização)
    e filtra os intervalos desejados.
    """
    # 1. Identificar todas as colunas de banda
    cols_todas = [c for c in df.columns if 'Band_' in c]
    
    # 2. Criar dicionário {coluna: comprimento_onda}
    mapa_ondas = {}
    for c in cols_todas:
        wl = extrair_valor_onda(c)
        if wl is not None:
            mapa_ondas[c] = wl
            
    # 3. Ordenar as colunas pelo comprimento de onda (Crescente)
    # Isso é CRUCIAL para o filtro de suavização funcionar
    cols_ordenadas = sorted(mapa_ondas.keys(), key=lambda x: mapa_ondas[x])
    
    # 4. Filtrar por intervalo (se houver)
    if not intervalos:
        return cols_ordenadas # Retorna tudo ordenado
    
    cols_finais = []
    print(f"Filtrando intervalos: {intervalos}")
    for col in cols_ordenadas:
        wl = mapa_ondas[col]
        for (inicio, fim) in intervalos:
            if inicio <= wl <= fim:
                cols_finais.append(col)
                break
                
    return cols_finais

# ================= FUNÇÕES DE AVALIAÇÃO E PLOTAGEM =================

def plotar_predicao(y_real, y_pred, titulo, modelo_nome):
    mse = mean_squared_error(y_real, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_real, y_pred)
    
    plt.figure(figsize=(7, 6))
    plt.scatter(y_real, y_pred, color='navy', alpha=0.6, edgecolors='white', s=70)
    
    # Linha 1:1
    min_val = min(y_real.min(), y_pred.min())
    max_val = max(y_real.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal (1:1)', linewidth=2)
    
    plt.title(f"{modelo_nome}\nR²: {r2:.3f} | RMSE: {rmse:.2f}", fontsize=12)
    plt.xlabel("Dose Real (N)")
    plt.ylabel("Dose Predita")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()
    
    return r2, rmse

def plotar_importancia_rf(modelo, colunas_X):
    """Plota quais bandas foram mais importantes para o Random Forest"""
    importancias = modelo.feature_importances_
    wls = [extrair_valor_onda(c) for c in colunas_X]
    
    plt.figure(figsize=(10, 5))
    plt.bar(wls, importancias, color='teal', width=5)
    plt.title("Importância das Bandas (Random Forest)")
    plt.xlabel("Comprimento de Onda (nm)")
    plt.ylabel("Importância Relativa")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# ================= EXECUÇÃO PRINCIPAL =================

if __name__ == "__main__":
    print("--- INICIANDO PROCESSAMENTO ---")
    
    # 1. Carregar
    if not os.path.exists(ARQUIVO_DADOS):
        sys.exit(f"ERRO: Arquivo {ARQUIVO_DADOS} não encontrado.")
    
    df = pd.read_csv(ARQUIVO_DADOS, sep=';', decimal=',')
    print(f"Dados carregados: {df.shape[0]} amostras.")

    # 2. Preparar X (Features Espectrais)
    cols_features = preparar_dados_espectrais(df, INTERVALOS_DE_INTERESSE)
    
    if len(cols_features) == 0:
        sys.exit("ERRO: Nenhuma coluna selecionada. Verifique os nomes das colunas ou intervalos.")
        
    print(f"Bandas selecionadas: {len(cols_features)} (De {extrair_valor_onda(cols_features[0])}nm até {extrair_valor_onda(cols_features[-1])}nm)")

    X = df[cols_features].apply(pd.to_numeric, errors='coerce').fillna(0)
    y_dose = pd.to_numeric(df['Dose_N'], errors='coerce').fillna(0)

    # 3. Aplicar Suavização (CRÍTICO PARA SEU DADO)
    if USAR_SUAVIZACAO:
        print(f"Aplicando filtro Savitzky-Golay (Janela={WINDOW_LENGTH}, Poly={POLY_ORDER})...")
        # axis=1 aplica linha por linha (espectro por espectro)
        X_smooth = savgol_filter(X, window_length=WINDOW_LENGTH, polyorder=POLY_ORDER, axis=1)
        X = pd.DataFrame(X_smooth, columns=cols_features)
    
    # Validação Cruzada Leave-One-Out
    loo = LeaveOneOut()

    # ---------------------------------------------------------
    # MODELO 1: Random Forest (Geralmente melhor para Agricultura)
    # ---------------------------------------------------------
    print("\n>>> Rodando RANDOM FOREST (Não-Linear)...")
    # n_estimators: número de árvores
    # max_depth: profundidade máxima (evita decorar demais)
    rf_model = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
    
    # Predição via CV
    y_pred_rf = cross_val_predict(rf_model, X, y_dose, cv=loo, n_jobs=-1)
    
    r2_rf, rmse_rf = plotar_predicao(y_dose, y_pred_rf, "Predição de Doses", "Random Forest Regressor")
    
    # Ajustar modelo final para ver importância das bandas
    rf_model.fit(X, y_dose)
    plotar_importancia_rf(rf_model, cols_features)

    # ---------------------------------------------------------
    # MODELO 2: PLSR (Clássico / Baseline)
    # ---------------------------------------------------------
    print(f"\n>>> Rodando PLSR ({N_COMPONENTES_PLSR} Componentes)...")
    pls_model = PLSRegression(n_components=N_COMPONENTES_PLSR, scale=True)
    
    y_pred_pls = cross_val_predict(pls_model, X, y_dose, cv=loo, n_jobs=-1)
    y_pred_pls = y_pred_pls.flatten()
    
    r2_pls, rmse_pls = plotar_predicao(y_dose, y_pred_pls, "Predição de Doses", "PLSR Clássico")

    # ---------------------------------------------------------
    # COMPARAÇÃO FINAL
    # ---------------------------------------------------------
    print(f"\n{'='*30}")
    print("RESUMO DOS RESULTADOS")
    print(f"{'='*30}")
    print(f"Random Forest -> R²: {r2_rf:.4f} | RMSE: {rmse_rf:.2f}")
    print(f"PLSR Clássico -> R²: {r2_pls:.4f} | RMSE: {rmse_pls:.2f}")
    
    if r2_rf > r2_pls:
        print("\nCONCLUSÃO: O Random Forest performou melhor (relação não-linear detectada).")
    else:
        print("\nCONCLUSÃO: O PLSR performou melhor (relação linear predominante).")
    
    print("\nDica: Se o R² ainda estiver baixo (< 0.6), revise se houve outliers na coleta de campo.")