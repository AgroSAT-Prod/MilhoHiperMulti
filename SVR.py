import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Importações para SVR e Processamento
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix

# ================= CONFIGURAÇÕES =================
PASTA_DO_SCRIPT = os.path.dirname(os.path.abspath(__file__))
ARQUIVO_DADOS = os.path.join(PASTA_DO_SCRIPT, 'DATASET_IA_PROCESSADO.csv')

SEED = 42

# --- INTERVALOS DE INTERESSE ---
INTERVALOS_DE_INTERESSE = [
    #(520, 580),  # Pico do Verde
    #(690, 760)   # Red Edge
] 
# Para usar tudo: INTERVALOS_DE_INTERESSE = []

# ================= FUNÇÕES AUXILIARES =================

def extrair_valor_onda(nome_coluna):
    """Transforma 'd1_Band_720.5nm' em float 720.5"""
    try:
        limpo = nome_coluna.replace('d1_', '').replace('Band_', '').replace('nm', '')
        return float(limpo)
    except:
        return None

def filtrar_bandas(colunas, intervalos):
    if not intervalos:
        return colunas
    
    cols_selecionadas = []
    print(f"\nFiltrando bandas nos intervalos: {intervalos}")
    for col in colunas:
        wl = extrair_valor_onda(col)
        if wl is not None:
            for (inicio, fim) in intervalos:
                if inicio <= wl <= fim:
                    cols_selecionadas.append(col)
                    break
    return cols_selecionadas

# ================= FUNÇÕES DE AVALIAÇÃO (REGRESSÃO) =================

def avaliar_regressao_svr(y_real, y_pred, titulo):
    """Avalia o modelo SVR (R² e RMSE) e plota Predito vs Medido."""
    mse = mean_squared_error(y_real, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_real, y_pred)
    
    print(f"\n{'='*20} {titulo} {'='*20}")
    print(f"RMSE (Erro Médio): {rmse:.4f}")
    print(f"R² (Correlação):   {r2:.4f}")
    
    # Gráfico Predito vs Medido
    plt.figure(figsize=(8, 8))
    plt.scatter(y_real, y_pred, color='#800080', alpha=0.6, edgecolors='k') # Roxo para SVR
    
    # Linha 1:1 (Perfeição)
    min_val = min(y_real.min(), y_pred.min())
    max_val = max(y_real.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 (Perfeito)')
    
    plt.title(f"{titulo}\nR²: {r2:.3f} | RMSE: {rmse:.2f}")
    plt.xlabel("Valor Real (Dose N)")
    plt.ylabel("Valor Predito (SVR)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plotar_coeficientes_svr(modelo_treinado, colunas_X, titulo):
    """
    Mostra os pesos (coeficientes) do SVR Linear.
    """
    # Extrai o SVR de dentro do pipeline
    svr = modelo_treinado.named_steps['svr']
    
    if svr.kernel != 'linear':
        print("Aviso: Gráfico de importância só disponível para SVR com kernel linear.")
        return

    # Coeficientes do hiperplano
    coefs = svr.coef_.flatten()
    
    # Cria eixo X numérico
    wls = [extrair_valor_onda(c) for c in colunas_X]

    plt.figure(figsize=(12, 6))
    plt.plot(wls, coefs, color='black', linewidth=1.5)
    plt.fill_between(wls, coefs, 0, where=(coefs > 0), color='green', alpha=0.3, label='Positivo (Aumenta Dose)')
    plt.fill_between(wls, coefs, 0, where=(coefs < 0), color='red', alpha=0.3, label='Negativo (Diminui Dose)')
    
    plt.axhline(0, color='gray', linestyle='--')
    plt.title(f"Coeficientes de Regressão SVR - {titulo}")
    plt.xlabel("Comprimento de Onda (nm)")
    plt.ylabel("Peso do Coeficiente (Beta)")
    plt.legend()
    
    # Marcação dos intervalos
    for (inicio, fim) in INTERVALOS_DE_INTERESSE:
        plt.axvspan(inicio, fim, color='orange', alpha=0.1)
        
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plotar_assinatura_media(dataframe, colunas_bandas, coluna_alvo, titulo):
    plt.figure(figsize=(12, 6))
    wls = [extrair_valor_onda(c) for c in colunas_bandas]
    grupos = dataframe.groupby(coluna_alvo)[colunas_bandas].mean()
    
    if len(grupos) == 2:
        cores = ['red', 'green']
    else:
        cores = plt.cm.viridis(np.linspace(0, 1, len(grupos)))

    for i, (classe, linha_media) in enumerate(grupos.iterrows()):
        plt.plot(wls, linha_media, label=f"{coluna_alvo}: {classe}", color=cores[i], linewidth=2)

    plt.title(f"Assinatura Espectral Média - {titulo}", fontsize=14)
    plt.xlabel("Comprimento de Onda (nm)")
    plt.ylabel("Intensidade")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    for (inicio, fim) in INTERVALOS_DE_INTERESSE:
        plt.axvspan(inicio, fim, color='orange', alpha=0.1)

    plt.tight_layout()
    plt.show()

# ================= EXECUÇÃO PRINCIPAL =================

if __name__ == "__main__":
    # 1. Carregar Dados
    if not os.path.exists(ARQUIVO_DADOS):
        sys.exit("ERRO: Arquivo processado não encontrado.")
        
    df = pd.read_csv(ARQUIVO_DADOS, sep=';', decimal=',')
    print(f"Dataset carregado: {len(df)} amostras.")

    # 2. Preparar Features (X)
    cols_todas = [c for c in df.columns if c.startswith('d1_Band_') or c.startswith('Band_')]
    cols_features = filtrar_bandas(cols_todas, INTERVALOS_DE_INTERESSE)
    
    if len(cols_features) == 0:
        sys.exit("ERRO: Nenhuma banda nos intervalos selecionados.")

    X = df[cols_features]
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # Define Validação Leave-One-Out
    loo = LeaveOneOut()

    # ---------------------------------------------------------
    # MODELO 1: Regressão de DOSES (SVR)
    # ---------------------------------------------------------
    # Converter dose para numérico
    y_dose = pd.to_numeric(df['Dose_N'], errors='coerce').fillna(0)
    
    print(f"\nRodando SVR (Linear) com Leave-One-Out...")
    
    # Pipeline: Normalização -> SVR Linear
    # C=1.0 e epsilon=0.1 são padrões bons para começar
    pipeline_svr = make_pipeline(StandardScaler(), SVR(kernel='linear', C=1.0, epsilon=0.1))
    
    # LOOCV
    y_pred_dose_loo = cross_val_predict(pipeline_svr, X, y_dose, cv=loo, n_jobs=-1)
    
    # Avaliação
    avaliar_regressao_svr(y_dose, y_pred_dose_loo, "SVR - Predição de Doses de Nitrogênio")
    
    # Coeficientes
    pipeline_svr.fit(X, y_dose)
    plotar_coeficientes_svr(pipeline_svr, cols_features, "Doses de N")
    
    plotar_assinatura_media(df, cols_features, 'Dose_N', "Médias por Dose")

    # ---------------------------------------------------------
    # MODELO 2: Classificação BINÁRIA (via Regressão SVR)
    # ---------------------------------------------------------
    print("\nRodando SVR para Binário (Com N vs Sem N)...")
    
    # Transforma classes em números: 0 (Sem N) e 1 (Com N)
    y_binario_num = np.where(y_dose > 0, 1, 0)
    y_binario_label = np.where(y_dose > 0, "Com N", "Sem N")
    
    pipeline_bin = make_pipeline(StandardScaler(), SVR(kernel='linear', C=1.0))
    
    # O SVR vai prever números como 0.8, 0.2, -0.1, etc.
    y_pred_bin_raw = cross_val_predict(pipeline_bin, X, y_binario_num, cv=loo, n_jobs=-1)
    
    # Thresholding: Se > 0.5 considera "Com N" (1), senão "Sem N" (0)
    y_pred_bin_class = np.where(y_pred_bin_raw > 0.5, "Com N", "Sem N")
    
    # Avaliação como Classificação
    print(f"\n{'='*20} SVR Binário (Threshold 0.5) {'='*20}")
    acc = accuracy_score(y_binario_label, y_pred_bin_class)
    print(f"Acurácia Global: {acc:.2%}")
    
    cm = confusion_matrix(y_binario_label, y_pred_bin_class, labels=["Sem N", "Com N"])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Sem N", "Com N"], yticklabels=["Sem N", "Com N"])
    plt.title(f"Matriz de Confusão SVR (Convertido)\nAcurácia: {acc:.2%}")
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.show()
    
    # Coeficientes Binários
    pipeline_bin.fit(X, y_binario_num)
    plotar_coeficientes_svr(pipeline_bin, cols_features, "Binário (Discriminante)")
    
    print("\nProcesso concluído!")