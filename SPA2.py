import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Importações para Machine Learning e SPA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ================= CONFIGURAÇÕES =================
PASTA_DO_SCRIPT = os.path.dirname(os.path.abspath(__file__))
ARQUIVO_DADOS = os.path.join(PASTA_DO_SCRIPT, 'DATASET_IA_PROCESSADO.csv')

SEED = 42

# 1. Definição das bandas ruidosas (vistas no Heatmap) para remoção
BANDAS_PARA_REMOVER = [558.50, 719.40] 
TOLERANCIA_REMOCAO = 1.0 # Margem de erro para encontrar a banda (ex: 558.5 +/- 1)

# 2. Configurações do SPA (Seleção de Variáveis)
SPA_MAX_VARIAVEIS = 12   # O SPA vai tentar achar até 12 bandas ótimas
SPA_TEST_SIZE = 0.3      # Divisão interna do SPA

# 3. Intervalos de interesse (Focando onde a planta responde)
INTERVALOS_DE_INTERESSE = [
    (520, 580),   # Pico do Verde
    (690, 760)    # Red Edge (Onde o N mais aparece)
] 

# ================= CLASSE SPA (ALGORITMO DE SELEÇÃO) =================

class SPA_Selector:
    def __init__(self, max_vars=20, test_size=0.3):
        self.max_vars = max_vars
        self.test_size = test_size
        self.selected_cols = []
        
    def fit(self, X, y):
        print(f" > [SPA] Iniciando seleção (Buscando as {self.max_vars} bandas mais distintas)...")
        
        # Split interno para validação
        X_cal, X_val, y_cal, y_val = train_test_split(X, y, test_size=self.test_size, random_state=SEED)
        
        X_cal_mat = np.array(X_cal)
        X_val_mat = np.array(X_val)
        y_cal_mat = np.array(y_cal)
        y_val_mat = np.array(y_val)
        
        n_cols = X_cal_mat.shape[1]
        feature_names = X.columns.tolist()
        k_max = min(self.max_vars, n_cols - 1)
        
        best_chain = None
        min_rmse_global = float('inf')
        
        # --- FASE 1: PROJEÇÕES ---
        chains = {} 
        
        # Otimização: Se houver muitas bandas, testamos todas. É rápido para <500 colunas.
        for k0 in range(n_cols):
            x_projected = X_cal_mat.copy()
            selected_indices = [k0]
            
            for k in range(1, k_max):
                last_idx = selected_indices[-1]
                v_last = x_projected[:, last_idx].reshape(-1, 1)
                
                norm_sq = np.dot(v_last.T, v_last)
                if norm_sq < 1e-10: break 
                
                # Projeção Ortogonal (Remove a informação que já temos)
                proj_factor = np.dot(v_last.T, x_projected) / norm_sq
                x_projected = x_projected - np.dot(v_last, proj_factor)
                
                norms = np.sum(x_projected**2, axis=0)
                norms[selected_indices] = -1 
                next_idx = np.argmax(norms)
                selected_indices.append(next_idx)
            
            chains[k0] = selected_indices

        # --- FASE 2: AVALIAÇÃO VIA MLR ---
        lr = LinearRegression()
        
        for k0, indices_chain in chains.items():
            # Testa subconjuntos crescentes da cadeia
            for n_vars in range(1, k_max + 1):
                subset = indices_chain[:n_vars]
                lr.fit(X_cal_mat[:, subset], y_cal_mat)
                y_pred = lr.predict(X_val_mat[:, subset])
                rmse = np.sqrt(np.mean((y_val_mat - y_pred)**2))
                
                if rmse < min_rmse_global:
                    min_rmse_global = rmse
                    best_chain = subset
        
        self.selected_cols = [feature_names[i] for i in best_chain]
        return self.selected_cols

# ================= FUNÇÕES AUXILIARES =================

def extrair_valor_onda(nome_coluna):
    """Lê 'd1_Band_720.5nm' e retorna 720.5"""
    try:
        limpo = nome_coluna.replace('d1_', '').replace('Band_', '').replace('nm', '')
        return float(limpo)
    except:
        return None

def filtrar_e_limpar_bandas(colunas, intervalos, bandas_ruins):
    """
    1. Filtra pelo intervalo (Verde/RedEdge).
    2. Remove as bandas ruins (ruído).
    """
    cols_boas = []
    print(f"\nFiltrando bandas e removendo ruídos...")
    
    for col in colunas:
        wl = extrair_valor_onda(col)
        if wl is not None:
            # Checa se é banda ruim
            e_ruim = any(abs(wl - ruim) < TOLERANCIA_REMOCAO for ruim in bandas_ruins)
            if e_ruim:
                # print(f"   X Removendo banda ruidosa: {col}")
                continue
                
            # Checa intervalo
            for (inicio, fim) in intervalos:
                if inicio <= wl <= fim:
                    cols_boas.append(col)
                    break
    return cols_boas

def plotar_espectro_selecionado(dataframe, todas_cols, cols_spa, coluna_alvo, titulo):
    # Plota a média da derivada (Savitzky-Golay) e as linhas escolhidas pelo SPA
    todas_ordenadas = sorted(todas_cols, key=lambda x: extrair_valor_onda(x))
    wls_full = [extrair_valor_onda(c) for c in todas_ordenadas]
    wls_spa = [extrair_valor_onda(c) for c in cols_spa]
    
    grupos = dataframe.groupby(coluna_alvo)[todas_ordenadas].mean()
    
    plt.figure(figsize=(12, 6))
    cores = plt.cm.viridis(np.linspace(0, 1, len(grupos)))

    # Plota as curvas de derivada (Background)
    for i, (classe, linha_media) in enumerate(grupos.iterrows()):
        plt.plot(wls_full, linha_media, label=f"{classe}", color=cores[i], linewidth=2)

    # Plota as linhas verticais do SPA
    for j, wl in enumerate(wls_spa):
        lbl = 'Bandas SPA' if j == 0 else ""
        plt.axvline(x=wl, color='red', linestyle='--', alpha=0.7, label=lbl)

    plt.title(f"Bandas Selecionadas pelo SPA (Sobre Derivada) - {titulo}", fontsize=14)
    plt.xlabel("Comprimento de Onda (nm)")
    plt.ylabel("1ª Derivada (Inclinação)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def avaliar_modelo(y_true, y_pred, titulo, labels_nomes=None):
    acc = accuracy_score(y_true, y_pred)
    print(f"\n[{titulo}] Acurácia (LOOCV): {acc:.2%}")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels_nomes if labels_nomes else "auto",
                yticklabels=labels_nomes if labels_nomes else "auto")
    plt.title(f"Matriz de Confusão: {titulo}")
    plt.tight_layout()
    plt.show()

def plotar_importancia(modelo, colunas_X, titulo):
    importancias = modelo.feature_importances_
    indices = np.argsort(importancias)[::-1]
    
    plt.figure(figsize=(10, 5))
    plt.title(f"Feature Importance Random Forest - {titulo}")
    plt.bar(range(len(colunas_X)), importancias[indices], align="center", color='#2ca02c')
    plt.xticks(range(len(colunas_X)), [colunas_X[i] for i in indices], rotation=45)
    plt.tight_layout()
    plt.show()

# ================= EXECUÇÃO PRINCIPAL =================

if __name__ == "__main__":
    if not os.path.exists(ARQUIVO_DADOS):
        sys.exit("ERRO: Arquivo PROCESSADO não encontrado. Rode o preprocessamento.py primeiro.")
        
    df = pd.read_csv(ARQUIVO_DADOS, sep=';', decimal=',')
    print(f"Dataset carregado: {len(df)} amostras.")

    # 1. SELEÇÃO DE COLUNAS (USAR APENAS DERIVADA 'd1_')
    # O pipeline anterior gerou colunas raw ('Band_') e derivada ('d1_Band_')
    # Vamos focar SOMENTE nas derivadas para evitar colinearidade.
    cols_derivada = [c for c in df.columns if c.startswith('d1_Band_')]
    
    # Filtra intervalos e remove bandas ruins
    cols_limpas = filtrar_e_limpar_bandas(cols_derivada, INTERVALOS_DE_INTERESSE, BANDAS_PARA_REMOVER)
    
    if len(cols_limpas) == 0:
        sys.exit("ERRO: Nenhuma banda restou após filtro e limpeza.")

    # Prepara X e y numérico (para o SPA)
    X_full = df[cols_limpas].apply(pd.to_numeric, errors='coerce').fillna(0)
    y_target_numeric = pd.to_numeric(df['Dose_N'], errors='coerce').fillna(0)

    # 2. RODAR SPA (Seleção de Variáveis)
    print("\n" + "="*40)
    print(" 1. SELEÇÃO DE BANDAS ÓTIMAS (SPA)")
    print("="*40)
    
    spa = SPA_Selector(max_vars=SPA_MAX_VARIAVEIS, test_size=SPA_TEST_SIZE)
    cols_finais = spa.fit(X_full, y_target_numeric)
    
    # Ordena para exibição
    wls_finais = sorted([extrair_valor_onda(c) for c in cols_finais])
    print(f"\nRESULTADO FINAL SPA:")
    print(f" > Bandas Analisadas (Derivada): {len(cols_limpas)}")
    print(f" > Bandas Selecionadas: {len(cols_finais)}")
    print(f" > Comprimentos de Onda: {wls_finais}")

    # Atualiza X apenas com as vencedoras
    X_selected = X_full[cols_finais]
    loo = LeaveOneOut()

    # 3. TREINAMENTO E VALIDAÇÃO (Random Forest)
    print("\n" + "="*40)
    print(" 2. TREINAMENTO DO MODELO (Random Forest)")
    print("="*40)

    # --- Análise por DOSE ---
    y_dose = df['Dose_N'].astype(str)
    print(f"\n> Classificando por DOSE ({len(y_dose.unique())} classes)...")
    
    # Mostra onde as bandas caíram no espectro
    plotar_espectro_selecionado(df, cols_limpas, cols_finais, 'Dose_N', "Doses")
    
    rf_dose = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    y_pred_dose = cross_val_predict(rf_dose, X_selected, y_dose, cv=loo, n_jobs=-1)
    
    classes_dose = sorted(y_dose.unique(), key=lambda x: float(x))
    avaliar_modelo(y_dose, y_pred_dose, "Classificação Doses", labels_nomes=classes_dose)
    
    # Importância das Features (Qual banda pesou mais?)
    rf_dose.fit(X_selected, y_dose)
    plotar_importancia(rf_dose, cols_finais, "Doses")

    # --- Análise BINÁRIA (Com N vs Sem N) ---
    print(f"\n> Classificando BINÁRIO (Com N vs Sem N)...")
    y_binario = np.where(y_target_numeric > 0, "Com N", "Sem N")
    
    rf_bin = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    y_pred_bin = cross_val_predict(rf_bin, X_selected, y_binario, cv=loo, n_jobs=-1)
    avaliar_modelo(y_binario, y_pred_bin, "Binário")
    
    print("\nProcesso concluído com sucesso.")