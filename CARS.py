import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Importações para CARS e PLS
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import scale

# Importações para Machine Learning
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ================= CONFIGURAÇÕES =================
PASTA_DO_SCRIPT = os.path.dirname(os.path.abspath(__file__))
ARQUIVO_DADOS = os.path.join(PASTA_DO_SCRIPT, 'DATASET_IA_PROCESSADO.csv')

SEED = 42

# Parâmetros do CARS
CARS_ITERATIONS = 50   
CARS_CV_FOLDS = 5      
CARS_N_COMPONENTS = 5  # Componentes PLS

# Intervalos de interesse
INTERVALOS_DE_INTERESSE = [
    (520, 580),   # Pico do Verde
    (690, 760)    # Red Edge
] 

# ================= CLASSE CARS =================

class CARS_Selector:
    def __init__(self, n_iter=50, n_components=3, cv=5):
        self.n_iter = n_iter
        self.n_components = n_components
        self.cv = cv
        self.best_subset = None
        self.best_rmse = float('inf')
        
    def fit(self, X, y):
        X_vals = np.array(X)
        y_vals = np.array(y)
        n_samples, n_vars = X_vals.shape
        
        RMSECV_list = []
        feat_num_list = []
        
        # Parâmetros do decaimento exponencial (EDF)
        a = (n_vars / 2) ** (1 / (self.n_iter - 1))
        k = np.log(n_vars / 2) / (self.n_iter - 1)
        
        curr_subset_idx = np.arange(n_vars)
        
        print(f" > Iniciando CARS com {n_vars} bandas e {self.n_iter} iterações...")
        
        for i in range(self.n_iter):
            # 1. Quantas variáveis manter (Decaimento)
            n_keep = int(np.round(n_vars * np.exp(-k * i)))
            if n_keep < 2: n_keep = 2
            
            # 2. Amostragem Monte Carlo
            rand_idx = np.random.choice(n_samples, int(n_samples * 0.8), replace=True)
            X_cal = X_vals[rand_idx][:, curr_subset_idx]
            y_cal = y_vals[rand_idx]
            
            # 3. Treinar PLS para pegar pesos
            n_comp_actual = min(self.n_components, len(curr_subset_idx) - 1)
            pls = PLSRegression(n_components=n_comp_actual)
            pls.fit(X_cal, y_cal)
            
            coefs = np.abs(pls.coef_).flatten()
            weights = coefs / np.sum(coefs)
            
            # 4. Seleção Adaptativa (Roleta viciada pelos pesos)
            sorted_idx_local = np.argsort(weights)[::-1]
            top_local_indices = sorted_idx_local[:n_keep]
            
            curr_subset_idx = curr_subset_idx[top_local_indices]
            
            # 5. Validação Cruzada
            kf = KFold(n_splits=self.cv, shuffle=True, random_state=SEED)
            errors = []
            X_subset = X_vals[:, curr_subset_idx]
            
            try:
                for train_idx, test_idx in kf.split(X_vals):
                    pls_val = PLSRegression(n_components=min(self.n_components, len(curr_subset_idx)-1))
                    pls_val.fit(X_subset[train_idx], y_vals[train_idx])
                    y_pred = pls_val.predict(X_subset[test_idx])
                    errors.append(np.mean((y_vals[test_idx] - y_pred.flatten())**2))
                
                rmse_curr = np.sqrt(np.mean(errors))
            except:
                rmse_curr = float('inf')
            
            RMSECV_list.append(rmse_curr)
            feat_num_list.append(len(curr_subset_idx))
            
            if rmse_curr < self.best_rmse:
                self.best_rmse = rmse_curr
                self.best_subset = curr_subset_idx.copy()
            
            if i % 10 == 0:
                print(f"   Iter {i}: {len(curr_subset_idx)} bandas | RMSECV: {rmse_curr:.4f}")

        # Gráfico de diagnóstico do CARS
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(feat_num_list, marker='.')
        plt.title('Decaimento Variáveis')
        plt.gca().invert_xaxis()
        
        plt.subplot(1, 2, 2)
        plt.plot(RMSECV_list, color='red', marker='.')
        plt.axvline(x=np.argmin(RMSECV_list), color='blue', linestyle='--', label='Ótimo')
        plt.title('RMSECV')
        plt.legend()
        plt.tight_layout()
        plt.show()

        return X.columns[self.best_subset].tolist()

# ================= FUNÇÕES AUXILIARES =================

def extrair_valor_onda(nome_coluna):
    """Transforma 'd1_Band_720.5nm' em float 720.5"""
    try:
        limpo = nome_coluna.replace('d1_', '').replace('Band_', '').replace('nm', '')
        return float(limpo)
    except:
        return 0.0

def filtrar_bandas(colunas, intervalos):
    if not intervalos: return colunas
    cols_selecionadas = []
    print(f"\nFiltrando bandas nos intervalos: {intervalos}")
    for col in colunas:
        wl = extrair_valor_onda(col)
        if wl > 0:
            for (inicio, fim) in intervalos:
                if inicio <= wl <= fim:
                    cols_selecionadas.append(col)
                    break
    return cols_selecionadas

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
    top_n = min(15, len(colunas_X))
    indices_top = indices[:top_n]
    plt.figure(figsize=(10, 5))
    plt.title(f"Feature Importance (Seleção CARS) - {titulo}")
    nomes = [colunas_X[i].replace('d1_Band_', '').replace('nm', '') + 'nm' for i in indices_top]
    plt.bar(range(top_n), importancias[indices_top], color='#2ca02c')
    plt.xticks(range(top_n), nomes, rotation=45)
    plt.tight_layout()
    plt.show()

# ================= NOVA FUNÇÃO DE PLOTAGEM =================

def plotar_espectro_com_destaque(dataframe, todas_cols, cols_selecionadas, coluna_alvo, titulo):
    """
    Plota a curva média completa (onda) e destaca com linhas verticais
    as bandas que foram selecionadas pelo CARS.
    """
    # 1. Organizar TODAS as bandas para o eixo X (para formar a onda)
    todas_cols_ordenadas = sorted(todas_cols, key=lambda x: extrair_valor_onda(x))
    wls_full = [extrair_valor_onda(c) for c in todas_cols_ordenadas]
    
    # 2. Extrair apenas os comprimentos de onda selecionados
    wls_selected = [extrair_valor_onda(c) for c in cols_selecionadas]
    
    # 3. Calcular a média por classe usando TODAS as bandas
    grupos = dataframe.groupby(coluna_alvo)[todas_cols_ordenadas].mean()
    
    plt.figure(figsize=(12, 6))
    
    # Definir cores
    if len(grupos) == 2:
        cores = ['#e74c3c', '#27ae60'] # Vermelho e Verde
    else:
        cores = plt.cm.viridis(np.linspace(0, 1, len(grupos)))

    # 4. Plotar as curvas espectrais contínuas (Background)
    for i, (classe, linha_media) in enumerate(grupos.iterrows()):
        plt.plot(wls_full, linha_media, label=f"{classe}", color=cores[i], linewidth=2.5)

    # 5. Adicionar linhas verticais nas bandas selecionadas
    # Usamos ymin/ymax para a linha não extrapolar muito o gráfico
    ymin, ymax = plt.ylim()
    for j, wl in enumerate(wls_selected):
        # Só adiciona label na primeira linha para não poluir a legenda
        label_line = 'Bandas CARS' if j == 0 else ""
        plt.axvline(x=wl, color='black', linestyle='--', alpha=0.5, linewidth=1, label=label_line)
        
        # Opcional: Colocar um pontinho na interseção (estético)
        # plt.scatter([wl]*len(grupos), grupos[todas_cols_ordenadas].loc[:, f"d1_Band_{wl}nm" if ...], color='black', s=10, zorder=5)

    plt.title(f"Espectro Completo com Seleção CARS - {titulo}", fontsize=14)
    plt.xlabel("Comprimento de Onda (nm)")
    plt.ylabel("Intensidade / Derivada")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Adicionar destaque de região de fundo (opcional, para visualização dos intervalos)
    for (inicio, fim) in INTERVALOS_DE_INTERESSE:
        plt.axvspan(inicio, fim, color='yellow', alpha=0.05)
        
    plt.tight_layout()
    plt.show()

# ================= EXECUÇÃO PRINCIPAL =================

if __name__ == "__main__":
    if not os.path.exists(ARQUIVO_DADOS):
        sys.exit("ERRO: Arquivo não encontrado.")
    
    df = pd.read_csv(ARQUIVO_DADOS, sep=';', decimal=',')
    print(f"Dataset carregado: {len(df)} amostras.")

    # 1. Prepara TODAS as bandas candidatas (para plotar a onda depois)
    cols_todas_raw = [c for c in df.columns if c.startswith('d1_Band_') or c.startswith('Band_')]
    cols_candidatas = filtrar_bandas(cols_todas_raw, INTERVALOS_DE_INTERESSE)
    
    X_full = df[cols_candidatas].apply(pd.to_numeric, errors='coerce').fillna(0)
    y_target_numeric = pd.to_numeric(df['Dose_N'], errors='coerce').fillna(0)

    # 2. Executar CARS
    print("\n" + "="*40)
    print(" INICIANDO ALGORITMO CARS")
    print("="*40)
    
    cars = CARS_Selector(n_iter=CARS_ITERATIONS, n_components=CARS_N_COMPONENTS)
    cols_finais = cars.fit(X_full, y_target_numeric)
    
    print(f"\nRESULTADO CARS:")
    print(f" > Bandas Iniciais: {len(cols_candidatas)}")
    print(f" > Bandas Selecionadas: {len(cols_finais)}")
    print(f" > Comprimentos: {sorted([extrair_valor_onda(c) for c in cols_finais])}")

    X_selected = X_full[cols_finais]
    loo = LeaveOneOut()

    # 3. Classificação e Plotagem Corrigida
    print("\n[1/2] Análise por DOSE...")
    y_dose = df['Dose_N'].astype(str)
    
    # --- AQUI ESTÁ A MUDANÇA: Plotar espectro completo + bandas selecionadas ---
    plotar_espectro_com_destaque(df, cols_candidatas, cols_finais, 'Dose_N', "Doses")
    
    rf_dose = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    y_pred_dose = cross_val_predict(rf_dose, X_selected, y_dose, cv=loo, n_jobs=-1)
    avaliar_modelo(y_dose, y_pred_dose, "Doses")

    print("\n[2/2] Análise BINÁRIA...")
    y_binario = np.where(y_target_numeric > 0, "Com Nitrogenio", "Sem Nitrogenio")
    df['Status_N'] = np.where(y_target_numeric > 0, "Com N", "Sem N")
    
    plotar_espectro_com_destaque(df, cols_candidatas, cols_finais, 'Status_N', "Binário")
    
    rf_bin = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    y_pred_bin = cross_val_predict(rf_bin, X_selected, y_binario, cv=loo, n_jobs=-1)
    avaliar_modelo(y_binario, y_pred_bin, "Binário")
    
    # Feature Importance (para ver qual das selecionadas pesou mais no RF)
    rf_bin.fit(X_selected, y_binario)
    plotar_importancia(rf_bin, cols_finais, "Binário")
    
    print("\nProcesso concluído.")