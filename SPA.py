import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Importações para SPA e MLR
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Importações para Random Forest final
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ================= CONFIGURAÇÕES =================
PASTA_DO_SCRIPT = os.path.dirname(os.path.abspath(__file__))
ARQUIVO_DADOS = os.path.join(PASTA_DO_SCRIPT, 'DATASET_IA_PROCESSADO.csv')

SEED = 42

# Configurações do SPA
SPA_MAX_VARIAVEIS = 20  # Máximo de bandas que queremos selecionar
SPA_TEST_SIZE = 0.3     # Tamanho do conjunto de validação interna do SPA

# Intervalos de interesse
INTERVALOS_DE_INTERESSE = [
    (520, 580),   # Pico do Verde
    (690, 760)    # Red Edge
] 

# ================= CLASSE SPA (Successive Projections Algorithm) =================

class SPA_Selector:
    def __init__(self, max_vars=20, test_size=0.3):
        self.max_vars = max_vars
        self.test_size = test_size
        self.selected_cols = []
        self.best_rmse = float('inf')
        
    def fit(self, X, y):
        """
        X: DataFrame das bandas
        y: Alvo numérico (Dose N)
        """
        print(f" > Iniciando SPA (Max {self.max_vars} variáveis)...")
        
        # 1. Preparação dos dados (Divisão Calibração/Validação interna)
        # O SPA precisa validar qual cadeia de variáveis funciona melhor
        X_cal, X_val, y_cal, y_val = train_test_split(X, y, test_size=self.test_size, random_state=SEED)
        
        X_cal_mat = np.array(X_cal)
        X_val_mat = np.array(X_val)
        y_cal_mat = np.array(y_cal)
        y_val_mat = np.array(y_val)
        
        n_samples, n_cols = X_cal_mat.shape
        feature_names = X.columns.tolist()
        
        # Se tivermos menos colunas que o max_vars solicitado, ajustamos
        k_max = min(self.max_vars, n_cols - 1)
        
        # Armazena o RMSE para cada "Cadeia" gerada começando por uma banda diferente
        best_chain = None
        min_rmse_global = float('inf')
        
        # --- FASE 1: PROJEÇÕES (Busca Geométrica) ---
        # O SPA original testa começar por CADA variável possível k(0)
        # Para ganhar tempo se houver muitas bandas, podemos limitar, 
        # mas para <500 bandas, rodar tudo é rápido.
        
        print(f"   Calculando projeções...")
        
        # Matriz para guardar as cadeias de índices selecionados
        # chains[i] = lista de índices selecionados começando pela banda i
        chains = {} 
        
        for k0 in range(n_cols):
            # Inicializa projeção
            x_cal_projected = X_cal_mat.copy()
            selected_indices = [k0]
            
            # A primeira banda é a k0
            # Agora projetamos as outras no subespaço ortogonal a k0
            
            for k in range(1, k_max):
                # Última variável selecionada
                last_idx = selected_indices[-1]
                v_last = x_cal_projected[:, last_idx].reshape(-1, 1)
                
                # Operador de Projeção Ortogonal: P = I - (v v^T) / (v^T v)
                # Aplicamos a projeção em todas as colunas restantes
                norm_sq = np.dot(v_last.T, v_last)
                
                if norm_sq == 0: break # Evitar divisão por zero
                
                # Projeção: X_new = X_old - v * (v^T * X_old) / (v^T * v)
                projection_factor = np.dot(v_last.T, x_cal_projected) / norm_sq
                x_cal_projected = x_cal_projected - np.dot(v_last, projection_factor)
                
                # A próxima variável é a que tem a maior norma (maior vetor residual)
                norms = np.sum(x_cal_projected**2, axis=0)
                
                # Zera as normas das que já foram escolhidas para não repetir
                norms[selected_indices] = -1 
                
                next_idx = np.argmax(norms)
                selected_indices.append(next_idx)
            
            chains[k0] = selected_indices

        # --- FASE 2: AVALIAÇÃO (MLR - Multiple Linear Regression) ---
        print(f"   Avaliando melhores cadeias via MLR...")
        
        # Vamos testar qual cadeia e qual número de variáveis dá o menor erro no dataset de validação
        lr = LinearRegression()
        
        for k0, indices_chain in chains.items():
            # Testa subsets crescentes: [Var1], [Var1, Var2], ...
            for n_vars in range(1, k_max + 1):
                subset = indices_chain[:n_vars]
                
                # Treina MLR
                lr.fit(X_cal_mat[:, subset], y_cal_mat)
                y_pred = lr.predict(X_val_mat[:, subset])
                
                # Calcula RMSE
                rmse = np.sqrt(np.mean((y_val_mat - y_pred)**2))
                
                if rmse < min_rmse_global:
                    min_rmse_global = rmse
                    best_chain = subset
        
        self.selected_cols = [feature_names[i] for i in best_chain]
        self.best_rmse = min_rmse_global
        
        return self.selected_cols

# ================= FUNÇÕES AUXILIARES =================

def extrair_valor_onda(nome_coluna):
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
    plt.title(f"Feature Importance (Seleção SPA) - {titulo}")
    nomes = [colunas_X[i].replace('d1_Band_', '').replace('nm', '') + 'nm' for i in indices_top]
    plt.bar(range(top_n), importancias[indices_top], color='#2ca02c')
    plt.xticks(range(top_n), nomes, rotation=45)
    plt.tight_layout()
    plt.show()

# ================= PLOTAGEM ESPECTRAL (Mantida a que você gostou) =================

def plotar_espectro_com_destaque(dataframe, todas_cols, cols_selecionadas, coluna_alvo, titulo):
    todas_cols_ordenadas = sorted(todas_cols, key=lambda x: extrair_valor_onda(x))
    wls_full = [extrair_valor_onda(c) for c in todas_cols_ordenadas]
    wls_selected = [extrair_valor_onda(c) for c in cols_selecionadas]
    grupos = dataframe.groupby(coluna_alvo)[todas_cols_ordenadas].mean()
    
    plt.figure(figsize=(12, 6))
    if len(grupos) == 2: cores = ['#e74c3c', '#27ae60']
    else: cores = plt.cm.viridis(np.linspace(0, 1, len(grupos)))

    # Curvas de fundo
    for i, (classe, linha_media) in enumerate(grupos.iterrows()):
        plt.plot(wls_full, linha_media, label=f"{classe}", color=cores[i], linewidth=2.5)

    # Linhas verticais das bandas selecionadas
    for j, wl in enumerate(wls_selected):
        label_line = 'Bandas SPA' if j == 0 else ""
        plt.axvline(x=wl, color='black', linestyle='--', alpha=0.6, linewidth=1.2, label=label_line)

    plt.title(f"Bandas Selecionadas pelo SPA - {titulo}", fontsize=14)
    plt.xlabel("Comprimento de Onda (nm)")
    plt.ylabel("Intensidade")
    plt.legend()
    plt.grid(True, alpha=0.3)
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

    # 1. Preparação
    cols_todas_raw = [c for c in df.columns if c.startswith('d1_Band_') or c.startswith('Band_')]
    cols_candidatas = filtrar_bandas(cols_todas_raw, INTERVALOS_DE_INTERESSE)
    
    # SPA requer dados numéricos
    X_full = df[cols_candidatas].apply(pd.to_numeric, errors='coerce').fillna(0)
    y_target_numeric = pd.to_numeric(df['Dose_N'], errors='coerce').fillna(0)

    # 2. Executar SPA
    print("\n" + "="*40)
    print(" INICIANDO ALGORITMO SPA")
    print("="*40)
    
    spa = SPA_Selector(max_vars=SPA_MAX_VARIAVEIS, test_size=0.3)
    cols_finais = spa.fit(X_full, y_target_numeric)
    
    print(f"\nRESULTADO SPA:")
    print(f" > Bandas Iniciais: {len(cols_candidatas)}")
    print(f" > Bandas Selecionadas: {len(cols_finais)}")
    
    wls_ordenados = sorted([extrair_valor_onda(c) for c in cols_finais])
    print(f" > Comprimentos: {wls_ordenados}")

    X_selected = X_full[cols_finais]
    loo = LeaveOneOut()

    # 3. Classificação
    print("\n[1/2] Classificando por DOSE (Random Forest + SPA bands)...")
    y_dose = df['Dose_N'].astype(str)
    
    # Plota antes para visualizar
    plotar_espectro_com_destaque(df, cols_candidatas, cols_finais, 'Dose_N', "Doses")
    
    rf_dose = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    y_pred_dose = cross_val_predict(rf_dose, X_selected, y_dose, cv=loo, n_jobs=-1)
    avaliar_modelo(y_dose, y_pred_dose, "Doses")

    print("\n[2/2] Classificando Binário...")
    y_binario = np.where(y_target_numeric > 0, "Com Nitrogenio", "Sem Nitrogenio")
    df['Status_N'] = np.where(y_target_numeric > 0, "Com N", "Sem N")
    
    plotar_espectro_com_destaque(df, cols_candidatas, cols_finais, 'Status_N', "Binário")
    
    rf_bin = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    y_pred_bin = cross_val_predict(rf_bin, X_selected, y_binario, cv=loo, n_jobs=-1)
    avaliar_modelo(y_binario, y_pred_bin, "Binário")
    
    rf_bin.fit(X_selected, y_binario)
    plotar_importancia(rf_bin, cols_finais, "Binário")
    
    print("\nProcesso concluído.")