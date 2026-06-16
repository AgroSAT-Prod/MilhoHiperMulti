import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# ================= CONFIGURAÇÕES =================
PASTA_DO_SCRIPT = os.path.dirname(os.path.abspath(__file__))
ARQUIVO_DADOS = os.path.join(PASTA_DO_SCRIPT, 'DATASET_IA_PROCESSADO.csv')

SEED = 42

BANDAS_ALVO = [
    430.0, 450.0, 460.0, 500.0, 520.0, 550.0, 600.0, 625.0,
    640.0, 660.0, 685.0, 720.0, 722.0, 740.0, 970.0, 990.0, 995.0
]

SPA_MAX_VARIAVEIS = 10
SPA_TEST_SIZE = 0.3

# ================= CLASSE SPA =================
class SPA_Selector:
    def __init__(self, max_vars=20, test_size=0.3):
        self.max_vars = max_vars
        self.test_size = test_size
        self.selected_cols = []
        self.best_rmse = None
        
    def fit(self, X, y):
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        
        print(f" > [SPA] Iniciando seleção (Buscando as {self.max_vars} bandas mais responsivas ao N)...")
        
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
        
        chains = {}
        
        for k0 in range(n_cols):
            x_projected = X_cal_mat.copy()
            selected_indices = [k0]
            
            for k in range(1, k_max):
                last_idx = selected_indices[-1]
                v_last = x_projected[:, last_idx].reshape(-1, 1)
                
                norm_sq = np.dot(v_last.T, v_last)
                if norm_sq < 1e-10:
                    break
                
                proj_factor = np.dot(v_last.T, x_projected) / norm_sq
                x_projected = x_projected - np.dot(v_last, proj_factor)
                
                norms = np.sum(x_projected**2, axis=0)
                norms[selected_indices] = -1
                next_idx = np.argmax(norms)
                selected_indices.append(next_idx)
            
            chains[k0] = selected_indices

        lr = LinearRegression()
        
        for k0, indices_chain in chains.items():
            for n_vars in range(1, k_max + 1):
                subset = indices_chain[:n_vars]
                lr.fit(X_cal_mat[:, subset], y_cal_mat)
                y_pred = lr.predict(X_val_mat[:, subset])
                rmse = np.sqrt(np.mean((y_val_mat - y_pred)**2))
                
                if rmse < min_rmse_global:
                    min_rmse_global = rmse
                    best_chain = subset
        
        self.selected_cols = [feature_names[i] for i in best_chain]
        self.best_rmse = min_rmse_global
        return self.selected_cols

# ================= CLASE PLS =================
class PLS_Selector:
    def __init__(self, max_vars=20, test_size=0.3, n_components=7):
        self.max_vars = max_vars
        self.test_size = test_size
        self.n_components = n_components
        self.selected_cols = []
        self.best_rmse = None
        
    def fit(self, X, y):
        from sklearn.model_selection import train_test_split
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.metrics import mean_squared_error
        
        print(f" > [PLS] Iniciando seleção (Buscando as {self.max_vars} bandas mais importantes)...")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=SEED)
        
        X_train_mat = np.array(X_train)
        X_test_mat = np.array(X_test)
        y_train_mat = np.array(y_train)
        y_test_mat = np.array(y_test)
        
        feature_names = X.columns.tolist()
        
        n_comp = min(self.n_components, X_train_mat.shape[1], len(X_train_mat) - 1)
        pls = PLSRegression(n_components=n_comp, scale=True)
        pls.fit(X_train_mat, y_train_mat.reshape(-1, 1))
        
        vip = self._calcular_vip(pls, X_train_mat)
        vip_sorted_idx = np.argsort(vip)[::-1]
        
        n_select = min(self.max_vars, len(feature_names))
        
        best_rmse = float('inf')
        best_n = 1
        
        for n in range(1, n_select + 1):
            subset_idx = vip_sorted_idx[:n]
            
            pls_test = PLSRegression(n_components=min(3, n), scale=True)
            pls_test.fit(X_train_mat[:, subset_idx], y_train_mat.reshape(-1, 1))
            
            y_pred = pls_test.predict(X_test_mat[:, subset_idx]).flatten()
            rmse = np.sqrt(mean_squared_error(y_test_mat, y_pred))
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_n = n
        
        final_indices = vip_sorted_idx[:best_n]
        self.selected_cols = [feature_names[i] for i in final_indices]
        self.best_rmse = best_rmse
        
        return self.selected_cols
    
    def _calcular_vip(self, modelo, X):
        T = modelo.x_scores_
        W = modelo.x_weights_
        Q = modelo.y_loadings_.flatten()
        
        m = X.shape[1]
        n_comp = T.shape[1]
        
        VIP = np.zeros(m)
        SS = np.sum(T**2, axis=0) * (Q**2)
        SS_total = np.sum(SS)
        
        for i in range(m):
            numerador = 0
            for k in range(n_comp):
                w_norm = np.linalg.norm(W[:, k])
                if w_norm > 0:
                    numerador += (W[i, k] / w_norm)**2 * SS[k]
            
            if SS_total != 0:
                VIP[i] = np.sqrt(m * numerador / SS_total)
            else:
                VIP[i] = 0
        
        return VIP

# ================= FUNÇÕES AUXILIARES =================

def extrair_valor_onda(nome_coluna):
    try:
        limpo = nome_coluna.replace('d1_', '').replace('Band_', '').replace('nm', '')
        return float(limpo)
    except:
        return None

def selecionar_colunas_por_lista(todas_colunas, bandas_alvo, tolerancia=1.5):
    selecionadas = []
    bandas_encontradas = []
    
    for banda in bandas_alvo:
        for col in todas_colunas:
            wl = extrair_valor_onda(col)
            if wl is not None and abs(wl - banda) <= tolerancia:
                if col not in selecionadas:
                    selecionadas.append(col)
                    bandas_encontradas.append(banda)
                break
    return selecionadas, bandas_encontradas

def plotar_correlacao_espectral(dataframe, todas_cols, titulo="Resposta Espectral ao Nitrogênio"):
    todas_ordenadas = sorted(todas_cols, key=lambda x: extrair_valor_onda(x))
    wls = [extrair_valor_onda(c) for c in todas_ordenadas]
    
    correlacoes = []
    y_dose = pd.to_numeric(dataframe['Dose_N'], errors='coerce').fillna(0)
    
    for col in todas_ordenadas:
        corr = dataframe[col].corr(y_dose)
        correlacoes.append(corr)
    
    plt.figure(figsize=(14, 6))
    cores = ['red' if c > 0 else 'blue' for c in correlacoes]
    plt.bar(range(len(wls)), correlacoes, color=cores, alpha=0.7, edgecolor='black', linewidth=0.5)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    plt.title(titulo, fontsize=14, fontweight='bold')
    plt.xlabel("Comprimento de Onda (nm)", fontsize=12)
    plt.ylabel("Correlação de Pearson (r)", fontsize=12)
    plt.xticks(range(len(wls)), [f'{w:.0f}' for w in wls], rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('correlacao_espectral_nitrogenio.png', dpi=300)
    plt.show()

def plotar_espectro_selecionado(dataframe, todas_cols, cols_spa, titulo):
    todas_ordenadas = sorted(todas_cols, key=lambda x: extrair_valor_onda(x))
    wls_full = [extrair_valor_onda(c) for c in todas_ordenadas]
    wls_spa = [extrair_valor_onda(c) for c in cols_spa]
    
    correlacoes = []
    y_dose = pd.to_numeric(dataframe['Dose_N'], errors='coerce').fillna(0)
    
    for col in todas_ordenadas:
        corr = dataframe[col].corr(y_dose)
        correlacoes.append(abs(corr))
    
    plt.figure(figsize=(14, 6))
    plt.plot(wls_full, correlacoes, label='Correlação com N (|r|)', color='gray', linewidth=2, alpha=0.7)
    plt.fill_between(wls_full, 0, correlacoes, color='gray', alpha=0.1)
    
    for i, wl in enumerate(wls_spa):
        idx_prox = (np.abs(np.array(wls_full) - wl)).argmin()
        r_val = correlacoes[idx_prox]
        
        plt.scatter(wl, r_val, color='red', s=150, zorder=5, edgecolors='darkred', linewidth=2)
        plt.text(wl, r_val + 0.08, f"{wl:.0f}nm\n(r={r_val:.3f})", 
                 ha='center', fontsize=9, fontweight='bold', color='darkred')
        plt.axvline(x=wl, color='red', linestyle='--', alpha=0.4, linewidth=1.5)
    
    plt.title(f"Bandas Selecionadas pelo SPA - {titulo}", fontsize=14, fontweight='bold')
    plt.xlabel("Comprimento de Onda (nm)", fontsize=12)
    plt.ylabel("Correlação Absoluta de Pearson (|r|)", fontsize=12)
    plt.ylim(0, max(correlacoes) * 1.2)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('spa_bandas_selecionadas.png', dpi=300)
    plt.show()

def plotar_boxplots_bandas(dataframe, cols_selecionadas, target_col='Dose_N', top_n=6):
    n_cols = min(top_n, len(cols_selecionadas))
    
    plt.figure(figsize=(16, 5))
    df_sorted = dataframe.sort_values(by=target_col, 
                                      key=lambda col: pd.to_numeric(col, errors='coerce'))
    
    for i, col in enumerate(cols_selecionadas[:n_cols]):
        plt.subplot(2, 3, i+1)
        sns.boxplot(x=target_col, y=col, data=df_sorted, hue=target_col, legend=False)
        wl = extrair_valor_onda(col)
        plt.title(f"Banda: {wl:.0f} nm", fontweight='bold')
        plt.ylabel("1ª Derivada")
        plt.xlabel("Dose de N (kg/ha)")
        plt.grid(True, alpha=0.2, axis='y')
    
    plt.suptitle("Separação das Classes nas Bandas Selecionadas", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('boxplots_bandas_nitrogenio.png', dpi=300)
    plt.show()

def calcular_importancia_nitrogenio(dataframe, colunas, target_col='Dose_N'):
    y_dose = pd.to_numeric(dataframe[target_col], errors='coerce').fillna(0)
    importancias = {}
    
    for col in colunas:
        X_col = pd.to_numeric(dataframe[col], errors='coerce').fillna(0)
        wl = extrair_valor_onda(col)
        
        corr = abs(X_col.corr(y_dose))
        grupos = []
        doses_unicas = sorted(y_dose.unique())
        
        for dose in doses_unicas:
            mask = y_dose == dose
            grupos.append(X_col[mask].values)
        
        media_geral = X_col.mean()
        ss_between = sum(len(g) * (np.mean(g) - media_geral)**2 for g in grupos)
        ss_within = sum(np.sum((g - np.mean(g))**2) for g in grupos)
        
        df_between = len(doses_unicas) - 1
        df_within = len(X_col) - len(doses_unicas)
        
        if df_between > 0 and df_within > 0 and ss_within > 0:
            ms_between = ss_between / df_between
            ms_within = ss_within / df_within
            f_statistic = ms_between / ms_within if ms_within > 0 else 0
        else:
            f_statistic = 0
        
        importancia_combinada = (corr * 0.5) + (f_statistic / (f_statistic + 1) * 0.5)
        
        importancias[f'{wl:.2f}nm'] = {
            'coluna': col,
            'wavelength': wl,
            'correlacao': corr,
            'f_statistic': f_statistic,
            'importancia': importancia_combinada
        }
    
    return pd.DataFrame(importancias).T.sort_values('importancia', ascending=False)

def plotar_importancia_nitrogenio(df_importancia, titulo="Importância das Bandas para Resposta ao N"):
    plt.figure(figsize=(12, 6))
    
    wls = [float(idx.split('nm')[0]) for idx in df_importancia.index]
    importancias = df_importancia['importancia'].values.astype(float)
    correlacoes = df_importancia['correlacao'].values.astype(float)
    
    ordem = np.argsort(wls)
    wls = np.array(wls)[ordem]
    importancias = importancias[ordem]
    correlacoes = correlacoes[ordem]
    
    # Normalizar correlações para [0, 1]
    correlacoes_norm = (correlacoes - correlacoes.min()) / (correlacoes.max() - correlacoes.min() + 1e-10)
    cores = plt.cm.RdYlGn(correlacoes_norm)
    
    plt.bar(range(len(wls)), importancias, color=cores, edgecolor='black', linewidth=1.5)
    plt.xticks(range(len(wls)), [f'{w:.0f}' for w in wls], rotation=45)
    plt.ylabel('Importância para Resposta ao N', fontsize=12, fontweight='bold')
    plt.xlabel('Comprimento de Onda (nm)', fontsize=12, fontweight='bold')
    plt.title(titulo, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('importancia_nitrogenio.png', dpi=300)
    plt.show()
    
    return df_importancia

def exibir_ranking_importancia(df_importancia, titulo="Ranking de Bandas por Resposta ao N"):
    print(f"\n{'='*70}")
    print(f" {titulo.upper()}")
    print(f"{'='*70}")
    print(f"\n{'Rank':<6} {'Banda (nm)':<15} {'Importância':<15} {'Correlação':<15} {'F-stat':<15}")
    print("-" * 70)
    
    for idx, (band, row) in enumerate(df_importancia.iterrows(), 1):
        print(f"{idx:<6} {row['wavelength']:<15.2f} {row['importancia']:<15.4f} {row['correlacao']:<15.4f} {row['f_statistic']:<15.4f}")
    
    print(f"{'='*70}\n")

def plotar_importancia_features(modelo, colunas_X, titulo):
    if isinstance(modelo, Pipeline):
        rf_model = modelo.named_steps['rf']
    else:
        rf_model = modelo
    
    importancias = rf_model.feature_importances_
    indices = np.argsort(importancias)[::-1]
    
    wls = [extrair_valor_onda(colunas_X[i]) for i in indices]
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(colunas_X)), importancias[indices], align="center", 
            color='steelblue', edgecolor='black', alpha=0.8)
    plt.title(f"Importância das Bandas no Random Forest - {titulo}", 
              fontsize=14, fontweight='bold')
    plt.xlabel("Bandas (nm)", fontsize=12)
    plt.ylabel("Importância", fontsize=12)
    plt.xticks(range(len(colunas_X)), [f'{w:.0f}' for w in wls], rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'importancia_rf_{titulo.lower().replace(" ", "_")}.png', dpi=300)
    plt.show()

def avaliar_modelo(y_true, y_pred, titulo, labels_nomes=None):
    print(f"\n[{titulo}] Relatório de Classificação:")
    print(classification_report(y_true, y_pred, target_names=labels_nomes, zero_division=0))
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels_nomes if labels_nomes else "auto",
                yticklabels=labels_nomes if labels_nomes else "auto",
                cbar_kws={'label': 'Contagem'})
    plt.title(f"Matriz de Confusão: {titulo}", fontsize=14, fontweight='bold')
    plt.xlabel("Predito", fontsize=12)
    plt.ylabel("Verdadeiro", fontsize=12)
    plt.tight_layout()
    plt.savefig(f'matriz_confusao_{titulo.lower().replace(" ", "_")}.png', dpi=300)
    plt.show()

# ================= EXECUÇÃO PRINCIPAL =================

if __name__ == "__main__":
    print("=== SELEÇÃO DE BANDAS POR RESPOSTA AO NITROGÊNIO (SPA + PLS) ===\n")
    
    if not os.path.exists(ARQUIVO_DADOS):
        sys.exit(f"ERRO: Arquivo {ARQUIVO_DADOS} não encontrado.")
    
    df = pd.read_csv(ARQUIVO_DADOS, sep=';', decimal=',')
    print(f"Dataset carregado: {len(df)} amostras.\n")

    print("="*70)
    print(" 1. FILTRAGEM PARA BANDAS ESPECÍFICAS")
    print("="*70)
    
    cols_totais = [c for c in df.columns if 'Band_' in c and c.startswith('d1_')]
    cols_selecionadas, bandas_encontradas = selecionar_colunas_por_lista(cols_totais, BANDAS_ALVO)
    
    print(f"\nBandas mapeadas: {len(cols_selecionadas)}")
    print(f"Wavelengths (nm): {sorted([extrair_valor_onda(c) for c in cols_selecionadas])}\n")
    
    if len(cols_selecionadas) == 0:
        sys.exit("ERRO: Nenhuma banda foi encontrada.")
    
    X_full = df[cols_selecionadas].apply(pd.to_numeric, errors='coerce').fillna(0)
    y_target = pd.to_numeric(df['Dose_N'], errors='coerce').fillna(0)

    print("="*70)
    print(" 2. ANÁLISE DE RESPOSTA ESPECTRAL AO NITROGÊNIO")
    print("="*70)
    plotar_correlacao_espectral(df, cols_selecionadas)

    print("\n" + "="*70)
    print(" 3. SELEÇÃO DE BANDAS ÓTIMAS (SPA)")
    print("="*70)
    
    spa = SPA_Selector(max_vars=SPA_MAX_VARIAVEIS, test_size=SPA_TEST_SIZE)
    cols_finais = spa.fit(X_full, y_target)
    
    wls_finais = sorted([extrair_valor_onda(c) for c in cols_finais])
    print(f"\n✓ RESULTADO FINAL SPA: {len(cols_finais)} bandas selecionadas")
    print(f"  Wavelengths (nm): {wls_finais}")
    print(f"  RMSE de Validação: {spa.best_rmse:.4f}\n")

    print("\n" + "="*70)
    print(" 3B. SELEÇÃO DE BANDAS ÓTIMAS (PLS)")
    print("="*70)
    
    pls = PLS_Selector(max_vars=SPA_MAX_VARIAVEIS, test_size=SPA_TEST_SIZE, n_components=3)
    cols_finais_pls = pls.fit(X_full, y_target)
    
    wls_finais_pls = sorted([extrair_valor_onda(c) for c in cols_finais_pls])
    print(f"\n✓ RESULTADO FINAL PLS: {len(cols_finais_pls)} bandas selecionadas")
    print(f"  Wavelengths (nm): {wls_finais_pls}")
    print(f"  RMSE de Validação: {pls.best_rmse:.4f}\n")

    print("\n" + "="*70)
    print(" 4. VISUALIZAÇÕES E VALIDAÇÃO (SPA)")
    print("="*70)
    
    plotar_espectro_selecionado(df, cols_selecionadas, cols_finais, "Resposta ao N (SPA)")
    plotar_boxplots_bandas(df, cols_finais, top_n=6)

    print("\n" + "="*70)
    print(" 4B. VISUALIZAÇÕES E VALIDAÇÃO (PLS)")
    print("="*70)
    
    plotar_espectro_selecionado(df, cols_selecionadas, cols_finais_pls, "Resposta ao N (PLS)")
    plotar_boxplots_bandas(df, cols_finais_pls, top_n=6)

    print("\n" + "="*70)
    print(" 5. CLASSIFICAÇÃO POR DOSE (MULTI-CLASSE) - SPA")
    print("="*70)
    
    X_selected = X_full[cols_finais]
    y_dose = df['Dose_N'].astype(str)
    classes_dose = sorted(y_dose.unique(), key=lambda x: float(x))
    
    print(f"\nClasses (Doses): {classes_dose}")
    
    pipeline_dose = Pipeline([
        ('smote', SMOTE(random_state=SEED, k_neighbors=1)),
        ('rf', RandomForestClassifier(n_estimators=100, class_weight='balanced', 
                                     random_state=SEED, n_jobs=1))
    ])
    
    pipeline_dose.fit(X_selected, y_dose)
    y_pred_dose = pipeline_dose.predict(X_selected)
    
    avaliar_modelo(y_dose, y_pred_dose, "Classificação por Dose (SPA)", labels_nomes=classes_dose)
    plotar_importancia_features(pipeline_dose, cols_finais, "Classificação por Dose (SPA)")
    
    df_imp_dose = calcular_importancia_nitrogenio(df, cols_finais, 'Dose_N')
    plotar_importancia_nitrogenio(df_imp_dose, titulo="Importância das Bandas para Resposta ao N (SPA Multi-classe)")
    exibir_ranking_importancia(df_imp_dose, titulo="Ranking de Bandas por Resposta ao N (SPA Multi-classe)")

    print("\n" + "="*70)
    print(" 6. CLASSIFICAÇÃO BINÁRIA (COM N vs SEM N) - SPA")
    print("="*70)
    
    y_binario = np.where(y_target > 0, "Com N", "Sem N")
    
    pipeline_bin = Pipeline([
        ('smote', SMOTE(random_state=SEED, k_neighbors=3)),
        ('rf', RandomForestClassifier(n_estimators=100, class_weight='balanced', 
                                     random_state=SEED, n_jobs=1))
    ])
    
    pipeline_bin.fit(X_selected, y_binario)
    y_pred_bin = pipeline_bin.predict(X_selected)
    
    avaliar_modelo(y_binario, y_pred_bin, "Classificação Binária (SPA)", labels_nomes=["Sem N", "Com N"])
    
    df_temp = df.copy()
    df_temp['N_binario'] = np.where(y_target > 0, 1, 0)
    
    df_imp_bin = calcular_importancia_nitrogenio(df_temp, cols_finais, 'N_binario')
    plotar_importancia_nitrogenio(df_imp_bin, titulo="Importância das Bandas para Resposta ao N (SPA Binário)")
    exibir_ranking_importancia(df_imp_bin, titulo="Ranking de Bandas por Resposta ao N (SPA Binária)")

    print("\n" + "="*70)
    print(" 7. CLASSIFICAÇÃO POR DOSE (MULTI-CLASSE) - PLS")
    print("="*70)
    
    X_selected_pls = X_full[cols_finais_pls]
    
    pipeline_dose_pls = Pipeline([
        ('smote', SMOTE(random_state=SEED, k_neighbors=1)),
        ('rf', RandomForestClassifier(n_estimators=100, class_weight='balanced', 
                                     random_state=SEED, n_jobs=1))
    ])
    
    pipeline_dose_pls.fit(X_selected_pls, y_dose)
    y_pred_dose_pls = pipeline_dose_pls.predict(X_selected_pls)
    
    avaliar_modelo(y_dose, y_pred_dose_pls, "Classificação por Dose (PLS)", labels_nomes=classes_dose)
    plotar_importancia_features(pipeline_dose_pls, cols_finais_pls, "Classificação por Dose (PLS)")
    
    df_imp_dose_pls = calcular_importancia_nitrogenio(df, cols_finais_pls, 'Dose_N')
    plotar_importancia_nitrogenio(df_imp_dose_pls, titulo="Importância das Bandas para Resposta ao N (PLS Multi-classe)")
    exibir_ranking_importancia(df_imp_dose_pls, titulo="Ranking de Bandas por Resposta ao N (PLS Multi-classe)")
    
    print("\n" + "="*70)
    print(" 8. CLASSIFICAÇÃO BINÁRIA (COM N vs SEM N) - PLS")
    print("="*70)
    
    pipeline_bin_pls = Pipeline([
        ('smote', SMOTE(random_state=SEED, k_neighbors=3)),
        ('rf', RandomForestClassifier(n_estimators=100, class_weight='balanced', 
                                     random_state=SEED, n_jobs=1))
    ])