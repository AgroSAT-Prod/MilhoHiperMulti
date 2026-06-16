import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import sys
import warnings
import optuna

from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ================= CONFIGURAÇÕES =================

ARQUIVO_DADOS = "Biomassa14_01Clorofila.csv"
NOME_ALVO_DESEJADO = 'Kilos'

# --- LISTA COMPLETA DE VARIÁVEIS ---
PREDITORES_FIXOS = [
    'NGRDI', 'MSAVI', 'MGRVI', 'RGBIndex', 'VARI', 'TGI', 'GLI',
    'ExGRaw', 'ExG', 'ExRM', 'ExGR', 'NDVI', 'SAVI', 'RDVI',
    'NLI', 'CVI', 'GNDVI', 'PanNDVI', 'NDRE', 'NDVIRE', 'SFDVI'
]

# Configurações Gerais
MAX_COMPONENTES = 20

# Otimização de Outliers
SIGMA_START = 0.5  
SIGMA_STOP  = 3.0  
SIGMA_STEP  = 0.1
MAX_REMOCAO_PERMITIDA = 0.20

# ================= FUNÇÕES =================

def detectar_separador(caminho):
    with open(caminho, 'r', encoding='utf-8-sig') as f:
        linha = f.readline()
    for sep in [';', ',', '\t', '|']:
        if sep in linha:
            return sep
    return ','

def escolher_n_componentes(X, y, max_comp):
    n_max = min(max_comp, X.shape[0] - 1, X.shape[1])
    if n_max < 1: return 1, []
    loo, rmses = LeaveOneOut(), []
    for n in range(1, n_max + 1):
        pls = PLSRegression(n_components=n, scale=True)
        try:
            y_cv_pred = cross_val_predict(pls, X, y, cv=loo).ravel()
            rmse = np.sqrt(mean_squared_error(y, y_cv_pred))
            rmses.append(rmse)
        except: rmses.append(float('inf'))
    if not rmses: return 1, []
    return int(np.argmin(rmses)) + 1, rmses

def remover_outliers_iterativo(X, y_real, y_nl, sigma=1.5, max_iter=10, min_amostras=5):
    mascara_atual = np.ones(len(y_real), dtype=bool)
    for rodada in range(max_iter):
        X_iter, y_nl_iter, y_re_iter = X[mascara_atual], y_nl[mascara_atual], y_real[mascara_atual]
        if len(y_re_iter) <= min_amostras: break
        n_comp = min(3, len(y_re_iter) - 1, X_iter.shape[1])
        if n_comp < 1: break
        try:
            pls = PLSRegression(n_components=n_comp, scale=True)
            y_pred_nl_cv = cross_val_predict(pls, X_iter, y_nl_iter, cv=LeaveOneOut()).ravel()
        except: break
        mean_log, std_log = np.mean(np.log(y_re_iter)), np.std(np.log(y_re_iter))
        y_pred_cv = np.exp((y_pred_nl_cv * std_log) + mean_log)
        residuos = y_re_iter - y_pred_cv
        if np.std(residuos) == 0: break
        is_outlier = np.abs(residuos - np.mean(residuos)) > (sigma * np.std(residuos))
        if is_outlier.sum() == 0: break
        mascara_atual[np.where(mascara_atual)[0][is_outlier]] = False
    return mascara_atual

def rodar_otimizacao_sigma(nome_cenario, X_input, y_input):
    print(f"\n--- Testando {nome_cenario} ({len(y_input)} amostras) ---")
    if len(y_input) < 5:
        print("   AVISO: Amostras insuficientes para rodar PLS.")
        return -1.0, 0.0, np.ones(len(y_input), dtype=bool)

    y_log = np.log(y_input)
    std_log = np.std(y_log)
    if std_log == 0: std_log = 1
    y_nl = (y_log - np.mean(y_log)) / std_log
    
    melhor_r2 = -np.inf
    melhor_sigma = None
    melhor_mascara = None
    
    sigmas = np.arange(SIGMA_START, SIGMA_STOP + 0.01, SIGMA_STEP)
    for sigma in sigmas:
        mascara = remover_outliers_iterativo(X_input, y_input, y_nl, sigma=sigma)
        n_mantidos = mascara.sum()
        perc_removido = 1 - (n_mantidos / len(y_input))
        if perc_removido > MAX_REMOCAO_PERMITIDA or n_mantidos < 4: continue
        
        X_sub, y_sub = X_input[mascara], y_input[mascara]
        y_log_sub = np.log(y_sub)
        y_nl_sub = (y_log_sub - np.mean(y_log_sub)) / np.std(y_log_sub)
        try:
            n, _ = escolher_n_componentes(X_sub, y_nl_sub, MAX_COMPONENTES)
            pls = PLSRegression(n_components=n, scale=True)
            y_cv_nl = cross_val_predict(pls, X_sub, y_nl_sub, cv=LeaveOneOut()).ravel()
            y_cv = np.exp((y_cv_nl * np.std(y_log_sub)) + np.mean(y_log_sub))
            r2 = r2_score(y_sub, y_cv)
        except: r2 = -1
        
        if r2 > melhor_r2:
            melhor_r2, melhor_sigma, melhor_mascara = r2, sigma, mascara

    if melhor_sigma is None:
        print("   AVISO: Nenhum filtro funcionou. Usando TODOS os dados.")
        melhor_sigma = 0.0
        melhor_mascara = np.ones(len(y_input), dtype=bool)
        try:
            n, _ = escolher_n_componentes(X_input, y_nl, MAX_COMPONENTES)
            pls = PLSRegression(n_components=n, scale=True)
            y_cv_nl = cross_val_predict(pls, X_input, y_nl, cv=LeaveOneOut()).ravel()
            y_cv = np.exp((y_cv_nl * np.std(y_log)) + np.mean(y_log))
            melhor_r2 = r2_score(y_input, y_cv)
        except: melhor_r2 = -1.0

    print(f"   -> Resultado: Sigma={melhor_sigma:.1f} | R²={melhor_r2:.4f}")
    return melhor_r2, melhor_sigma, melhor_mascara

# ================= EXECUÇÃO PRINCIPAL (SUPER OTIMIZAÇÃO PLSR) =================

if __name__ == "__main__":
    print(f"=== ANÁLISE AVANÇADA EXCLUSIVA PLSR: '{NOME_ALVO_DESEJADO}' ===\n")
    if not os.path.exists(ARQUIVO_DADOS): sys.exit(f"ERRO: Arquivo '{ARQUIVO_DADOS}' não encontrado.")
    sep = detectar_separador(ARQUIVO_DADOS)
    
    try:
        df = pd.read_csv(ARQUIVO_DADOS, sep=sep, decimal=',', thousands='.', na_values=['-', ' -', '- ', 'nan', 'NaN'], encoding='utf-8-sig')
        df.columns = df.columns.str.strip()
    except Exception as e: sys.exit(f"Erro ao ler CSV: {e}")
    
    print(f"-> Total linhas no CSV: {len(df)}")

    cols_alvo = [c for c in df.columns if NOME_ALVO_DESEJADO.lower() in c.lower()]
    if not cols_alvo: sys.exit("ERRO: Alvo não encontrado.")
    ALVO = cols_alvo[0]
    
    df = df.dropna(subset=[ALVO])
    if df[ALVO].dtype == 'object': 
        df[ALVO] = pd.to_numeric(df[ALVO].astype(str).str.replace('.', '').str.replace(',', '.'), errors='coerce')
    df = df.dropna(subset=[ALVO])
    df = df[df[ALVO] > 0.0001].reset_index(drop=True)
    
    if len(df) < 5: sys.exit("ERRO: Poucas amostras válidas.")

    cols_candidatas = [c for c in PREDITORES_FIXOS if c in df.columns]
    cols_validas = []
    for c in cols_candidatas:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        if df[c].notna().sum() > (0.5 * len(df)): cols_validas.append(c)
    
    if not cols_validas: sys.exit("ERRO: Nenhuma variável restou após filtro de 50%.")
    df_full = df[cols_validas + [ALVO]].dropna().reset_index(drop=True)
    
    if len(df_full) < 5: sys.exit("ERRO: Poucas amostras completas.")

    X_base = df_full[cols_validas]
    y_base = df_full[ALVO].values

    # === FASE 1: TORNEIO DE OUTLIERS (Mantido com PLS base) ===
    print("\n=== FASE 1: LIMPEZA DE DADOS (OUTLIERS) ===")
    _, sigma_vencedor, mascara_vencedora = rodar_otimizacao_sigma("Base Limpa", X_base, y_base)
    
    X_clean = X_base[mascara_vencedora].reset_index(drop=True)
    y_clean = y_base[mascara_vencedora]
    print(f"\n-> Dados Limpos para Otimização: {len(y_clean)} amostras")

    # === FASE 2: SUPER OPTUNA (PLSR + TARGET TRANSFORM) ===
    print("\n=== FASE 2: BUSCA DE ARQUITETURA PLSR (LOG/SQRT/NONE) ===")
    
    def objective(trial):
        # 1. Transformação do Alvo (Target)
        transform_name = trial.suggest_categorical('target_transform', ['log', 'sqrt', 'none'])
        
        # Aplica transformação temporária
        if transform_name == 'log':
            y_trial = np.log(y_clean)
        elif transform_name == 'sqrt':
            y_trial = np.sqrt(y_clean)
        else:
            y_trial = y_clean
            
        # Normalizar o Y transformado (Z-score) para comparar erros de escalas diferentes
        mean_y_t = np.mean(y_trial)
        std_y_t  = np.std(y_trial)
        if std_y_t == 0: std_y_t = 1
        y_trial_norm = (y_trial - mean_y_t) / std_y_t
        
        # 2. Configuração Exclusiva do PLSR
        try:
            max_c = min(MAX_COMPONENTES, X_clean.shape[1], X_clean.shape[0]-1)
            n_comp = trial.suggest_int('pls_n_comp', 1, max_c)
            
            model = PLSRegression(n_components=n_comp, scale=True, max_iter=2000)
                
            # 3. Cross Validation (Leave-One-Out)
            loo = LeaveOneOut()
            y_pred_cv_norm = cross_val_predict(model, X_clean, y_trial_norm, cv=loo, n_jobs=-1)
            
            erro = mean_squared_error(y_trial_norm, y_pred_cv_norm)
            return erro
            
        except Exception as e:
            return float('inf')

    # Roda Otimização Focada
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50, show_progress_bar=True) # 50 trials é mais que suficiente para PLS
    
    best = study.best_params
    print("\n-> MELHOR ARQUITETURA ENCONTRADA:")
    print(f"   Transformação Alvo: {best['target_transform'].upper()}")
    print(f"   Componentes PLS: {best['pls_n_comp']}")

    # === TREINAMENTO FINAL DO PLSR VENCEDOR ===
    
    # 1. Recria Transformação do Y
    if best['target_transform'] == 'log':
        y_final_trans = np.log(y_clean)
    elif best['target_transform'] == 'sqrt':
        y_final_trans = np.sqrt(y_clean)
    else:
        y_final_trans = y_clean
        
    mean_yf = np.mean(y_final_trans)
    std_yf  = np.std(y_final_trans)
    y_final_norm = (y_final_trans - mean_yf) / std_yf
    
    # 2. Recria Modelo PLS
    model_final = PLSRegression(n_components=best['pls_n_comp'], scale=True, max_iter=2000)
        
    # 3. Predição Final CV
    y_pred_norm_cv = cross_val_predict(model_final, X_clean, y_final_norm, cv=LeaveOneOut(), n_jobs=-1)
    
    # 4. Destransformação Completa
    y_pred_trans = (y_pred_norm_cv * std_yf) + mean_yf
    
    if best['target_transform'] == 'log':
        y_pred_final = np.exp(y_pred_trans)
    elif best['target_transform'] == 'sqrt':
        y_pred_final = y_pred_trans ** 2
    else:
        y_pred_final = y_pred_trans
        
    # Métricas Finais
    r2_final = r2_score(y_clean, y_pred_final)
    rmse_final = np.sqrt(mean_squared_error(y_clean, y_pred_final))
    
    print(f"\n=== RESULTADO FINAL ===\n R² = {r2_final:.4f} | RMSE = {rmse_final:.2f}")

    # === GRÁFICOS ORIGINAIS ===
    fig = plt.figure(figsize=(18, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
    
    # Plot 1: Observado vs Predito
    ax1 = fig.add_subplot(gs[0])
    ax1.scatter(y_clean, y_pred_final, c='#3498db', edgecolors='k', s=80, alpha=0.7, label='Predição Final')
    
    min_v = min(y_clean.min(), y_pred_final.min()) * 0.9
    max_v = max(y_clean.max(), y_pred_final.max()) * 1.1
    ax1.plot([min_v, max_v], [min_v, max_v], 'r--', lw=2)
    ax1.set_xlabel('Observado (Reais)')
    ax1.set_ylabel('Predito (Cross-Validation)')
    ax1.set_title(f"Modelo: PLSR ({best['target_transform']})\nComponentes={best['pls_n_comp']} | R²={r2_final:.3f} | RMSE={rmse_final:.2f}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Importância das Variáveis (PLS Coefs)
    ax2 = fig.add_subplot(gs[1])
    
    model_final.fit(X_clean, y_final_norm) 
    coefs = np.abs(model_final.coef_).ravel()
    if coefs.sum() > 0: coefs = coefs / coefs.sum()
    indices = np.argsort(coefs)
    if len(indices) > 15: indices = indices[-15:]
    
    ax2.barh(range(len(indices)), coefs[indices], color='#2ecc71', edgecolor='k')
    ax2.set_yticks(range(len(indices)))
    ax2.set_yticklabels(np.array(X_clean.columns)[indices])
    ax2.set_title('Importância das Variáveis (Coeficientes PLS Normalizados)')

    plt.tight_layout()
    plt.savefig('resultado_super_modelo_plsr.png', dpi=150)
    print("-> Gráfico Salvo: resultado_super_modelo_plsr.png")

    # === GRÁFICOS: SCATTER ÍNDICES vs KILOS ===
    print("\n-> Gerando scatter plots dos índices vs Kilos...")

    n_indices = len(cols_validas)
    n_cols_scatter = 4
    n_rows_scatter = int(np.ceil(n_indices / n_cols_scatter))

    fig2, axes = plt.subplots(
        n_rows_scatter, n_cols_scatter,
        figsize=(n_cols_scatter * 4.5, n_rows_scatter * 4),
        constrained_layout=True
    )
    axes = np.array(axes).ravel()

    # Paleta de cores ordenada por correlação (será calculada abaixo)
    cmap = plt.get_cmap('RdYlGn')

    # Calcula correlações para colorir os títulos
    correlacoes = {}
    for col in cols_validas:
        x_vals = X_clean[col].values
        try:
            r, p = pearsonr(x_vals, y_clean)
            correlacoes[col] = (r, p)
        except:
            correlacoes[col] = (0.0, 1.0)

    for i, col in enumerate(cols_validas):
        ax = axes[i]
        x_vals = X_clean[col].values
        r, p = correlacoes[col]

        # Cor dos pontos baseada no r de Pearson (vermelho=negativo, verde=positivo)
        cor_norm = (r + 1) / 2  # normaliza -1..1 para 0..1
        cor = cmap(cor_norm)

        ax.scatter(x_vals, y_clean, color=cor, edgecolors='k', linewidths=0.4,
                   s=55, alpha=0.75)

        # Linha de tendência linear
        try:
            z = np.polyfit(x_vals, y_clean, 1)
            p_fit = np.poly1d(z)
            x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
            ax.plot(x_line, p_fit(x_line), 'r--', lw=1.5, alpha=0.85)
        except:
            pass

        # Símbolo de significância
        if p < 0.001:
            sig = '***'
        elif p < 0.01:
            sig = '**'
        elif p < 0.05:
            sig = '*'
        else:
            sig = 'ns'

        ax.set_title(f"{col}\nr={r:.3f} {sig}", fontsize=9, fontweight='bold',
                     color='#1a1a2e')
        ax.set_xlabel(col, fontsize=8)
        ax.set_ylabel(ALVO, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25, linestyle='--')

    # Desativa subplots extras
    for j in range(n_indices, len(axes)):
        axes[j].set_visible(False)

    fig2.suptitle(
        f'Scatter: Índices Espectrais vs {ALVO}  '
        f'(n={len(y_clean)} | * p<0.05  ** p<0.01  *** p<0.001)',
        fontsize=13, fontweight='bold', y=1.01
    )

    plt.savefig('scatter_indices_vs_kilos.png', dpi=150, bbox_inches='tight')
    print("-> Gráfico Salvo: scatter_indices_vs_kilos.png")

    # CSV Export
    df_out = df_full.iloc[mascara_vencedora].copy()
    df_out['Predito_Final'] = y_pred_final
    df_out['Residuo'] = y_clean - y_pred_final
    df_out.to_csv('resultado_super_modelo_plsr.csv', sep=';', decimal=',', index=False)
    print("-> CSV Salvo: resultado_super_modelo_plsr.csv")
    
    plt.show()