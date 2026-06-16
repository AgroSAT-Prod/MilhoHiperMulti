import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import sys
import warnings

from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ================= CONFIGURAÇÕES =================

ARQUIVO_DADOS = r"C:\Users\muril\OneDrive\Documentos\Experiments-Prod\ProjetoMilho\Biomassa14_01Clorofila.csv"
NOME_ALVO_DESEJADO = 'Kilos'

PREDITORES_FIXOS = [
    'NGRDI', 'MSAVI', 'MGRVI', 'RGBIndex', 'VARI', 'TGI', 'GLI',
    'ExGRaw', 'ExG', 'ExRM', 'ExGR', 'NDVI', 'SAVI', 'RDVI',
    'NLI', 'CVI', 'GNDVI', 'PanNDVI', 'NDRE', 'NDVIRE', 'SFDVI'
]

MAX_COMPONENTES = 20

# ================= FUNÇÕES =================

def detectar_separador(caminho):
    with open(caminho, 'r', encoding='utf-8-sig') as f:
        linha = f.readline()
    for sep in [';', ',', '\t', '|']:
        if sep in linha:
            return sep
    return ','

def detectar_skip_rows(caminho, sep):
    """Detecta quantas linhas pular até encontrar os cabeçalhos corretos."""
    for skip in [0, 1, 2]:
        try:
            df_test = pd.read_csv(caminho, sep=sep, encoding='utf-8-sig',
                                  dtype=str, quotechar='"', skiprows=skip, nrows=0)
            cols = [c.strip().replace('"', '') for c in df_test.columns]
            if any(NOME_ALVO_DESEJADO.lower() in c.lower() for c in cols):
                return skip
        except:
            pass
    return 0

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

# ================= EXECUÇÃO PRINCIPAL =================

if __name__ == "__main__":
    print(f"=== ANÁLISE PLSR BASELINE: '{NOME_ALVO_DESEJADO}' ===\n")

    if not os.path.exists(ARQUIVO_DADOS):
        sys.exit(f"ERRO: Arquivo '{ARQUIVO_DADOS}' não encontrado.")

    sep = detectar_separador(ARQUIVO_DADOS)
    skip = detectar_skip_rows(ARQUIVO_DADOS, sep)
    print(f"-> Separador: '{sep}' | Linhas de cabeçalho ignoradas: {skip}")

    try:
        # Lê como string para tratar aspas e formatação BR manualmente
        df = pd.read_csv(ARQUIVO_DADOS, sep=sep, encoding='utf-8-sig',
                         dtype=str, quotechar='"', skiprows=skip)
        df.columns = df.columns.str.strip().str.replace('"', '', regex=False)
        
        # Converte todas as colunas: remove aspas, troca ponto de milhar e vírgula decimal
        for col in df.columns:
            s = df[col].astype(str).str.strip().str.replace('"', '', regex=False)
            s = s.str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(s, errors='coerce')

    except Exception as e:
        sys.exit(f"Erro ao ler CSV: {e}")

    print(f"-> Total linhas no CSV: {len(df)}")
    print(f"-> Colunas: {list(df.columns)}")

    # Encontra coluna alvo
    cols_alvo = [c for c in df.columns if NOME_ALVO_DESEJADO.lower() in c.lower()]
    if not cols_alvo:
        sys.exit(f"ERRO: Coluna '{NOME_ALVO_DESEJADO}' não encontrada. Disponíveis: {list(df.columns)}")
    ALVO = cols_alvo[0]

    df = df.dropna(subset=[ALVO])
    df = df[df[ALVO] > 0.0001].reset_index(drop=True)

    if len(df) < 5:
        sys.exit("ERRO: Poucas amostras válidas.")

    cols_candidatas = [c for c in PREDITORES_FIXOS if c in df.columns]
    cols_validas = [c for c in cols_candidatas if df[c].notna().sum() > (0.5 * len(df))]

    if not cols_validas:
        sys.exit(f"ERRO: Nenhuma variável preditora válida. Disponíveis: {list(df.columns)}")

    df_full = df[cols_validas + [ALVO]].dropna().reset_index(drop=True)

    if len(df_full) < 5:
        sys.exit("ERRO: Poucas amostras completas.")

    X = df_full[cols_validas]
    y = df_full[ALVO].values

    print(f"-> Amostras: {len(y)} | Preditores ({len(cols_validas)}): {cols_validas}")

    # === SELEÇÃO DE COMPONENTES ===
    print("\n-> Selecionando número ótimo de componentes via LOO-CV...")
    y_log = np.log(y)
    mean_log, std_log = np.mean(y_log), np.std(y_log)
    if std_log == 0: std_log = 1
    y_norm = (y_log - mean_log) / std_log

    n_comp_otimo, _ = escolher_n_componentes(X, y_norm, MAX_COMPONENTES)
    print(f"-> Componentes selecionados: {n_comp_otimo}")

    # === MODELO FINAL LOO-CV ===
    model = PLSRegression(n_components=n_comp_otimo, scale=True, max_iter=2000)
    y_pred_norm_cv = cross_val_predict(model, X, y_norm, cv=LeaveOneOut(), n_jobs=-1).ravel()
    y_pred_final = np.exp((y_pred_norm_cv * std_log) + mean_log)

    r2_final   = r2_score(y, y_pred_final)
    rmse_final = np.sqrt(mean_squared_error(y, y_pred_final))

    print(f"\n=== RESULTADO FINAL ===")
    print(f" R²   = {r2_final:.4f}")
    print(f" RMSE = {rmse_final:.4f}")

    # === GRÁFICOS ===
    fig = plt.figure(figsize=(16, 6))
    gs = gridspec.GridSpec(1, 2)

    ax1 = fig.add_subplot(gs[0])
    ax1.scatter(y, y_pred_final, c='#3498db', edgecolors='k', s=80, alpha=0.7)
    min_v = min(y.min(), y_pred_final.min()) * 0.9
    max_v = max(y.max(), y_pred_final.max()) * 1.1
    ax1.plot([min_v, max_v], [min_v, max_v], 'r--', lw=2)
    ax1.set_xlabel('Observado (Kilos)')
    ax1.set_ylabel('Predito (LOO-CV)')
    ax1.set_title(f"PLSR Baseline\nComponentes={n_comp_otimo} | R²={r2_final:.3f} | RMSE={rmse_final:.4f}")
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[1])
    model.fit(X, y_norm)
    coefs = np.abs(model.coef_).ravel()
    if coefs.sum() > 0: coefs = coefs / coefs.sum()
    indices = np.argsort(coefs)
    if len(indices) > 15: indices = indices[-15:]
    ax2.barh(range(len(indices)), coefs[indices], color='#2ecc71', edgecolor='k')
    ax2.set_yticks(range(len(indices)))
    ax2.set_yticklabels(np.array(X.columns)[indices])
    ax2.set_title('Importância das Variáveis (Coeficientes PLS Normalizados)')

    plt.tight_layout()
    plt.savefig('resultado_plsr_baseline.png', dpi=150)
    print("-> Gráfico salvo: resultado_plsr_baseline.png")

    df_out = df_full.copy()
    df_out['Predito_Final'] = y_pred_final
    df_out['Residuo'] = y - y_pred_final
    df_out.to_csv('resultado_plsr_baseline.csv', sep=';', decimal=',', index=False)
    print("-> CSV salvo: resultado_plsr_baseline.csv")

    plt.show()