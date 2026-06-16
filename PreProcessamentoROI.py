import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os
import sys
import glob

# ================= CONFIGURAÇÕES =================
PASTA_DO_SCRIPT = os.path.dirname(os.path.abspath(__file__))

PADRAO_BUSCA = os.path.join(PASTA_DO_SCRIPT, '*.csv')
ARQUIVO_AGRONOMICO = os.path.join(PASTA_DO_SCRIPT, 'Dataset', 'hiperespectral_ROIs.csv')
NOME_ARQUIVO_SAIDA = 'DATASET_IA_PROCESSADOROI.csv'

# Processamento espectral
TRIM_MIN = 420.0
TRIM_MAX = 950.0
SG_WINDOW = 11
SG_POLY = 2
SG_DERIV = 1

# ================= FUNÇÕES =================

def extrair_wavelengths(cols):
    wls = []
    for c in cols:
        try:
            wls.append(float(c.replace('Band_', '').replace('d1_Band_', '').replace('nm', '')))
        except:
            pass
    return wls

def cortar_extremidades(df, cols, wl_min, wl_max):
    print(f"   > [1/3] Corte espectral ({wl_min}-{wl_max} nm)")
    cols_ok = []
    for c in cols:
        try:
            wl = float(c.replace('Band_', '').replace('nm', ''))
            if wl_min <= wl <= wl_max:
                cols_ok.append(c)
        except:
            pass
    return df[cols_ok]

def aplicar_snv(df, cols):
    print("   > [2/3] SNV")
    X = df[cols].values
    media = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    X_snv = (X - media) / (std + 1e-8)
    return pd.DataFrame(X_snv, columns=cols)

def aplicar_savgol(df, cols):
    print(f"   > [3/3] Savitzky-Golay (deriv={SG_DERIV})")
    X = df[cols].values
    X_sg = savgol_filter(X, window_length=SG_WINDOW, polyorder=SG_POLY, deriv=SG_DERIV)
    return pd.DataFrame(X_sg, columns=[f"d1_{c}" for c in cols])

def plotar_comparacao(dados, wls):
    plt.figure(figsize=(14, 10))
    plt.suptitle("Evolução do Pré-Processamento Espectral", fontsize=16)

    for i, key in enumerate(dados.keys()):
        ax = plt.subplot(2, 2, i + 1)
        df_plot = dados[key]
        wl = wls[key]

        if len(df_plot) > 30:
            idx = np.random.choice(len(df_plot), 30, replace=False)
            X = df_plot.iloc[idx].values
        else:
            X = df_plot.values

        ax.plot(wl, X.T, alpha=0.6)
        ax.set_title(key)
        ax.set_xlabel("Comprimento de onda (nm)")
        ax.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# ================= EXECUÇÃO =================

if __name__ == "__main__":
    print("=== INICIANDO PIPELINE FINAL (ROIs) ===")

    # -------- 1. LEITURA DOS CSVs --------
    csvs = glob.glob(PADRAO_BUSCA)
    dfs = []

    print("1. Lendo CSVs espectrais...")
    for arq in csvs:
        if NOME_ARQUIVO_SAIDA in arq or 'Planilha' in arq:
            continue
        try:
            df = pd.read_csv(arq, sep=';', decimal=',')
            if len(df.columns) < 5:
                df = pd.read_csv(arq, sep=',', decimal='.')
            dfs.append(df)
            print(f"   OK: {os.path.basename(arq)}")
        except Exception as e:
            print(f"   ERRO: {arq} -> {e}")

    if not dfs:
        sys.exit("ERRO: Nenhum CSV válido encontrado.")

    df_raw = pd.concat(dfs, ignore_index=True)
    print(f"   > Total de amostras (ROIs): {len(df_raw)}")

    # -------- 2. PROCESSAMENTO ESPECTRAL --------
    cols_bandas = [c for c in df_raw.columns if c.startswith("Band_")]
    cols_meta = [c for c in df_raw.columns if c not in cols_bandas]

    estados = {}
    wls = {}

    estados['1. Raw'] = df_raw[cols_bandas]
    wls['1. Raw'] = extrair_wavelengths(cols_bandas)

    df_trim = cortar_extremidades(df_raw, cols_bandas, TRIM_MIN, TRIM_MAX)
    cols_trim = df_trim.columns.tolist()
    estados['2. Corte'] = df_trim
    wls['2. Corte'] = extrair_wavelengths(cols_trim)

    df_snv = aplicar_snv(df_trim, cols_trim)
    estados['3. SNV'] = df_snv
    wls['3. SNV'] = extrair_wavelengths(cols_trim)

    df_sg = aplicar_savgol(df_snv, cols_trim)
    estados['4. SG'] = df_sg
    wls['4. SG'] = extrair_wavelengths(cols_trim)

    plotar_comparacao(estados, wls)

    # -------- 3. LEITURA DO EXCEL --------
    print("\n2. Lendo CSV agronômico...")
    try:
        df_agro = pd.read_csv(ARQUIVO_AGRONOMICO, sep=';', decimal=',')

        # normaliza nomes de colunas
        df_agro.columns = [str(c).strip().lower() for c in df_agro.columns]

        # ajuste conforme os nomes reais no CSV
        col_id = 'id_amostra' if 'id_amostra' in df_agro.columns else 'id'
        col_dose = next(c for c in df_agro.columns if 'dose' in c)
        col_clor = next(c for c in df_agro.columns if 'clorofila' in c)

        print(f"   > Colunas detectadas: {df_agro.columns.tolist()}")

    except Exception as e:
        sys.exit(f"Erro ao ler CSV agronômico: {e}")


    # -------- 4. UNIFICAÇÃO FINAL --------
    print("\n3. Cruzando dados...")
    df_full = pd.concat([df_raw[cols_meta], df_sg], axis=1)

    dataset = []
    erros = 0

    for _, row in df_full.iterrows():
        try:
            id_img = int(row['ID_Amostra'])
            parte = str(row['Parte']).strip().upper()

            match = df_agro[df_agro[col_id] == id_img]
            if match.empty:
                erros += 1
                continue

            agro = match.iloc[0]
            y = agro[col_clor_m] if parte == 'M' else agro[col_clor_s]

            linha = {
                'ID': id_img,
                'Parte': parte,
                'Dose_N': agro[col_dose],
                'Y_Clorofila': float(y) if pd.notna(y) else np.nan
            }

            for c in df_sg.columns:
                linha[c] = row[c]

            dataset.append(linha)

        except:
            erros += 1

    # -------- SALVAR --------
    if not dataset:
        sys.exit("ERRO: dataset final vazio.")

    df_final = pd.DataFrame(dataset)
    df_final.to_csv(NOME_ARQUIVO_SAIDA, index=False, sep=';', decimal=',')

    print(f"\n✅ FINALIZADO COM SUCESSO")
    print(f"Amostras finais: {len(df_final)}")
    print(f"IDs não cruzados: {erros}")
    print(f"Arquivo salvo: {NOME_ARQUIVO_SAIDA}")
