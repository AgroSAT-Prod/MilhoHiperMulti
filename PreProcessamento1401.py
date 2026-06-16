import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os
import sys
import glob
import unicodedata

# ================= CONFIGURAÇÕES =================
PASTA_DO_SCRIPT = os.path.dirname(os.path.abspath(__file__))

# PADRÃO DE BUSCA (Ajuste conforme necessário)
# Se quiser ler a pasta toda: os.path.join(PASTA_DO_SCRIPT, '*.csv')
# Se quiser o arquivo específico:
PADRAO_BUSCA = os.path.join(r'C:\Users\muril\OneDrive\Documentos\Experiments-Prod\ProjetoMilho\leituras_hiperspectrais_COMPLETO.csv') 

ARQUIVO_AGRONOMICO = os.path.join(PASTA_DO_SCRIPT, 'Dataset', 'PlanilhaFiltrada2.csv')
NOME_ARQUIVO_SAIDA = 'DATASET_IA_PROCESSADO_JAN2026.csv'

# Parâmetros de Processamento
TRIM_MIN = 420.0  
TRIM_MAX = 950.0  
SG_WINDOW = 11    
SG_POLY = 2       
SG_DERIV = 1      

# Parâmetros de Outliers
REMOVER_OUTLIERS = True
FATOR_IQR = 1.5  # 1.5 é o padrão estatístico (Tukey)

# ================= FUNÇÕES AUXILIARES =================

def extrair_wavelengths(cols):
    wls = []
    for c in cols:
        try:
            txt = c.replace('d1_', '').replace('Band_', '').replace('nm', '')
            val = float(txt)
            wls.append(val)
        except: pass
    return wls

def normalizar_texto(texto):
    if not isinstance(texto, str): return str(texto)
    nfkd = unicodedata.normalize('NFKD', texto)
    return "".join([c for c in nfkd if not unicodedata.combining(c)]).lower().strip()

def limpar_float(dados):
    if isinstance(dados, pd.Series):
        return pd.to_numeric(dados.astype(str).str.replace(',', '.'), errors='coerce')
    try:
        if isinstance(dados, str): return float(dados.replace(',', '.'))
        return float(dados)
    except: return np.nan

def identificar_colunas_chave(df):
    cols_norm = {normalizar_texto(c): c for c in df.columns}
    col_id = None
    for p in ['id_amostra', 'id', 'sample', 'amostra']:
        if p in cols_norm: col_id = cols_norm[p]; break
    col_parte = None
    for p in ['parte_planta', 'parte', 'part', 'terco']:
        if p in cols_norm: col_parte = cols_norm[p]; break
    return col_id, col_parte

# ================= FUNÇÕES DE LIMPEZA ESTATÍSTICA =================

def remover_outliers_iqr(df, coluna_valor, coluna_grupo):
    """
    Remove outliers usando o método IQR (Interquartile Range) agrupado por Tratamento.
    Isso evita remover valores altos que são reais devido ao tratamento.
    """
    df_limpo = df.copy()
    indices_outliers = []
    
    # Remove NaNs antes de calcular para não dar erro
    df_valid = df_limpo.dropna(subset=[coluna_valor])
    
    grupos = df_valid.groupby(coluna_grupo)
    
    print(f"\n   > Analisando outliers por grupo ('{coluna_grupo}')...")
    
    for nome, grupo in grupos:
        q1 = grupo[coluna_valor].quantile(0.25)
        q3 = grupo[coluna_valor].quantile(0.75)
        iqr = q3 - q1
        
        limite_inf = q1 - (FATOR_IQR * iqr)
        limite_sup = q3 + (FATOR_IQR * iqr)
        
        # Filtra quem está fora dos limites
        outliers_grupo = grupo[(grupo[coluna_valor] < limite_inf) | (grupo[coluna_valor] > limite_sup)]
        
        if not outliers_grupo.empty:
            print(f"     - Grupo '{nome}': {len(outliers_grupo)} outliers removidos (Limites: {limite_inf:.2f} a {limite_sup:.2f})")
            indices_outliers.extend(outliers_grupo.index.tolist())
            
    # Remove as linhas do dataframe original
    if indices_outliers:
        df_limpo = df_limpo.drop(indices_outliers)
    
    return df_limpo, len(indices_outliers)

# ================= PROCESSAMENTO ESPECTRAL =================

def cortar_extremidades(dataframe, cols_bandas, min_wl, max_wl):
    print(f"   > [1/3] Cortando extremidades ({min_wl}-{max_wl}nm)...")
    cols_para_manter = []
    for col in cols_bandas:
        try:
            txt_num = col.replace('d1_', '').replace('Band_', '').replace('nm', '')
            wl = float(txt_num)
            if min_wl <= wl <= max_wl: cols_para_manter.append(col)
        except: pass
    if not cols_para_manter:
        print("   ⚠️  AVISO: Nenhuma banda no intervalo. Mantendo originais.")
        return dataframe[cols_bandas]
    return dataframe[cols_para_manter]

def aplicar_snv(dataframe, cols_bandas):
    print("   > [2/3] Aplicando SNV (Normalização)...")
    X = dataframe[cols_bandas].values
    if X.shape[1] == 0: return dataframe[cols_bandas]
    media = X.mean(axis=1, keepdims=True)
    desvio = X.std(axis=1, keepdims=True)
    X_snv = (X - media) / (desvio + 1e-8)
    return pd.DataFrame(X_snv, columns=cols_bandas)

def aplicar_savgol(dataframe, cols_bandas):
    print(f"   > [3/3] Aplicando Savitzky-Golay (Win={SG_WINDOW}, Deriv={SG_DERIV})...")
    if len(cols_bandas) < SG_WINDOW:
        print(f"   ⚠️  PULANDO SG: Bandas insuficientes.")
        return dataframe[cols_bandas]
    X = dataframe[cols_bandas].values
    try:
        X_proc = savgol_filter(X, window_length=SG_WINDOW, polyorder=SG_POLY, deriv=SG_DERIV)
        novas_cols = [c if c.startswith('d1_') else f"d1_{c}" for c in cols_bandas]
        return pd.DataFrame(X_proc, columns=novas_cols)
    except: return dataframe[cols_bandas]

def plotar_comparacao_4_paineis(dados_dict, wls_dict):
    """Gera painel 2x2 com as 4 etapas do processamento."""
    print("\n   > Gerando gráficos de 4 etapas...")
    plt.figure(figsize=(14, 10))
    plt.suptitle("Evolução do Processamento Espectral (4 Etapas)", fontsize=16)
    
    chaves = list(dados_dict.keys())
    posicoes = [1, 2, 3, 4] 
    
    for i in range(min(4, len(chaves))):
        key = chaves[i]
        ax = plt.subplot(2, 2, posicoes[i])
        df_plot = dados_dict[key]
        wls = wls_dict[key]
        
        if df_plot is not None and not df_plot.empty and len(wls) == df_plot.shape[1]:
            # Plota amostra (máx 50 linhas para não pesar)
            X = df_plot.iloc[np.random.choice(len(df_plot), min(50, len(df_plot)), replace=False)].values
            ax.plot(wls, X.T, alpha=0.5, linewidth=0.8)
            
        ax.set_title(key, fontweight='bold')
        ax.set_xlabel('Comprimento de Onda (nm)')
        ax.grid(True, alpha=0.3)
        
        if "Derivada" in key: ax.set_ylabel("1ª Derivada")
        elif "SNV" in key: ax.set_ylabel("Reflectância Norm.")
        else: ax.set_ylabel("Reflectância")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    nome_img = 'processamento_espectral_4steps.png'
    plt.savefig(nome_img, dpi=300)
    print(f"   > Gráfico salvo como: {nome_img}")
    print("   > Exibindo na tela...")
    plt.show()

# ================= EXECUÇÃO PRINCIPAL =================

if __name__ == "__main__":
    print("=== INICIANDO PROCESSAMENTO (COM REMOÇÃO DE OUTLIERS) ===")
    
    # 1. CARREGAMENTO
    todos_csvs = glob.glob(PADRAO_BUSCA)
    lista_dfs = []
    print(f"1. Buscando arquivos...")
    for arq in todos_csvs:
        base_name = os.path.basename(arq)
        if (base_name != NOME_ARQUIVO_SAIDA) and ("PlanilhaFiltrada" not in base_name) and ("_VALIDO" not in base_name) and (".png" not in base_name):
            try:
                df_temp = pd.read_csv(arq, sep=';', decimal=',')
                if len(df_temp.columns) < 5: df_temp = pd.read_csv(arq, sep=',', decimal='.')
                lista_dfs.append(df_temp)
                print(f"   Ok: {base_name}")
            except: pass

    if not lista_dfs: sys.exit("ERRO: Nenhum CSV encontrado.")
    df_raw = pd.concat(lista_dfs, ignore_index=True)
    
    col_id_img, col_parte_img = identificar_colunas_chave(df_raw)
    if not col_id_img: col_id_img = df_raw.columns[0]
    if not col_parte_img: col_parte_img = 'Parte' if 'Parte' in df_raw.columns else 'Parte_Planta'
    
    print(f"   > Colunas Imagem -> ID: '{col_id_img}' | Parte: '{col_parte_img}'")

    # 2. PIPELINE ESPECTRAL
    cols_bandas_raw = [c for c in df_raw.columns if "Band_" in c or ("d1_" in c and "Band" not in c)]
    if not cols_bandas_raw: 
        cols_bandas_raw = [c for c in df_raw.columns if c.replace('.', '', 1).isdigit() or c.startswith('d1_')]

    cols_meta = [c for c in df_raw.columns if c not in cols_bandas_raw]
    
    # --- ETAPAS DO PROCESSAMENTO ---
    # 1. Raw
    df_raw_bands = df_raw[cols_bandas_raw]
    
    # 2. Trimmed
    df_trimmed = cortar_extremidades(df_raw, cols_bandas_raw, TRIM_MIN, TRIM_MAX)
    cols_validas = df_trimmed.columns.tolist()
    
    # 3. SNV
    df_snv = aplicar_snv(df_trimmed, cols_validas)
    
    # 4. Savitzky-Golay
    df_sg = aplicar_savgol(df_snv, cols_validas)
    
    # --- VISUALIZAÇÃO (4 GRÁFICOS) ---
    dados_plot = {
        '1. Dados Brutos (Raw)': df_raw_bands,
        '2. Corte Espectral': df_trimmed,
        '3. Normalização (SNV)': df_snv,
        '4. Savitzky-Golay (1ª Derivada)': df_sg
    }
    
    wls_plot = {
        '1. Dados Brutos (Raw)': extrair_wavelengths(df_raw_bands.columns),
        '2. Corte Espectral': extrair_wavelengths(df_trimmed.columns),
        '3. Normalização (SNV)': extrair_wavelengths(df_snv.columns),
        '4. Savitzky-Golay (1ª Derivada)': extrair_wavelengths(df_sg.columns)
    }
    
    # Chama explicitamente a função de 4 painéis
    plotar_comparacao_4_paineis(dados_plot, wls_plot)

    # Prepara DF Final
    df_espectral_proc = pd.concat([df_raw[cols_meta], df_sg], axis=1)
    df_espectral_proc['ID_Numeric'] = pd.to_numeric(df_espectral_proc[col_id_img], errors='coerce')

    # 3. LEITURA AGRONÔMICA
    print(f"\n2. Lendo CSV Agronômico...")
    try:
        df_agro = pd.read_csv(ARQUIVO_AGRONOMICO, header=3, sep=';', decimal=',', encoding='latin1')
    except: sys.exit("Erro fatal ao ler CSV agronômico.")

    cols_orig = list(df_agro.columns)
    cols_norm = [normalizar_texto(c) for c in cols_orig]
    
    idx_id = next((i for i, c in enumerate(cols_norm) if c == 'id'), -1)
    if idx_id == -1: idx_id = next((i for i, c in enumerate(cols_norm) if 'id' in c and 'parcela' in c), -1)
    col_id_agro = cols_orig[idx_id] if idx_id != -1 else cols_orig[0]
    
    idx_trat = next((i for i, c in enumerate(cols_norm) if 'tratamento' in c), -1)
    idx_dose = next((i for i, c in enumerate(cols_norm) if 'dose' in c), -1)
    col_trat_agro = cols_orig[idx_trat] if idx_trat != -1 else None
    col_dose_agro = cols_orig[idx_dose] if idx_dose != -1 else None

    cols_amostra = [c for c in cols_orig if 'amostra' in str(c).lower()]
    cols_medio = cols_amostra[0:2] if len(cols_amostra) >= 2 else []
    cols_sup = cols_amostra[2:4] if len(cols_amostra) >= 4 else []
    
    print(f"   > Colunas Agro -> ID: {col_id_agro} | Trat: {col_trat_agro} | Dose: {col_dose_agro}")

    print("\n3. Realizando Cruzamento...")
    df_agro['ID_Numeric_Agro'] = pd.to_numeric(df_agro[col_id_agro], errors='coerce')
    df_agro = df_agro.dropna(subset=['ID_Numeric_Agro'])

    df_final = pd.merge(df_espectral_proc, df_agro, left_on='ID_Numeric', right_on='ID_Numeric_Agro', how='left')

    val_m = df_final[cols_medio].apply(limpar_float).mean(axis=1) if cols_medio else np.nan
    val_s = df_final[cols_sup].apply(limpar_float).mean(axis=1) if cols_sup else np.nan
    
    col_p = df_final[col_parte_img].astype(str).str.strip().str.upper()
    df_final['Y_Clorofila'] = np.select([col_p == 'M', col_p == 'S'], [val_m, val_s], default=np.nan)
    
    if col_trat_agro: df_final['Tratamento'] = df_final[col_trat_agro]
    if col_dose_agro:
        df_final['Dose_N'] = pd.to_numeric(
            df_final[col_dose_agro].astype(str).str.replace(',', '.'), 
            errors='coerce'
        )
    
    df_final.rename(columns={col_id_img: 'ID', col_parte_img: 'Parte'}, inplace=True)

    # ================= 4. REMOÇÃO DE OUTLIERS =================
    if REMOVER_OUTLIERS and 'Tratamento' in df_final.columns:
        print("\n4. Removendo Outliers (IQR Method)...")
        print(f"   Total antes: {len(df_final)}")
        
        # Filtra apenas linhas que têm clorofila válida para análise
        df_com_cloro = df_final.dropna(subset=['Y_Clorofila'])
        df_sem_cloro = df_final[df_final['Y_Clorofila'].isna()]
        
        # Remove outliers apenas no dataset válido
        df_limpo, n_removidos = remover_outliers_iqr(df_com_cloro, 'Y_Clorofila', 'Tratamento')
        
        # Reintegra (mantemos as linhas sem clorofila pois podem ser úteis para outra coisa, 
        # ou se quiser descartar, basta não concatenar df_sem_cloro)
        df_final = pd.concat([df_limpo, df_sem_cloro], ignore_index=True)
        
        print(f"   Total removido: {n_removidos}")
        print(f"   Total final: {len(df_final)}")
    else:
        print("\n4. Pulo Remoção de Outliers (Tratamento não encontrado ou desativado)")

    # 5. SALVAMENTO
    cols_fixas = ['ID', 'Parte', 'Tratamento', 'Dose_N', 'Y_Clorofila']
    cols_espectro = list(df_sg.columns)
    cols_export = [c for c in cols_fixas if c in df_final.columns] + cols_espectro
    df_export = df_final[cols_export]
    
    if len(df_export) > 0:
        df_export.to_csv(NOME_ARQUIVO_SAIDA, index=False, sep=';', decimal=',')
        print(f"\n{'='*60}")
        print(f" SUCESSO! Arquivo gerado: {NOME_ARQUIVO_SAIDA}")
        
        validos = df_export.dropna(subset=['Y_Clorofila'])
        arq_valid = NOME_ARQUIVO_SAIDA.replace('.csv', '_VALIDO.csv')
        validos.to_csv(arq_valid, index=False, sep=';', decimal=',')
        print(f" Arquivo limpo salvo: {arq_valid}")
        print(f"{'='*60}\n")
    else:
        print("ERRO: Dataset vazio.")