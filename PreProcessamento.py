import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os
import sys
import glob

# ================= CONFIGURAÇÕES =================
PASTA_DO_SCRIPT = os.path.dirname(os.path.abspath(__file__))

# Busca todos os CSVs na pasta (Múltiplos arquivos)
PADRAO_BUSCA = os.path.join(PASTA_DO_SCRIPT, '*.csv') 
ARQUIVO_AGRONOMICO = os.path.join(PASTA_DO_SCRIPT, 'Dataset', 'PlanilhaFiltrada.xlsx')
NOME_ARQUIVO_SAIDA = 'DATASET_IA_PROCESSADOROI.csv'

# Parâmetros de Processamento Espectral
TRIM_MIN = 420.0  
TRIM_MAX = 950.0  
SG_WINDOW = 11    
SG_POLY = 2       
SG_DERIV = 1      

# ================= FUNÇÕES DE PROCESSAMENTO =================

def extrair_wavelengths(cols):
    wls = []
    for c in cols:
        try:
            val = float(c.replace('Band_', '').replace('d1_Band_', '').replace('nm', ''))
            wls.append(val)
        except: pass
    return wls

def cortar_extremidades(dataframe, cols_bandas, min_wl, max_wl):
    print(f"   > [1/3] Cortando extremidades ({min_wl}-{max_wl}nm)...")
    cols_para_manter = []
    for col in cols_bandas:
        try:
            wl = float(col.replace('Band_', '').replace('nm', ''))
            if min_wl <= wl <= max_wl:
                cols_para_manter.append(col)
        except: pass
    return dataframe[cols_para_manter]

def aplicar_snv(dataframe, cols_bandas):
    print("   > [2/3] Aplicando SNV (Normalização)...")
    X = dataframe[cols_bandas].values
    media = X.mean(axis=1, keepdims=True)
    desvio = X.std(axis=1, keepdims=True)
    X_snv = (X - media) / (desvio + 1e-8)
    return pd.DataFrame(X_snv, columns=cols_bandas)

def aplicar_savgol(dataframe, cols_bandas):
    print(f"   > [3/3] Aplicando Savitzky-Golay (Win={SG_WINDOW}, Deriv={SG_DERIV})...")
    X = dataframe[cols_bandas].values
    X_proc = savgol_filter(X, window_length=SG_WINDOW, polyorder=SG_POLY, deriv=SG_DERIV)
    novas_cols = [f"d1_{c}" for c in cols_bandas]
    return pd.DataFrame(X_proc, columns=novas_cols)

def plotar_comparacao(dados_dict, wls_dict):
    """Gera o painel de 4 gráficos para validação visual."""
    plt.figure(figsize=(14, 10))
    plt.suptitle("Evolução do Pré-Processamento Espectral", fontsize=16)
    
    posicoes = [1, 2, 3, 4]
    chaves = list(dados_dict.keys())
    
    for i, key in enumerate(chaves):
        ax = plt.subplot(2, 2, posicoes[i])
        df_plot = dados_dict[key]
        wls = wls_dict[key]
        
        # Plota até 30 amostras aleatórias para não pesar a visualização
        if len(df_plot) > 30:
            indices = np.random.choice(len(df_plot), 30, replace=False)
            X = df_plot.iloc[indices].values
        else:
            X = df_plot.values
        
        ax.plot(wls, X.T, alpha=0.6, linewidth=1)
        ax.set_title(key, fontsize=12, fontweight='bold')
        ax.set_xlabel("Comprimento de Onda (nm)")
        ax.grid(True, alpha=0.3)
        
        if "Derivada" in key: ax.set_ylabel("1ª Derivada")
        elif "SNV" in key: ax.set_ylabel("Reflectância Norm.")
        else: ax.set_ylabel("Reflectância Bruta")
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    print("\n   > Exibindo gráficos comparativos...")
    plt.show()

# ================= EXECUÇÃO PRINCIPAL =================

if __name__ == "__main__":
    print("=== INICIANDO PROCESSAMENTO COMPLETO (FINAL) ===")
    
    # ---------------------------------------------------------
    # 1. Carregar Múltiplos Arquivos de Imagem
    # ---------------------------------------------------------
    todos_csvs = glob.glob(PADRAO_BUSCA)
    lista_dfs = []
    
    print("1. Lendo arquivos de imagem (.csv)...")
    for arq in todos_csvs:
        # Ignora arquivos de sistema ou o próprio output
        if NOME_ARQUIVO_SAIDA not in arq and "Planilha" not in arq:
            try:
                # Tenta padrão BR (ponto e vírgula)
                df_temp = pd.read_csv(arq, sep=';', decimal=',')
                # Fallback para padrão US (vírgula) se falhar
                if len(df_temp.columns) < 5:
                     df_temp = pd.read_csv(arq, sep=',', decimal='.')
                
                lista_dfs.append(df_temp)
                print(f"   Ok: {os.path.basename(arq)}")
            except Exception as e:
                print(f"   Erro ao ler {os.path.basename(arq)}: {e}")

    if not lista_dfs:
        sys.exit("ERRO CRÍTICO: Nenhum CSV de imagem encontrado.")
        
    # Junta tudo num DataFrame só
    df_raw = pd.concat(lista_dfs, ignore_index=True)
    print(f"   > Total de espectros importados: {len(df_raw)}")

    # ---------------------------------------------------------
    # 2. Pipeline de Tratamento Espectral
    # ---------------------------------------------------------
    cols_bandas_raw = [c for c in df_raw.columns if "Band_" in c]
    cols_meta = [c for c in df_raw.columns if "Band_" not in c]
    
    # Dicionários para o plot
    estados_dados = {}
    estados_wls = {}
    
    # --- ETAPA 1: RAW ---
    estados_dados['1. Dados Brutos (Raw)'] = df_raw[cols_bandas_raw]
    estados_wls['1. Dados Brutos (Raw)'] = extrair_wavelengths(cols_bandas_raw)
    
    # --- ETAPA 2: CORTE ---
    df_trimmed = cortar_extremidades(df_raw, cols_bandas_raw, TRIM_MIN, TRIM_MAX)
    cols_validas = df_trimmed.columns.tolist()
    estados_dados['2. Corte'] = df_trimmed
    estados_wls['2. Corte'] = extrair_wavelengths(cols_validas)
    
    # --- ETAPA 3: SNV ---
    df_snv = aplicar_snv(df_trimmed, cols_validas)
    estados_dados['3. SNV'] = df_snv
    estados_wls['3. SNV'] = extrair_wavelengths(cols_validas)
    
    # --- ETAPA 4: SAVITZKY-GOLAY ---
    df_sg = aplicar_savgol(df_snv, cols_validas)
    estados_dados['4. Savitzky-Golay'] = df_sg
    estados_wls['4. Savitzky-Golay'] = extrair_wavelengths(cols_validas)
    
    # MOSTRAR GRÁFICOS
    plotar_comparacao(estados_dados, estados_wls)

    # ---------------------------------------------------------
    # 3. Cruzamento com Excel (A CORREÇÃO)
    # ---------------------------------------------------------
    print("\n2. Cruzando com Excel (PlanilhaFiltrada.xlsx)...")
    
    try:
        # LÊ A PARTIR DA LINHA 3 (HEADER=2) - Correção crucial
        df_agro = pd.read_excel(ARQUIVO_AGRONOMICO, header=2)
        
        # Normaliza nomes das colunas
        df_agro.columns = [str(c).strip().lower() for c in df_agro.columns]
        
        # Mapeamento baseado no diagnóstico
        col_id = 'id parcela'
        col_dose = 'dose_n_kg'
        
        # Busca dinâmica para clorofila
        col_clor_m = next((c for c in df_agro.columns if 'médio' in c and 'clorofila' in c), None)
        col_clor_s = next((c for c in df_agro.columns if 'superior' in c and 'clorofila' in c), None)
        
        print(f"   > Colunas Mapeadas -> ID: '{col_id}', Dose: '{col_dose}'")
        
    except Exception as e:
        sys.exit(f"Erro ao ler Excel: {e}")

    # ---------------------------------------------------------
    # 4. Unificação Final
    # ---------------------------------------------------------
    df_proc_full = pd.concat([df_raw[cols_meta], df_sg], axis=1)
    dataset_final = []
    ids_nao_encontrados = 0
    
    for index, row in df_proc_full.iterrows():
        try:
            # Tenta pegar ID e Parte da imagem
            id_img = int(row['ID_Amostra'])
            parte = str(row['Parte_Planta']).strip().upper() # M ou S
            
            # Busca correspondência no Excel
            match = df_agro[df_agro[col_id] == id_img]
            
            if len(match) > 0:
                dados_agro = match.iloc[0]
                
                # Extrai Dose
                dose_valor = dados_agro[col_dose]
                
                # Extrai Clorofila (Lógica M vs S)
                y_cloro = None
                if parte == 'M' and col_clor_m: y_cloro = dados_agro[col_clor_m]
                elif parte == 'S' and col_clor_s: y_cloro = dados_agro[col_clor_s]
                
                # Tratamento numérico da clorofila
                try: val_float = float(y_cloro)
                except: val_float = np.nan
                
                linha = {
                    'ID': id_img,
                    'Parte': parte,
                    'Tratamento': dados_agro.get('tratamento', 'N/A'), # Tenta pegar nome do tratamento
                    'Dose_N': dose_valor,
                    'Y_Clorofila': val_float
                }
                
                # Adiciona todas as bandas processadas
                for c_banda in df_sg.columns:
                    linha[c_banda] = row[c_banda]
                    
                dataset_final.append(linha)
            else:
                ids_nao_encontrados += 1
                
        except Exception as e:
            continue # Pula linhas com erro de leitura

    # Salvar Arquivo Final
    if dataset_final:
        df_final = pd.DataFrame(dataset_final)
        df_final.to_csv(NOME_ARQUIVO_SAIDA, index=False, sep=';', decimal=',')
        
        print(f"\nSUCESSO TOTAL! {len(df_final)} amostras processadas e unificadas.")
        print(f"Arquivo salvo: {NOME_ARQUIVO_SAIDA}")
        
        # Mostra quais doses foram encontradas (Prova Real)
        doses_achadas = sorted(df_final['Dose_N'].astype(float).unique())
        print(f"Doses encontradas no arquivo final: {doses_achadas}")
        
        if ids_nao_encontrados > 0:
            print(f"Aviso: {ids_nao_encontrados} espectros não cruzaram com o Excel.")
    else:
        print("ERRO FATAL: O arquivo final ficou vazio. Verifique os IDs.")