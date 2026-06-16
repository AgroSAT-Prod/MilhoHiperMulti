import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import levene, shapiro
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import os
import unicodedata
from datetime import datetime

# ================= CONFIGURAÇÕES =================
ARQUIVO_DADOS = 'spectral_indices_rois2.csv'
ARQUIVO_AGRONOMICO = os.path.join('Dataset', 'PlanilhaFiltrada2.csv')

# ================= FUNÇÕES DE FORMATAÇÃO =================

def linha(char="═", comprimento=90):
    return char * comprimento

def titulo_secao(texto, level=1):
    if level == 1:
        print("\n" + linha("═"))
        print(f"  {texto.upper()}")
        print(linha("═"))
    elif level == 2:
        print("\n" + linha("─"))
        print(f"  ► {texto}")
        print(linha("─"))
    else:
        print(f"\n  ▸ {texto}")

def simbolo_significancia(p_value):
    if pd.isna(p_value): return ""
    if p_value < 0.001: return "***"
    elif p_value < 0.01: return "**"
    elif p_value < 0.05: return "*"
    else: return "ns"

# ================= FUNÇÕES DE LEITURA ROBUSTA =================

def normalizar_texto(texto):
    """Remove acentos e deixa minúsculo para facilitar busca de colunas."""
    if not isinstance(texto, str):
        return str(texto)
    nfkd = unicodedata.normalize('NFKD', texto)
    return "".join([c for c in nfkd if not unicodedata.combining(c)]).lower().strip()

def ler_arquivo_agronomico(caminho):
    """Tenta ler o arquivo agronômico lidando com diferentes headers e formatos."""
    if not os.path.exists(caminho):
        return None
    
    # Tenta ler assumindo que o cabeçalho está na linha 4 (header=3) como no seu exemplo funcional
    try:
        if caminho.lower().endswith('.csv'):
            # Tenta separador ponto-e-vírgula (comum no Brasil)
            df = pd.read_csv(caminho, header=3, sep=';', decimal=',', encoding='latin1')
            if len(df.columns) < 2: # Se falhar, tenta vírgula
                df = pd.read_csv(caminho, header=3, sep=',', decimal='.', encoding='utf-8')
        else:
            df = pd.read_excel(caminho, header=3)
        return df
    except Exception as e:
        print(f"  ⚠️  Erro na leitura primária: {e}")
        return None

def cruzar_com_excel(df_espectral, arquivo_caminho):
    """
    Cruza dados espectrais com agronômicos.
    Procura especificamente por TRATAMENTO e DOSE para Análise Fatorial.
    """
    print(f"\n  [PROCESSAMENTO] Lendo arquivo agronômico: {arquivo_caminho}")
    df_agro = ler_arquivo_agronomico(arquivo_caminho)
    
    if df_agro is None:
        print("  ❌ Erro: Falha ao abrir o arquivo agronômico.")
        return None

    # --- 1. IDENTIFICAÇÃO DE COLUNAS ---
    cols_originais = list(df_agro.columns)
    cols_norm = [normalizar_texto(c) for c in cols_originais]
    
    # Busca ID
    idx_id = next((i for i, c in enumerate(cols_norm) if c == 'id'), -1)
    if idx_id == -1: idx_id = next((i for i, c in enumerate(cols_norm) if 'id' in c and 'parcela' in c), -1)
    col_id = cols_originais[idx_id] if idx_id != -1 else cols_originais[0]

    # Busca TRATAMENTO
    idx_trat = next((i for i, c in enumerate(cols_norm) if 'tratamento' in c or 'produto' in c), -1)
    col_trat = cols_originais[idx_trat] if idx_trat != -1 else None

    # Busca DOSE (Essencial para ANOVA Fatorial)
    idx_dose = next((i for i, c in enumerate(cols_norm) if 'dose' in c), -1)
    col_dose = cols_originais[idx_dose] if idx_dose != -1 else None

    # Busca CLOROFILA (Médio e Superior)
    # Tenta lógica de posição "amostra" primeiro
    cols_amostra = [c for c in cols_originais if 'amostra' in str(c).lower()]
    cols_medio = []
    cols_sup = []

    if len(cols_amostra) >= 4:
        # Assume padrão: 2 primeiros = Médio, 2 seguintes = Superior
        cols_medio = cols_amostra[0:2]
        cols_sup = cols_amostra[2:4]
    else:
        # Busca por nome
        cols_medio = [c for c in cols_originais if 'medio' in normalizar_texto(c) and 'clorofila' in normalizar_texto(c)]
        cols_sup = [c for c in cols_originais if 'superior' in normalizar_texto(c) and 'clorofila' in normalizar_texto(c)]

    print(f"  ✓ ID: '{col_id}'")
    print(f"  ✓ Tratamento: '{col_trat}' | Dose: '{col_dose}'")
    
    if not col_trat or not col_dose:
        print("  ❌ ERRO CRÍTICO: Não foi possível encontrar colunas distintas para 'Tratamento' e 'Dose'.")
        print(f"  Colunas disponíveis: {cols_originais}")
        return None

    # --- 2. PREPARAÇÃO PARA MERGE ---
    df_espectral['ID_Numeric'] = pd.to_numeric(df_espectral['ID_Amostra'], errors='coerce')
    df_agro[col_id] = pd.to_numeric(df_agro[col_id], errors='coerce')
    df_agro = df_agro.dropna(subset=[col_id])

    cols_merge = [col_id, col_trat, col_dose] + cols_medio + cols_sup
    # Remove duplicatas mantendo ordem
    cols_merge = list(dict.fromkeys(cols_merge))

    try:
        df_merge = df_espectral.merge(df_agro[cols_merge], left_on='ID_Numeric', right_on=col_id, how='inner')
    except Exception as e:
        print(f"  ❌ Erro no Merge: {e}")
        return None

    # --- 3. LIMPEZA E CÁLCULO DE MÉDIAS ---
    def limpar_float(serie):
        if serie.dtype == object:
            return pd.to_numeric(serie.astype(str).str.replace(',', '.'), errors='coerce')
        return pd.to_numeric(serie, errors='coerce')

    # Calcula média Terço Médio
    if cols_medio:
        df_merge['Clorofila_M_Final'] = df_merge[cols_medio].apply(limpar_float).mean(axis=1)
    else:
        df_merge['Clorofila_M_Final'] = np.nan

    # Calcula média Terço Superior
    if cols_sup:
        df_merge['Clorofila_S_Final'] = df_merge[cols_sup].apply(limpar_float).mean(axis=1)
    else:
        df_merge['Clorofila_S_Final'] = np.nan

    # Seleciona qual usar baseada no arquivo espectral (Parte M ou S)
    df_merge['Y_Clorofila'] = np.where(df_merge['Parte'] == 'M', 
                                     df_merge['Clorofila_M_Final'], 
                                     df_merge['Clorofila_S_Final'])

    # --- 4. PADRONIZAÇÃO FINAL ---
    df_merge.rename(columns={col_trat: 'Tratamento', col_dose: 'Dose'}, inplace=True)
    
    # Garante que Dose é string para evitar erros no gráfico categórico
    df_merge['Dose'] = df_merge['Dose'].astype(str).str.replace('.0', '', regex=False)
    
    # Cria coluna de interação para o Tukey
    df_merge['Interacao'] = df_merge['Tratamento'] + " (" + df_merge['Dose'] + ")"
    
    df_final = df_merge.dropna(subset=['Y_Clorofila', 'Tratamento', 'Dose'])
    df_final = df_final[df_final['Y_Clorofila'] > 0] # Remove zeros

    return df_final

# ================= ANÁLISES TWO-WAY ANOVA =================

def executar_anova_two_way(df):
    """Executa ANOVA Two-Way com formatação profissional."""
    
    print("\n" + linha("╔", 90))
    print("║" + " " * 88 + "║")
    print("║  " + "ANÁLISE ESTATÍSTICA: ANOVA TWO-WAY (FATORIAL)".center(84) + "  ║")
    print("║  " + "EFEITOS DE TRATAMENTO E DOSE NO TEOR DE CLOROFILA".center(84) + "  ║")
    print("║" + " " * 88 + "║")
    print(linha("╚", 90))
    
    # ========== 1. ESTATÍSTICAS DESCRITIVAS ==========
    titulo_secao("1. ESTATÍSTICAS DESCRITIVAS (TRATAMENTO x DOSE)")
    
    stats_desc = df.groupby(['Tratamento', 'Dose'])['Y_Clorofila'].agg(['mean', 'std', 'count'])
    
    print("\n  {:<20} {:<10} {:>10} {:>10} {:>10}".format("Tratamento", "Dose", "Média", "DP", "N"))
    print("  " + "─" * 65)
    for (trat, dose), row in stats_desc.iterrows():
        print("  {:<20} {:<10} {:>10.2f} {:>10.2f} {:>10.0f}".format(
            str(trat)[:20], str(dose), row['mean'], row['std'], row['count']))

    # ========== 2. ANOVA TWO-WAY ==========
    titulo_secao("2. RESULTADOS DA ANOVA TWO-WAY")
    
    # Cria o modelo: Y ~ Tratamento + Dose + Interação
    model = ols('Y_Clorofila ~ C(Tratamento) + C(Dose) + C(Tratamento):C(Dose)', data=df).fit()
    
    # ANOVA Tipo II
    anova_table = sm.stats.anova_lm(model, typ=2)
    
    # Limpeza robusta de nomes de colunas
    rename_map = {'PR(>F)': 'p_value', 'Pr(>F)': 'p_value', 'Sum Sq': 'sum_sq', 'F value': 'F'}
    anova_table.rename(columns=rename_map, inplace=True)
    
    # Varredura final de colunas
    cols_clean = []
    for c in anova_table.columns:
        if 'PR' in c or 'Pr' in c: cols_clean.append('p_value')
        else: cols_clean.append(c)
    anova_table.columns = cols_clean

    print("\n  {:<25} {:>10} {:>10} {:>10} {:>12}".format("Fonte de Variação", "Sum Sq", "F value", "p-value", "Sig."))
    print("  " + "─" * 75)
    
    interpretacao = {}
    
    for row in anova_table.itertuples():
        source = row.Index.replace("C(Tratamento)", "Tratamento").replace("C(Dose)", "Dose").replace(":", " x ")
        val_p = getattr(row, 'p_value', np.nan)
        val_f = getattr(row, 'F', np.nan)
        val_sq = getattr(row, 'sum_sq', np.nan)
        
        sig = simbolo_significancia(val_p)
        print("  {:<25} {:>10.2f} {:>10.4f} {:>10.6f} {:>12}".format(source, val_sq, val_f, val_p, sig))
        
        if source != "Residual" and not pd.isna(val_p):
            interpretacao[source] = val_p < 0.05

    # ========== 3. COMPARAÇÃO DE MÉDIAS (TUKEY) ==========
    titulo_secao("3. COMPARAÇÃO DE MÉDIAS (TUKEY HSD)")
    
    print("\n  Comparando todas as combinações (Tratamento + Dose):")
    try:
        tukey = pairwise_tukeyhsd(endog=df['Y_Clorofila'], groups=df['Interacao'], alpha=0.05)
        tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
        tukey_df['p-adj'] = pd.to_numeric(tukey_df['p-adj'])
        tukey_df['meandiff'] = pd.to_numeric(tukey_df['meandiff'])
        
        sig_tukey = tukey_df[tukey_df['p-adj'] < 0.05]
        
        print(f"\n  Total de pares comparados: {len(tukey_df)}")
        print(f"  Diferenças significativas: {len(sig_tukey)}")
        
        if len(sig_tukey) > 0:
            print("\n  Top 5 Diferenças mais significativas:")
            print("  " + "─" * 80)
            for _, row in sig_tukey.sort_values('meandiff', ascending=False).head(5).iterrows():
                print(f"  • {row['group1']:<25} vs {row['group2']:<25} | Dif: {row['meandiff']:+.2f} | p: {row['p-adj']:.4f}")
    except Exception as e:
        print(f"  ⚠️  Erro no Tukey: {e}")

    # ========== 4. GRÁFICOS ==========
    titulo_secao("4. VISUALIZAÇÃO", level=2)
    sns.set_style("whitegrid")
    
    # Interaction Plot
    print("\n  ► Gerando Gráfico de Interação...")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.pointplot(data=df, x='Dose', y='Y_Clorofila', hue='Tratamento', 
                  capsize=.1, errorbar=('ci', 95), dodge=True, ax=ax1, palette='deep')
    ax1.set_title('Interação: Tratamento x Dose', fontweight='bold')
    plt.tight_layout()
    plt.savefig('01_interacao_fatorial.png', dpi=300)
    print("    ✓ Salvo: 01_interacao_fatorial.png")
    plt.show()

    # Heatmap
    print("\n  ► Gerando Heatmap de Médias...")
    pivot = df.pivot_table(values='Y_Clorofila', index='Tratamento', columns='Dose', aggfunc='mean')
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, cmap='YlGn', fmt='.1f', ax=ax2)
    ax2.set_title('Média de Clorofila (Matriz)', fontweight='bold')
    plt.tight_layout()
    plt.savefig('02_heatmap_fatorial.png', dpi=300)
    print("    ✓ Salvo: 02_heatmap_fatorial.png")
    plt.show()

# ================= EXECUÇÃO =================

if __name__ == "__main__":
    print("\n")
    print(linha("╔", 90))
    print("║" + " INICIANDO SISTEMA DE ANÁLISE FATORIAL ".center(88) + "║")
    print(linha("╚", 90))
    
    if os.path.exists(ARQUIVO_DADOS) and os.path.exists(ARQUIVO_AGRONOMICO):
        print("\n  Carregando dados...")
        try:
            # Passo 1: Leitura do espectral
            df_raw = pd.read_csv(ARQUIVO_DADOS, sep=';', decimal=',')
            
            # Passo 2: Cruzamento inteligente (usando lógica robusta do seu exemplo)
            df_final = cruzar_com_excel(df_raw, ARQUIVO_AGRONOMICO)
            
            if df_final is not None and not df_final.empty:
                # Passo 3: Executar Análise
                executar_anova_two_way(df_final)
            else:
                print("\n  ❌ Erro: Dataset vazio. Verifique se o arquivo agronômico tem colunas 'Tratamento' e 'Dose'.")
        except Exception as e:
            print(f"\n  ❌ Erro Crítico: {e}")
    else:
        print(f"\n  ❌ Arquivos não encontrados.\n  Verifique: {ARQUIVO_DADOS} e {ARQUIVO_AGRONOMICO}")