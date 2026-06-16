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
from datetime import datetime

# ================= CONFIGURAÇÕES =================
ARQUIVO_DADOS = 'spectral_indices_rois.csv'
ARQUIVO_AGRONOMICO = os.path.join('Dataset', 'PlanilhaFiltrada.xlsx')

# ================= FUNÇÕES DE FORMATAÇÃO =================

def linha(char="═", comprimento=90):
    """Cria uma linha formatada."""
    return char * comprimento

def titulo_secao(texto, level=1):
    """Formata título de seção."""
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
    """Retorna símbolo de significância."""
    if pd.isna(p_value): return ""
    if p_value < 0.001: return "***"
    elif p_value < 0.01: return "**"
    elif p_value < 0.05: return "*"
    else: return "ns"

# ================= FUNÇÕES AUXILIARES =================

def cruzar_com_excel(df_espectral, arquivo_excel):
    """Cruza dados espectrais com dados agronômicos buscando Tratamento e Dose."""
    try:
        df_agro = pd.read_excel(arquivo_excel, header=2)
        # Normalizar nomes de colunas
        df_agro.columns = [str(c).strip().lower() for c in df_agro.columns]
    except Exception as e:
        print(f"  ⚠️  Erro ao ler Excel: {e}")
        return None
    
    # Identificar colunas chave
    col_id = next((c for c in df_agro.columns if 'id' in c and 'parcela' in c), 'id parcela')
    
    # Busca coluna de TRATAMENTO (Produto)
    col_trat = next((c for c in df_agro.columns if 'tratamento' in c), None)
    
    # Busca coluna de DOSE
    col_dose = next((c for c in df_agro.columns if 'dose' in c), None)
    
    # Busca colunas de Clorofila
    col_clor_m = next((c for c in df_agro.columns if 'médio' in c and 'clorofila' in c), None)
    col_clor_s = next((c for c in df_agro.columns if 'superior' in c and 'clorofila' in c), None)
    
    if not col_trat or not col_dose:
        print("  ❌ ERRO: Não foi possível identificar colunas de 'Tratamento' e 'Dose' automaticamente.")
        print(f"  Colunas encontradas: {df_agro.columns.tolist()}")
        return None

    df_espectral['ID_Numeric'] = df_espectral['ID_Amostra'].astype(int)
    
    cols_to_merge = [col_id, col_trat, col_dose]
    if col_clor_m: cols_to_merge.append(col_clor_m)
    if col_clor_s: cols_to_merge.append(col_clor_s)

    # Merge
    df_merge = df_espectral.merge(df_agro[cols_to_merge], left_on='ID_Numeric', right_on=col_id, how='left')
    
    # Processar Clorofila
    if col_clor_m and col_clor_s:
        df_merge['Y_Clorofila'] = np.where(df_merge['Parte'] == 'M', df_merge[col_clor_m], df_merge[col_clor_s])
    else:
        df_merge['Y_Clorofila'] = np.nan
    
    # Renomear e padronizar
    df_merge.rename(columns={col_trat: 'Tratamento', col_dose: 'Dose'}, inplace=True)
    
    # Garantir que Dose seja tratada como string/categoria para visualização, mas manter ordem se numérico
    df_merge['Dose'] = df_merge['Dose'].astype(str)
    
    # Criar coluna combinada para Post-Hoc
    df_merge['Interacao'] = df_merge['Tratamento'] + " (" + df_merge['Dose'] + ")"
    
    return df_merge.dropna(subset=['Y_Clorofila', 'Tratamento', 'Dose'])

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
    
    stats_desc = df.groupby(['Tratamento', 'Dose'])['Y_Clorofila'].agg(['mean', 'std', 'count', 'min', 'max'])
    
    print("\n  {:<20} {:<10} {:>10} {:>10} {:>10}".format("Tratamento", "Dose", "Média", "DP", "N"))
    print("  " + "─" * 65)
    for (trat, dose), row in stats_desc.iterrows():
        print("  {:<20} {:<10} {:>10.2f} {:>10.2f} {:>10.0f}".format(
            trat[:20], dose, row['mean'], row['std'], row['count']))

    # ========== 2. VERIFICAÇÃO DE PRESSUPOSTOS ==========
    titulo_secao("2. VERIFICAÇÃO DE PRESSUPOSTOS")
    
    # Normalidade (Shapiro-Wilk nos resíduos do modelo)
    model = ols('Y_Clorofila ~ C(Tratamento) + C(Dose) + C(Tratamento):C(Dose)', data=df).fit()
    
    # Nota: Para N > 5000 o Shapiro pode ser impreciso, mas mantemos o cálculo.
    stat_shapiro, p_shapiro = shapiro(model.resid)
    
    print("\n  ► Normalidade dos Resíduos (Shapiro-Wilk):")
    print(f"    W = {stat_shapiro:.4f}, p-value = {p_shapiro:.6f}")
    if p_shapiro > 0.05:
        print("    ✓ Os resíduos seguem distribuição normal.")
    else:
        print("    ⚠️  Os resíduos NÃO seguem distribuição normal (ANOVA é robusta para N grande).")

    # Homogeneidade de Variâncias (Levene)
    print("\n  ► Homogeneidade de Variâncias (Levene):")
    grupos = [g['Y_Clorofila'].values for _, g in df.groupby('Interacao')]
    stat_levene, p_levene = levene(*grupos)
    print(f"    F = {stat_levene:.4f}, p-value = {p_levene:.6f}")
    
    # ========== 3. TABELA ANOVA TWO-WAY ==========
    titulo_secao("3. RESULTADOS DA ANOVA TWO-WAY")
    
    # ANOVA Tipo II
    anova_table = sm.stats.anova_lm(model, typ=2)
    
    # [CORREÇÃO ROBUSTA] - Limpeza total dos nomes das colunas
    # 1. Mapeamento explícito para variações conhecidas do statsmodels
    rename_map = {
        'PR(>F)': 'p_value',
        'Pr(>F)': 'p_value',
        'Sum Sq': 'sum_sq',
        'F value': 'F',
        'df': 'df'
    }
    anova_table.rename(columns=rename_map, inplace=True)

    # 2. Varredura final: se sobrar algo como 'PR(>F)' que o rename não pegou, força a mudança
    cols_clean = []
    for c in anova_table.columns:
        if 'PR' in c or 'Pr' in c: cols_clean.append('p_value')
        elif 'Sum' in c: cols_clean.append('sum_sq')
        else: cols_clean.append(c)
    anova_table.columns = cols_clean

    # Formatando a tabela de saída
    print("\n  {:<25} {:>10} {:>10} {:>10} {:>12}".format("Fonte de Variação", "Sum Sq", "F value", "p-value", "Sig."))
    print("  " + "─" * 75)
    
    interpretacao = {}
    
    # Iteração segura
    for row in anova_table.itertuples():
        source = row.Index.replace("C(Tratamento)", "Tratamento").replace("C(Dose)", "Dose").replace(":", " x ")
        
        # Obter valores de forma segura (usando getattr caso o nome varie)
        val_sum_sq = getattr(row, 'sum_sq', np.nan)
        val_f = getattr(row, 'F', np.nan)
        val_p = getattr(row, 'p_value', np.nan)
        
        sig = simbolo_significancia(val_p)
        
        print("  {:<25} {:>10.2f} {:>10.4f} {:>10.6f} {:>12}".format(
            source, val_sum_sq, val_f, val_p, sig))
        
        # Guardar para conclusão
        if source != "Residual" and not pd.isna(val_p):
            interpretacao[source] = val_p < 0.05

    # ========== 4. INTERPRETAÇÃO DOS EFEITOS ==========
    titulo_secao("4. INTERPRETAÇÃO DOS EFEITOS", level=2)
    
    interacao_sig = interpretacao.get("Tratamento x Dose", False)
    
    if interacao_sig:
        print("\n  ⚠️  INTERAÇÃO SIGNIFICATIVA DETECTADA!")
        print("  Isso significa que o efeito da DOSE depende de qual TRATAMENTO é utilizado (e vice-versa).")
        print("  Recomendação: Analisar a combinação 'Tratamento + Dose' em conjunto.")
    else:
        print("\n  ✓ Não há interação significativa.")
        print("  Os efeitos de Tratamento e Dose agem de forma independente.")

    # ========== 5. TESTE DE TUKEY (POST-HOC) ==========
    titulo_secao("5. COMPARAÇÃO DE MÉDIAS (TUKEY HSD)")
    
    print("\n  Comparando todas as combinações (Tratamento + Dose):")
    
    try:
        tukey = pairwise_tukeyhsd(endog=df['Y_Clorofila'], groups=df['Interacao'], alpha=0.05)
        
        # Converter para DataFrame
        tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
        tukey_df['p-adj'] = pd.to_numeric(tukey_df['p-adj'])
        tukey_df['meandiff'] = pd.to_numeric(tukey_df['meandiff'])
        
        sig_tukey = tukey_df[tukey_df['p-adj'] < 0.05]
        
        print(f"\n  Total de pares comparados: {len(tukey_df)}")
        print(f"  Diferenças significativas encontradas: {len(sig_tukey)}")
        
        if len(sig_tukey) > 0:
            print("\n  Top 5 Diferenças mais significativas:")
            print("  " + "─" * 80)
            for _, row in sig_tukey.sort_values('meandiff', ascending=False).head(5).iterrows():
                print(f"  • {row['group1']:<25} vs {row['group2']:<25} | Dif: {row['meandiff']:+.2f} | p: {row['p-adj']:.4f}")
    except Exception as e:
        print(f"  ⚠️  Não foi possível calcular o Tukey: {e}")

    # ========== 6. VISUALIZAÇÕES ==========
    titulo_secao("6. GERANDO GRÁFICOS", level=2)
    
    sns.set_style("whitegrid")
    
    # --- Gráfico 1: Interaction Plot ---
    print("\n  ► Gráfico 1: Gráfico de Interação (Interaction Plot)")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    # Ordenar Doses
    df_plot = df.copy()
    try:
        df_plot['Dose_Num'] = pd.to_numeric(df_plot['Dose'].str.replace(',', '.').str.extract(r'(\d+\.?\d*)')[0])
        df_plot = df_plot.sort_values('Dose_Num')
    except:
        pass 

    sns.pointplot(data=df_plot, x='Dose', y='Y_Clorofila', hue='Tratamento', 
                  capsize=.1, errorbar=('ci', 95), dodge=True, ax=ax1, palette='deep')
    
    ax1.set_title('Gráfico de Interação: Tratamento x Dose', fontweight='bold', fontsize=14)
    ax1.set_ylabel('Clorofila Média (SPAD)', fontsize=12)
    ax1.set_xlabel('Dose', fontsize=12)
    plt.tight_layout()
    plt.savefig('01_interacao_tratamento_dose.png', dpi=300)
    print("    ✓ Salvo: 01_interacao_tratamento_dose.png")
    plt.close() # Fecha a figura para economizar memória

    # --- Gráfico 2: Boxplot Agrupado ---
    print("\n  ► Gráfico 2: Boxplot Agrupado")
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    
    sns.boxplot(data=df, x='Tratamento', y='Y_Clorofila', hue='Dose', ax=ax2, palette='Set2')
    
    ax2.set_title('Distribuição de Clorofila por Tratamento e Dose', fontweight='bold', fontsize=14)
    ax2.set_ylabel('Clorofila (SPAD)')
    plt.legend(title='Dose', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('02_boxplot_fatorial.png', dpi=300)
    print("    ✓ Salvo: 02_boxplot_fatorial.png")
    plt.close()
    
    # --- Gráfico 3: Mapa de Calor (Heatmap) das Médias ---
    print("\n  ► Gráfico 3: Heatmap de Médias")
    pivot_table = df.pivot_table(values='Y_Clorofila', index='Tratamento', columns='Dose', aggfunc='mean')
    
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot_table, annot=True, cmap='YlGn', fmt='.1f', linewidths=.5, ax=ax3)
    ax3.set_title('Média de Clorofila (Matriz Tratamento x Dose)', fontweight='bold')
    plt.tight_layout()
    plt.savefig('03_heatmap_medias.png', dpi=300)
    print("    ✓ Salvo: 03_heatmap_medias.png")
    plt.close()

    # ========== 7. CONCLUSÃO FINAL ==========
    titulo_secao("7. MELHOR COMBINAÇÃO", level=2)
    
    melhor_combo = df.groupby('Interacao')['Y_Clorofila'].mean().idxmax()
    melhor_valor = df.groupby('Interacao')['Y_Clorofila'].mean().max()
    
    print(f"\n  ✓ A combinação com maior teor médio de clorofila foi:")
    print(f"    {melhor_combo.upper()} = {melhor_valor:.2f} SPAD")
    
    print("\n" + linha("╔", 90))
    print("║" + " ANÁLISE FATORIAL CONCLUÍDA ".center(88) + "║")
    print(linha("╚", 90))

# ================= EXECUÇÃO =================

if __name__ == "__main__":
    print("\n")
    print(linha("╔", 90))
    print("║" + " INICIANDO SISTEMA DE ANÁLISE ".center(88) + "║")
    print(linha("╚", 90))
    
    if os.path.exists(ARQUIVO_DADOS) and os.path.exists(ARQUIVO_AGRONOMICO):
        print("\n  Carregando dados...")
        try:
            df_raw = pd.read_csv(ARQUIVO_DADOS, sep=';', decimal=',')
            
            # Passo 1: Cruzamento e limpeza
            df_final = cruzar_com_excel(df_raw, ARQUIVO_AGRONOMICO)
            
            if df_final is not None and not df_final.empty:
                # Passo 2: Executar Análise Two-Way
                executar_anova_two_way(df_final)
            else:
                print("\n  ❌ Erro: Dataset vazio após cruzamento. Verifique os nomes das colunas no Excel.")
        except Exception as e:
            print(f"\n  ❌ Erro Crítico na execução: {e}")
    else:
        print(f"\n  ❌ Arquivos não encontrados.\n  Verifique: {ARQUIVO_DADOS} e {ARQUIVO_AGRONOMICO}")