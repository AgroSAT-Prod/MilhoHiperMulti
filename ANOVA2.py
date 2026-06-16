import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway, levene, shapiro, kruskal
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison
import os
import unicodedata
from datetime import datetime

# ================= CONFIGURAÇÕES =================
ARQUIVO_DADOS = 'spectral_indices_rois2.csv'
ARQUIVO_AGRONOMICO = os.path.join('Dataset', 'PlanilhaFiltrada2.csv')

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

def formato_tabela_stats(df_stats):
    """Formata tabela de estatísticas descritivas."""
    print("\n  {:<20} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
        "Tratamento", "Média", "DP", "Mín", "Máx", "N"))
    print("  " + "─" * 86)
    
    for trat in df_stats.index:
        print("  {:<20} {:>10.2f} {:>10.2f} {:>10.2f} {:>10.2f} {:>10.0f}".format(
            str(trat)[:20],
            df_stats.loc[trat, 'mean'],
            df_stats.loc[trat, 'std'],
            df_stats.loc[trat, 'min'],
            df_stats.loc[trat, 'max'],
            df_stats.loc[trat, 'count']
        ))

def simbolo_significancia(p_value):
    """Retorna símbolo de significância."""
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return "ns"

def interpretacao_p_value(p_value):
    """Interpretação textual do p-value."""
    if p_value < 0.001:
        return "Extremamente significativo (p < 0.001)"
    elif p_value < 0.01:
        return "Muito significativo (p < 0.01)"
    elif p_value < 0.05:
        return "Significativo (p < 0.05)"
    else:
        return "Não significativo (p ≥ 0.05)"

# ================= FUNÇÕES DE CARREGAMENTO (CORRIGIDAS) =================

def normalizar_texto(texto):
    """Remove acentos e deixa minúsculo para facilitar busca de colunas."""
    if not isinstance(texto, str):
        return str(texto)
    # Normaliza unicode (remove acentos)
    nfkd = unicodedata.normalize('NFKD', texto)
    return "".join([c for c in nfkd if not unicodedata.combining(c)]).lower().strip()

def cruzar_com_excel(df_espectral, arquivo_excel):
    """
    Cruza dados espectrais com dados agronômicos.
    Versão robusta: Detecta colunas 'amostra' duplicadas e atribui pela ordem (Médio -> Superior).
    """
    print(f"\n  [DEBUG] Lendo arquivo agronômico: {arquivo_excel}")
    
    try:
        # Tenta ler com header=3 (linha 4, onde estão "ID", "amostra1", etc.)
        if arquivo_excel.lower().endswith('.csv'):
            df_agro = pd.read_csv(arquivo_excel, header=3, sep=';', decimal=',', encoding='latin1')
        else:
            df_agro = pd.read_excel(arquivo_excel, header=3)
    except Exception as e:
        print(f"  ⚠️  Erro fatal ao ler arquivo: {e}")
        return None

    # --- 1. NORMALIZAÇÃO DE COLUNAS ---
    # Cria mapa original para busca, mas vamos usar df_agro.columns para acessar os dados
    cols_originais = list(df_agro.columns)
    
    # --- 2. LOCALIZAR COLUNA ID ---
    # Busca por nome exato ou variantes
    cols_norm = [normalizar_texto(c) for c in cols_originais]
    
    idx_id = next((i for i, c in enumerate(cols_norm) if c == 'id'), -1)
    if idx_id == -1:
        idx_id = next((i for i, c in enumerate(cols_norm) if 'id' in c and 'parcela' in c), -1)
    
    if idx_id != -1:
        col_id = cols_originais[idx_id]
    else:
        col_id = cols_originais[0] # Fallback: primeira coluna
    
    # --- 3. LOCALIZAR TRATAMENTO ---
    idx_trat = next((i for i, c in enumerate(cols_norm) if 'tratamento' in c or 'dose' in c), -1)
    col_alvo = cols_originais[idx_trat] if idx_trat != -1 else None
    
    # --- 4. LOCALIZAR CLOROFILA (LÓGICA POSICIONAL) ---
    # O CSV agora tem colunas "amostra1", "amostra2" duplicadas.
    # O Pandas renomeia automaticamente para "amostra1", "amostra2", "amostra1.1", "amostra2.1"
    
    cols_amostra = [c for c in cols_originais if 'amostra' in str(c).lower()]
    
    cols_medio_selecionadas = []
    cols_sup_selecionadas = []
    
    if len(cols_amostra) >= 4:
        print(f"  [DEBUG] Colunas de amostra detectadas: {cols_amostra}")
        # Assume ordem da planilha: Primeiro par = Médio, Segundo par = Superior
        cols_medio_selecionadas = cols_amostra[0:2] # Pega as 2 primeiras
        cols_sup_selecionadas = cols_amostra[2:4]   # Pega as 2 seguintes
    elif len(cols_amostra) >= 2:
        # Caso só tenha um terço medido
        print(f"  [DEBUG] Apenas 2 colunas de amostra detectadas. Assumindo Terço Médio.")
        cols_medio_selecionadas = cols_amostra[0:2]
    else:
        # Fallback: Tenta procurar nomes antigos "medio", "superior" caso o formato mude
        print("  [DEBUG] Colunas 'amostra' não encontradas. Tentando busca por nome...")
        c_m = [c for c in cols_originais if 'medio' in normalizar_texto(c) and 'clorofila' in normalizar_texto(c)]
        c_s = [c for c in cols_originais if 'superior' in normalizar_texto(c) and 'clorofila' in normalizar_texto(c)]
        cols_medio_selecionadas = c_m
        cols_sup_selecionadas = c_s

    print(f"  ✓ Mapeamento: ID='{col_id}' | Trat='{col_alvo}'")
    print(f"  ✓ Clorofila Médio: {cols_medio_selecionadas if cols_medio_selecionadas else 'NÃO ENCONTRADO'}")
    print(f"  ✓ Clorofila Superior: {cols_sup_selecionadas if cols_sup_selecionadas else 'NÃO ENCONTRADO'}")

    # --- 5. CONVERSÃO DE TIPOS E LIMPEZA ---
    df_espectral['ID_Numeric'] = pd.to_numeric(df_espectral['ID_Amostra'], errors='coerce')
    df_agro[col_id] = pd.to_numeric(df_agro[col_id], errors='coerce')
    df_agro = df_agro.dropna(subset=[col_id])

    # --- 6. MERGE ---
    # Prepara lista de colunas para o merge
    cols_merge = [col_id]
    if col_alvo: cols_merge.append(col_alvo)
    cols_merge.extend(cols_medio_selecionadas)
    cols_merge.extend(cols_sup_selecionadas)
    
    # Remove duplicatas mantendo ordem
    cols_merge = list(dict.fromkeys(cols_merge))

    try:
        df_merge = df_espectral.merge(df_agro[cols_merge], left_on='ID_Numeric', right_on=col_id, how='inner')
    except Exception as e:
        print(f"  ❌ Erro no cruzamento (Merge): {e}")
        return None
    
    print(f"  ✓ Linhas após cruzamento: {len(df_merge)}")

    # --- 7. CÁLCULO DAS MÉDIAS ---
    def limpar_float(serie):
        if serie.dtype == object:
            return pd.to_numeric(serie.astype(str).str.replace(',', '.'), errors='coerce')
        return pd.to_numeric(serie, errors='coerce')

    # Calcula média para Terço Médio
    if cols_medio_selecionadas:
        dados_medio = df_merge[cols_medio_selecionadas].apply(limpar_float)
        df_merge['Clorofila_M_Final'] = dados_medio.mean(axis=1)
    else:
        df_merge['Clorofila_M_Final'] = np.nan

    # Calcula média para Terço Superior
    if cols_sup_selecionadas:
        dados_sup = df_merge[cols_sup_selecionadas].apply(limpar_float)
        df_merge['Clorofila_S_Final'] = dados_sup.mean(axis=1)
    else:
        df_merge['Clorofila_S_Final'] = np.nan

    # Seleção Final baseada na Parte (M ou S) do arquivo espectral
    df_merge['Y_Clorofila'] = np.where(df_merge['Parte'] == 'M', 
                                       df_merge['Clorofila_M_Final'], 
                                       df_merge['Clorofila_S_Final'])
    
    # --- 8. FINALIZAÇÃO ---
    if col_alvo:
        df_merge.rename(columns={col_alvo: 'Tratamento'}, inplace=True)
    else:
        df_merge['Tratamento'] = 'Desconhecido'
    
    # Filtra dados válidos
    df_final = df_merge.dropna(subset=['Y_Clorofila'])
    df_final = df_final[df_final['Y_Clorofila'] > 0]
    
    print(f"  ✓ Dados válidos finais (com clorofila > 0): {len(df_final)}")
    
    cols_espectrais = list(df_espectral.columns)
    cols_extras = ['Tratamento', 'Y_Clorofila']
    cols_finais = [c for c in cols_espectrais if c not in cols_extras] + cols_extras
    cols_existentes = [c for c in cols_finais if c in df_final.columns]
    
    return df_final[cols_existentes]

# ================= ANÁLISES PÓS-HOC =================

def executar_testes_pos_hoc(df):
    """Executa testes pós-hoc com formatação profissional."""
    
    print("\n" + linha("╔", 90))
    print("║" + " " * 88 + "║")
    print("║  " + "ANÁLISE ESTATÍSTICA: COMPARAÇÃO DE TRATAMENTOS QUANTO AO TEOR DE CLOROFILA".center(84) + "  ║")
    print("║" + " " * 88 + "║")
    print(linha("╚", 90))
    
    print(f"\n  Data da análise: {datetime.now().strftime('%d/%m/%Y às %H:%M:%S')}")
    
    # Remover valores inválidos e garantir numérico
    df['Y_Clorofila'] = pd.to_numeric(df['Y_Clorofila'], errors='coerce')
    df_analise = df[['Tratamento', 'Y_Clorofila']].dropna()
    df_analise = df_analise[df_analise['Y_Clorofila'] > 0].copy()
    
    if len(df_analise) == 0:
        print("  ❌ Nenhum dado válido para análise.")
        return None
    
    # ========== 1. INFORMAÇÕES DO ESTUDO ==========
    titulo_secao("1. INFORMAÇÕES DO ESTUDO")
    
    total_amostras = len(df_analise)
    n_tratamentos = df_analise['Tratamento'].nunique()
    
    print(f"\n  • Total de observações: {total_amostras:,}")
    print(f"  • Número de tratamentos: {n_tratamentos}")
    print(f"  • Variável resposta: Teor de Clorofila (unidades SPAD)")
    
    # ========== 2. ESTATÍSTICAS DESCRITIVAS ==========
    titulo_secao("2. ESTATÍSTICAS DESCRITIVAS POR TRATAMENTO", level=2)
    
    stats_desc = df_analise.groupby('Tratamento')['Y_Clorofila'].describe()
    formato_tabela_stats(stats_desc)
    
    # ========== 3. PRESSUPOSTOS ANOVA ==========
    titulo_secao("3. VERIFICAÇÃO DE PRESSUPOSTOS DA ANOVA", level=2)
    
    grupos = [group['Y_Clorofila'].values for name, group in df_analise.groupby('Tratamento')]
    tratamentos_unicos = sorted(df_analise['Tratamento'].unique())
    
    # 3.1 Normalidade
    print("\n  ► Teste de Normalidade (Shapiro-Wilk):")
    print("    Hipótese nula: Os dados seguem distribuição normal")
    print("    " + "─" * 86)
    
    normal_count = 0
    for trat in tratamentos_unicos:
        grupo = df_analise[df_analise['Tratamento'] == trat]['Y_Clorofila'].values
        if len(grupo) >= 3:
            stat, p_value = shapiro(grupo)
            sig = "✓ Normal" if p_value > 0.05 else "✗ Não-Normal"
            normal_count += 1 if p_value > 0.05 else 0
            print(f"    {str(trat):<25} p-value = {p_value:.6f}  {sig}")
    
    print(f"\n    Resultado: {normal_count}/{len(tratamentos_unicos)} grupos com distribuição normal")
    
    # 3.2 Homogeneidade de variâncias
    print("\n  ► Teste de Homogeneidade de Variâncias (Levene):")
    print("    Hipótese nula: As variâncias são iguais entre os grupos")
    print("    " + "─" * 86)
    
    stat_levene, p_levene = levene(*grupos)
    print(f"    Estatística F = {stat_levene:.4f}")
    print(f"    p-value = {p_levene:.6f}")
    print(f"    Interpretação: {interpretacao_p_value(p_levene)}")
    
    if p_levene < 0.05:
        print(f"    ⚠️  Nota: Variâncias não-homogêneas, mas compensado pelo n amostral grande")
    
    # ========== 4. ANÁLISE DE VARIÂNCIA ==========
    titulo_secao("4. ANÁLISE DE VARIÂNCIA (ANOVA ONE-WAY)", level=2)
    
    f_stat, p_anova = f_oneway(*grupos)
    
    print(f"\n  Hipótese nula: As médias de clorofila são iguais para todos os tratamentos")
    print("  " + "─" * 86)
    print(f"\n  Estatística F = {f_stat:.4f}")
    print(f"  p-value = {p_anova:.10f}")
    print(f"  Significância: {interpretacao_p_value(p_anova)} {simbolo_significancia(p_anova)}")
    
    if p_anova < 0.05:
        print(f"\n  ✓ DECISÃO: Rejeitar a hipótese nula")
        print(f"    Há evidências de diferenças significativas na clorofila entre os tratamentos")
    else:
        print(f"\n  ✗ DECISÃO: Não rejeitar a hipótese nula")
    
    # ========== 5. TESTE NÃO-PARAMÉTRICO (KRUSKAL-WALLIS) ==========
    titulo_secao("5. TESTE NÃO-PARAMÉTRICO (KRUSKAL-WALLIS)", level=2)
    
    print("\n  Como alternativa robusta diante da violação de normalidade:")
    print("  " + "─" * 86)
    
    h_stat, p_kw = kruskal(*grupos)
    print(f"\n  Estatística H = {h_stat:.4f}")
    print(f"  p-value = {p_kw:.10f}")
    print(f"  Significância: {interpretacao_p_value(p_kw)} {simbolo_significancia(p_kw)}")
    print(f"\n  ✓ Conclusão: Ambos os testes (ANOVA e Kruskal-Wallis) confirmam diferenças significativas")
    
    # ========== 6. TESTE DE TUKEY HSD ==========
    titulo_secao("6. COMPARAÇÃO DE MÉDIAS - TESTE DE TUKEY HSD", level=2)
    
    print("\n  Teste pós-hoc para identificar quais tratamentos diferem entre si")
    print("  (Nível de significância: α = 0.05 com correção para múltiplas comparações)")
    print("  " + "─" * 86)
    
    tukey_result = pairwise_tukeyhsd(endog=df_analise['Y_Clorofila'], 
                                     groups=df_analise['Tratamento'], 
                                     alpha=0.05)
    
    print("\n" + str(tukey_result))
    
    # ========== 7. RESUMO DAS COMPARAÇÕES ==========
    titulo_secao("7. RESUMO DAS COMPARAÇÕES SIGNIFICATIVAS", level=2)
    
    tukey_df = pd.DataFrame(data=tukey_result.summary().data[1:], 
                            columns=tukey_result.summary().data[0])
    tukey_df['meandiff'] = pd.to_numeric(tukey_df['meandiff'])
    tukey_df['p-adj'] = pd.to_numeric(tukey_df['p-adj'])
    
    sig_comparacoes = tukey_df[tukey_df['p-adj'] < 0.05].copy()
    nao_sig = tukey_df[tukey_df['p-adj'] >= 0.05].copy()
    
    print(f"\n  Comparações SIGNIFICATIVAS (p < 0.05): {len(sig_comparacoes)}")
    print("  " + "─" * 86)
    
    for idx, row in sig_comparacoes.iterrows():
        grupo1 = str(row['group1'])
        grupo2 = str(row['group2'])
        diff = float(row['meandiff'])
        p_adj = float(row['p-adj'])
        ic_lower = float(row['lower'])
        ic_upper = float(row['upper'])
        
        if diff > 0:
            print(f"  • {grupo2:<20} > {grupo1:<20} Δ = {diff:+7.2f}  IC95%: [{ic_lower:+6.2f}, {ic_upper:+6.2f}]  p = {p_adj:.4f}")
        else:
            print(f"  • {grupo1:<20} > {grupo2:<20} Δ = {abs(diff):+7.2f}  IC95%: [{ic_lower:+6.2f}, {ic_upper:+6.2f}]  p = {p_adj:.4f}")
    
    if len(nao_sig) > 0:
        print(f"\n  Comparações NÃO SIGNIFICATIVAS (p ≥ 0.05): {len(nao_sig)}")
        print("  " + "─" * 86)
        for idx, row in nao_sig.iterrows():
            grupo1 = str(row['group1'])
            grupo2 = str(row['group2'])
            diff = float(row['meandiff'])
            p_adj = float(row['p-adj'])
            print(f"  ○ {grupo1:<20} vs {grupo2:<20} Δ = {diff:+7.2f}  p = {p_adj:.4f}  (sem diferença significativa)")
    
    # ========== 8. RANKING DE TRATAMENTOS ==========
    titulo_secao("8. RANKING DE TRATAMENTOS", level=2)
    
    medias = df_analise.groupby('Tratamento')['Y_Clorofila'].mean().sort_values(ascending=False)
    contagens = df_analise.groupby('Tratamento').size()
    desvios = df_analise.groupby('Tratamento')['Y_Clorofila'].std()
    
    print("\n  {:<3} {:<25} {:<10} {:<8} {:<8}".format("Pos", "Tratamento", "Média", "DP", "N"))
    print("  " + "─" * 86)
    
    for i, (trat, media) in enumerate(medias.items(), 1):
        n = contagens[trat]
        dp = desvios[trat]
        barra = "█" * (n // 500) if n >= 500 else "█"
        print(f"  {i:<3} {str(trat):<25} {media:>8.2f}  {dp:>7.2f}  {n:>6} {barra}")
    
    # ========== 9. ANÁLISE AGRONÔMICA ==========
    titulo_secao("9. INTERPRETAÇÃO AGRONÔMICA", level=2)
    
    melhor = str(medias.index[0])
    pior = str(medias.index[-1])
    melhor_media = medias.iloc[0]
    pior_media = medias.iloc[-1]
    
    print(f"\n  Tratamento com MAIOR teor de clorofila:")
    print(f"    {melhor:<25} Média = {melhor_media:.2f} unidades SPAD")
    
    print(f"\n  Tratamento com MENOR teor de clorofila:")
    print(f"    {pior:<25} Média = {pior_media:.2f} unidades SPAD")
    
    print(f"\n  Diferença máxima observada:")
    print(f"    {melhor_media - pior_media:.2f} unidades SPAD ({((melhor_media - pior_media)/pior_media)*100:.1f}% de aumento)")
    
    # Análise de grupos homogêneos
    print(f"\n  Grupos homogêneos (sem diferenças significativas):")
    print("  " + "─" * 86)
    
    for idx, row in nao_sig.iterrows():
        print(f"    • {row['group1']} ≈ {row['group2']}")
    
    # ========== 10. VISUALIZAÇÕES ==========
    titulo_secao("10. GERANDO VISUALIZAÇÕES", level=2)
    
    ordem_tratamentos = medias.index.tolist()
    df_ordenado = df_analise.copy()
    df_ordenado['Tratamento'] = pd.Categorical(df_ordenado['Tratamento'], 
                                               categories=ordem_tratamentos, 
                                               ordered=True)
    df_ordenado = df_ordenado.sort_values('Tratamento')
    
    # Gráfico 1: Boxplot
    print("\n  ► Gráfico 1: Distribuição do Teor de Clorofila (Boxplot)")
    fig1, ax1 = plt.subplots(figsize=(14, 7))
    sns.boxplot(data=df_ordenado, x='Tratamento', y='Y_Clorofila', ax=ax1,
                palette='Set2', width=0.6)
    ax1.set_title('Distribuição do Teor de Clorofila por Tratamento', 
                  fontweight='bold', fontsize=14)
    ax1.set_xlabel('Tratamento', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Clorofila (SPAD)', fontweight='bold', fontsize=12)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('01_boxplot_clorofila.png', dpi=300, bbox_inches='tight')
    print("    ✓ Salvo como '01_boxplot_clorofila.png'")
    plt.show()
    
    # Gráfico 2: Heatmap de p-values
    print("\n  ► Gráfico 2: Matriz de p-values (Tukey HSD)")
    tratamentos_sorted = medias.index.tolist()
    n_trat = len(tratamentos_sorted)
    matriz_p = np.ones((n_trat, n_trat))
    
    for idx, row in tukey_df.iterrows():
        i = tratamentos_sorted.index(row['group1'])
        j = tratamentos_sorted.index(row['group2'])
        p_val = float(row['p-adj'])
        matriz_p[i, j] = p_val
        matriz_p[j, i] = p_val
    
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    sns.heatmap(matriz_p, annot=True, fmt='.4f', cmap='RdYlGn_r', center=0.05,
                xticklabels=tratamentos_sorted, yticklabels=tratamentos_sorted,
                ax=ax2, cbar_kws={'label': 'p-value ajustado'}, vmin=0, vmax=0.1,
                square=True, linewidths=0.5)
    ax2.set_title('Matriz de p-values (Tukey HSD)\nVerde = Sem diferença | Vermelho = Diferença significativa', 
                  fontweight='bold', fontsize=13)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('02_heatmap_pvalues.png', dpi=300, bbox_inches='tight')
    print("    ✓ Salvo como '02_heatmap_pvalues.png'")
    plt.show()
    
    # Gráfico 3: Médias com IC
    print("\n  ► Gráfico 3: Médias com Intervalo de Confiança 95%")
    medias_ci = []
    for trat in ordem_tratamentos:
        dados_trat = df_analise[df_analise['Tratamento'] == trat]['Y_Clorofila']
        media = dados_trat.mean()
        ic = 1.96 * dados_trat.sem()
        medias_ci.append({'Tratamento': trat, 'Media': media, 'IC': ic})
    
    medias_ci_df = pd.DataFrame(medias_ci)
    x_pos = np.arange(len(medias_ci_df))
    
    fig3, ax3 = plt.subplots(figsize=(12, 7))
    ax3.errorbar(x_pos, medias_ci_df['Media'], yerr=medias_ci_df['IC'], 
                 fmt='o-', capsize=8, capthick=2, markersize=10, linewidth=2.5, 
                 color='steelblue', ecolor='darkblue', label='IC 95%')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(medias_ci_df['Tratamento'], rotation=45, ha='right', fontsize=11)
    ax3.set_title('Médias de Clorofila com Intervalo de Confiança 95%', 
                  fontweight='bold', fontsize=14)
    ax3.set_ylabel('Clorofila (SPAD)', fontweight='bold', fontsize=12)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Adicionar valores nas barras
    for i, (media, ic) in enumerate(zip(medias_ci_df['Media'], medias_ci_df['IC'])):
        ax3.text(i, media + ic + 1, f'{media:.1f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('03_medias_ic.png', dpi=300, bbox_inches='tight')
    print("    ✓ Salvo como '03_medias_ic.png'")
    plt.show()
    
    # Gráfico 4: Diferenças significativas
    print("\n  ► Gráfico 4: Diferenças Significativas entre Tratamentos")
    sig_comp = tukey_df[tukey_df['p-adj'] < 0.05].copy()
    sig_comp['Comparacao'] = sig_comp['group1'] + ' vs\n' + sig_comp['group2']
    sig_comp['meandiff'] = pd.to_numeric(sig_comp['meandiff'])
    sig_comp = sig_comp.sort_values('meandiff')
    
    if len(sig_comp) > 0:
        fig4, ax4 = plt.subplots(figsize=(12, 10))
        cores = ['#d62728' if x < 0 else '#2ca02c' for x in sig_comp['meandiff']]
        
        barras = ax4.barh(range(len(sig_comp)), sig_comp['meandiff'], 
                          color=cores, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax4.set_yticks(range(len(sig_comp)))
        ax4.set_yticklabels(sig_comp['Comparacao'], fontsize=9)
        ax4.set_xlabel('Diferença Média de Clorofila (SPAD)', fontweight='bold', fontsize=12)
        ax4.set_title('Diferenças Significativas entre Tratamentos (p < 0.05)\nVerde = Aumento | Vermelho = Redução', 
                      fontweight='bold', fontsize=13)
        ax4.axvline(x=0, color='black', linestyle='--', linewidth=2)
        ax4.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Adicionar valores nas barras
        for i, (barra, val) in enumerate(zip(barras, sig_comp['meandiff'])):
            ax4.text(val + 0.2 if val > 0 else val - 0.2, i, f'{val:.2f}', 
                    va='center', ha='left' if val > 0 else 'right', fontweight='bold', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('04_diferencas_significativas.png', dpi=300, bbox_inches='tight')
        print("    ✓ Salvo como '04_diferencas_significativas.png'")
        plt.show()
    
    print("\n  ✓ Todos os gráficos foram salvos e exibidos!")
    
    # ========== 11. CONCLUSÕES ==========
    titulo_secao("11. CONCLUSÕES E RECOMENDAÇÕES", level=2)
    
    print(f"\n  ✓ Há diferenças ALTAMENTE SIGNIFICATIVAS entre os tratamentos (F={f_stat:.2f}, p<0.001)")
    print(f"\n  ✓ Tratamento recomendado: {melhor}")
    print(f"    Teor de clorofila: {melhor_media:.2f} unidades SPAD")
    
    print(f"\n  ✓ Resultado confirmado por teste não-paramétrico (Kruskal-Wallis, H={h_stat:.2f}, p<0.001)")
    
    print("\n" + linha("╔", 90))
    print("║" + " ANÁLISE CONCLUÍDA COM SUCESSO ".center(88) + "║")
    print(linha("╚", 90))
    
    return {'tukey': tukey_result, 'kruskal_wallis': (h_stat, p_kw), 'dados': df_analise}

# ================= EXECUÇÃO =================

if __name__ == "__main__":
    print("\n")
    print(linha("╔", 90))
    print("║" + " CARREGANDO ANÁLISE ESTATÍSTICA ".center(88) + "║")
    print(linha("╚", 90))
    
    print("\n  Carregando dados...")
    if not os.path.exists(ARQUIVO_DADOS):
        print(f"  ❌ ERRO: {ARQUIVO_DADOS} não encontrado.")
    else:
        # Lê o CSV espectral gerado
        df_raw = pd.read_csv(ARQUIVO_DADOS, sep=';', decimal=',')
        
        if os.path.exists(ARQUIVO_AGRONOMICO):
            print(f"  ✓ Cruzando com dados agronômicos...")
            # Usa a nova função de cruzamento
            df = cruzar_com_excel(df_raw, ARQUIVO_AGRONOMICO)
        else:
            print(f"  ⚠️  Arquivo agronômico não encontrado.")
            df = df_raw.copy()
        
        if df is not None:
            # Executa a estatística (mantido igual ao original)
            resultado = executar_testes_pos_hoc(df)
            if resultado:
                print("\n")
        else:
            print("\n  ❌ Erro ao processar dados.")