import pandas as pd
import os
import unicodedata

# ================= CONFIGURAÇÕES =================
ARQUIVO_DADOS = 'spectral_indices_rois2.csv'
ARQUIVO_AGRONOMICO = os.path.join('Dataset', 'PlanilhaFiltrada2.csv')

# ================= FUNÇÕES AUXILIARES =================

def linha(char="═", comprimento=90):
    return char * comprimento

def normalizar_texto(texto):
    """Remove acentos e deixa minúsculo para facilitar busca de colunas."""
    if not isinstance(texto, str):
        return str(texto)
    nfkd = unicodedata.normalize('NFKD', texto)
    return "".join([c for c in nfkd if not unicodedata.combining(c)]).lower().strip()

def imprimir_tabela(titulo, df_agrupado, colunas_agrupamento):
    """Função genérica para formatar e imprimir tabelas de contagem."""
    print("\n" + linha("─", 90))
    print(f" {titulo}".center(90))
    print(linha("─", 90))

    # Prepara cabeçalho
    headers = [c for c in colunas_agrupamento] + ['Qtd_Amostras']
    
    # Print Cabeçalho
    print("  " + " | ".join([f"{h:<25}" for h in headers]))
    print("  " + "-" * (28 * len(headers)))

    total_check = 0
    for idx, row in df_agrupado.iterrows():
        # Formata valores (se for número float, arredonda, se não, string normal)
        valores = []
        for c in colunas_agrupamento:
            val = row[c]
            if isinstance(val, float):
                valores.append(f"{val:.0f}")
            else:
                valores.append(str(val))
        
        qtd = row['Qtd_Amostras']
        total_check += qtd
        print("  " + " | ".join([f"{v:<25}" for v in valores]) + f" | {qtd:>10}")

    print("  " + "-" * (28 * len(headers)))
    print(f"  {'TOTAL':<25}   | {total_check:>10}")

def carregar_e_analisar():
    print("\n" + linha("╔", 90))
    print("║" + " ANÁLISE DETALHADA: TRATAMENTOS E DOSES ".center(88) + "║")
    print(linha("╚", 90))

    # --- 1. CARREGAMENTO (Igual ao anterior) ---
    print(f"\n📂 Lendo arquivo espectral: {ARQUIVO_DADOS}")
    if not os.path.exists(ARQUIVO_DADOS):
        print(f"❌ ERRO: {ARQUIVO_DADOS} não encontrado.")
        return

    try:
        df_spec = pd.read_csv(ARQUIVO_DADOS, sep=';', decimal=',')
        if len(df_spec.columns) < 2:
            df_spec = pd.read_csv(ARQUIVO_DADOS, sep=',', decimal='.')
        
        df_spec['ID_Numeric'] = pd.to_numeric(df_spec['ID_Amostra'], errors='coerce')
        print(f"   ✓ {len(df_spec)} linhas carregadas.")
    except Exception as e:
        print(f"❌ Erro CSV: {e}")
        return

    # --- 2. AGRONÔMICO ---
    print(f"📂 Lendo arquivo agronômico: {ARQUIVO_AGRONOMICO}")
    if not os.path.exists(ARQUIVO_AGRONOMICO): return

    try:
        if ARQUIVO_AGRONOMICO.lower().endswith('.csv'):
            df_agro = pd.read_csv(ARQUIVO_AGRONOMICO, header=3, sep=';', decimal=',', encoding='latin1')
        else:
            df_agro = pd.read_excel(ARQUIVO_AGRONOMICO, header=3)
    except Exception as e:
        print(f"❌ Erro Agro: {e}")
        return

    # --- 3. MAPEAMENTO ---
    cols_originais = list(df_agro.columns)
    cols_norm = [normalizar_texto(c) for c in cols_originais]

    idx_id = next((i for i, c in enumerate(cols_norm) if c == 'id' or ('id' in c and 'parcela' in c)), -1)
    col_id = cols_originais[idx_id]

    idx_trat = next((i for i, c in enumerate(cols_norm) if 'tratamento' in c), -1)
    col_trat = cols_originais[idx_trat] if idx_trat != -1 else None

    idx_dose = next((i for i, c in enumerate(cols_norm) if 'dose' in c), -1)
    col_dose = cols_originais[idx_dose] if idx_dose != -1 else None

    print(f"   ✓ Colunas: ID='{col_id}' | Tratamento='{col_trat}' | Dose='{col_dose}'")

    # --- 4. MERGE ---
    df_agro[col_id] = pd.to_numeric(df_agro[col_id], errors='coerce')
    df_agro = df_agro.dropna(subset=[col_id])

    cols_to_merge = [col_id]
    if col_trat: cols_to_merge.append(col_trat)
    if col_dose and col_dose != col_trat: cols_to_merge.append(col_dose)

    df_final = df_spec.merge(df_agro[cols_to_merge], left_on='ID_Numeric', right_on=col_id, how='inner')
    print(f"   ✓ Amostras vinculadas: {len(df_final)}")

    # ================= 5. GERAÇÃO DOS RELATÓRIOS INDIVIDUAIS =================
    
    # RELATÓRIO 1: APENAS POR DOSE
    if col_dose:
        resumo_dose = df_final.groupby(col_dose).size().reset_index(name='Qtd_Amostras')
        resumo_dose = resumo_dose.sort_values(by=col_dose)
        imprimir_tabela("RESUMO POR DOSE DE NITROGÊNIO", resumo_dose, [col_dose])

    # RELATÓRIO 2: APENAS POR TRATAMENTO
    if col_trat:
        resumo_trat = df_final.groupby(col_trat).size().reset_index(name='Qtd_Amostras')
        resumo_trat = resumo_trat.sort_values(by='Qtd_Amostras', ascending=False)
        imprimir_tabela("RESUMO POR TIPO DE TRATAMENTO", resumo_trat, [col_trat])

    # RELATÓRIO 3: DETALHADO (INTERAÇÃO)
    if col_trat and col_dose:
        cols_mistas = [col_trat, col_dose]
        resumo_misto = df_final.groupby(cols_mistas).size().reset_index(name='Qtd_Amostras')
        resumo_misto = resumo_misto.sort_values(by=cols_mistas)
        imprimir_tabela("DETALHADO (TRATAMENTO x DOSE)", resumo_misto, cols_mistas)

    print("\n" + linha("═", 90))

if __name__ == "__main__":
    carregar_e_analisar()