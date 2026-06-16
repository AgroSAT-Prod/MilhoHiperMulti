import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# ================= CONFIGURAÇÕES =================
PASTA_DO_SCRIPT = os.path.dirname(os.path.abspath(__file__))
ARQUIVO_DADOS = os.path.join(PASTA_DO_SCRIPT, 'DATASET_IA_PROCESSADO.csv')

# Quão parecidas as bandas precisam ser para serem consideradas "Gêmeas"?
# 0.98 significa 98% de similaridade
LIMITE_COLINEARIDADE = 0.98 

# Intervalos de interesse (Focando onde importa)
INTERVALOS_DE_INTERESSE = [
    (520, 580),   # Verde
    (690, 760)    # Red Edge
]

def extrair_valor_onda(nome_coluna):
    try:
        limpo = nome_coluna.replace('d1_', '').replace('Band_', '').replace('nm', '')
        return float(limpo)
    except:
        return 0.0

def filtrar_bandas(colunas, intervalos):
    if not intervalos: return colunas
    cols = []
    for col in colunas:
        wl = extrair_valor_onda(col)
        if wl > 0:
            for (inicio, fim) in intervalos:
                if inicio <= wl <= fim:
                    cols.append(col)
                    break
    return cols

if __name__ == "__main__":
    if not os.path.exists(ARQUIVO_DADOS):
        sys.exit("Erro: Arquivo não encontrado.")
        
    df = pd.read_csv(ARQUIVO_DADOS, sep=';', decimal=',')
    
    # Prepara dados
    cols_todas = [c for c in df.columns if c.startswith('d1_Band_') or c.startswith('Band_')]
    cols_analise = filtrar_bandas(cols_todas, INTERVALOS_DE_INTERESSE)
    
    if len(cols_analise) == 0:
        sys.exit("Nenhuma banda encontrada nos intervalos.")

    print(f"Analisando colinearidade entre {len(cols_analise)} bandas...")
    X = df[cols_analise].apply(pd.to_numeric, errors='coerce').fillna(0)

    # 1. CÁLCULO DA MATRIZ DE CORRELAÇÃO
    # Usamos Pearson (linear) pois colinearidade estrita geralmente é linear
    corr_matrix = X.corr(method='pearson').abs()

    # 2. PLOTAGEM DO HEATMAP
    plt.figure(figsize=(12, 10))
    
    # Mascara a parte superior do triângulo (pois é espelhado) para limpar o visual
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(
        corr_matrix, 
        mask=mask,
        cmap='RdYlBu_r', # Vermelho = Muito Colinear, Azul = Diferente
        vmax=1.0, 
        vmin=0.0,
        square=True, 
        linewidths=.5,
        cbar_kws={"shrink": .5, "label": "Grau de Colinearidade (0 a 1)"}
    )
    
    plt.title("Mapa de Colinearidade Espectral\n(Áreas Vermelhas = Informação Redundante)", fontsize=16)
    plt.xlabel("Bandas (nm)")
    plt.ylabel("Bandas (nm)")
    plt.tight_layout()
    plt.show()

    # 3. LISTAGEM DOS "GÊMEOS" (Pares altamente correlacionados)
    # Transforma a matriz em uma lista de pares
    pares_redundantes = corr_matrix.unstack().sort_values(ascending=False)
    
    # Remove auto-correlação (banda com ela mesma = 1.0) e duplicatas
    pares_redundantes = pares_redundantes[pares_redundantes < 1.0] # Remove diagonal principal
    
    # Filtra pelo limite
    pares_altos = pares_redundantes[pares_redundantes > LIMITE_COLINEARIDADE]

    # Para não imprimir duplicados (A-B e B-A), vamos usar um set
    vistos = set()
    
    print(f"\n{'='*10} ALERTA DE REDUNDÂNCIA (R > {LIMITE_COLINEARIDADE}) {'='*10}")
    print(f"Total de pares redundantes encontrados: {len(pares_altos) // 2}")
    
    count = 0
    for (banda_a, banda_b), valor in pares_altos.items():
        # Lógica para evitar duplicatas A-B e B-A
        par_ordenado = tuple(sorted((banda_a, banda_b)))
        if par_ordenado not in vistos:
            vistos.add(par_ordenado)
            wl_a = extrair_valor_onda(banda_a)
            wl_b = extrair_valor_onda(banda_b)
            
            # Só mostra as primeiras 20 para não poluir o terminal
            if count < 20:
                print(f" ! {wl_a}nm e {wl_b}nm são {valor:.2%} idênticas.")
            count += 1
            
    if count >= 20:
        print(f"... e mais {count - 20} pares.")

    print("\nCONCLUSÃO:")
    print("Se você vê grandes blocos vermelhos no gráfico, o SPA/CARS é obrigatório.")
    print("Se o gráfico for todo azul/amarelo, seus dados já são distintos.")