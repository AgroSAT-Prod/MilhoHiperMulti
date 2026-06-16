import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from itertools import combinations

from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ================= CONFIGURAÇÕES =================
PASTA_DO_SCRIPT = os.path.dirname(os.path.abspath(__file__))
ARQUIVO_DADOS = os.path.join(PASTA_DO_SCRIPT, 'DATASET_IA_PROCESSADO.csv')

SEED = 42

# Conjunto de bandas de interesse
BANDAS_ALVO = [
    430.0, 450.0, 460.0, 500.0, 520.0, 550.0, 600.0, 625.0, 
    640.0, 660.0, 685.0, 720.0, 722.0, 740.0, 970.0, 990.0, 995.0
]

# ================= FUNÇÕES AUXILIARES =================

def extrair_valor_onda(nome_coluna):
    """Extrai o valor numérico da banda (ex: d1_Band_720nm -> 720.0)."""
    try:
        limpo = nome_coluna.replace('d1_', '').replace('Band_', '').replace('nm', '')
        return float(limpo)
    except:
        return None

def selecionar_colunas_por_lista(todas_colunas, bandas_alvo, tolerancia=1.5):
    """Filtra as colunas do DF que correspondem aos comprimentos de onda da lista."""
    selecionadas = []
    bandas_encontradas = []
    
    for banda in bandas_alvo:
        for col in todas_colunas:
            wl = extrair_valor_onda(col)
            if wl is not None and abs(wl - banda) <= tolerancia:
                if col not in selecionadas:
                    selecionadas.append(col)
                    bandas_encontradas.append(banda)
                break
    return selecionadas, bandas_encontradas

def extrair_wavelengths(colunas):
    """Extrai os comprimentos de onda de uma lista de nomes de colunas."""
    return [extrair_valor_onda(col) for col in colunas]

def diagnosticar_dados(X, y, cols_selecionadas):
    """Diagnóstico dos dados de entrada para identificar problemas."""
    print(f"\n{'='*70}")
    print(" DIAGNÓSTICO DOS DADOS")
    print(f"{'='*70}")
    print(f"Forma de X: {X.shape}")
    print(f"Forma de y: {y.shape}")
    print(f"Valores NaN em X: {X.isna().sum().sum()}")
    print(f"Valores NaN em y: {y.isna().sum()}")
    print(f"Valores inf em X: {np.isinf(X).sum().sum()}")
    print(f"Valores inf em y: {np.isinf(y).sum()}")
    print(f"\nEstatísticas de y (Clorofila):")
    print(f"  - Mínimo: {y.min():.4f}")
    print(f"  - Máximo: {y.max():.4f}")
    print(f"  - Média: {y.mean():.4f}")
    print(f"  - Desvio padrão: {y.std():.4f}")
    print(f"  - Amplitude: {y.max() - y.min():.4f}")
    
    if y.std() < 0.1:
        print(f"\n  ⚠ AVISO: Variância muito baixa em y!")
    
    print(f"\nEstatísticas de X (Bandas):")
    for i, col in enumerate(cols_selecionadas):
        X_col = X.iloc[:, i]
        print(f"  {col}: min={X_col.min():.2f}, max={X_col.max():.2f}, std={X_col.std():.2f}")
    
    print(f"\n{'='*70}\n")

# ================= FUNÇÕES DE VIP =================

def calcular_vip(modelo, X, y=None):
    """
    Calcula o VIP (Variable Importance in Projection) para um modelo PLSR.
    O VIP mede a importância de cada variável (banda) na predição.
    
    Interpretação:
    - VIP > 1: variável importante
    - VIP ≈ 1: variável média
    - VIP < 1: variável menos importante
    """
    T = modelo.x_scores_  # (n_samples, n_components)
    W = modelo.x_weights_  # (n_features, n_components)
    Q = modelo.y_loadings_.flatten()  # (n_components,)
    
    m = X.shape[1]  # número de features (bandas)
    n_comp = T.shape[1]  # número de componentes
    
    VIP = np.zeros(m)
    
    # Calcular SS (sum of squares para normalização)
    SS = np.sum(T**2, axis=0) * (Q**2)  # (n_components,)
    SS_total = np.sum(SS)
    
    for i in range(m):
        numerador = 0
        for k in range(n_comp):
            w_norm = np.linalg.norm(W[:, k])
            if w_norm > 0:
                # Contribuição normalizada do peso
                numerador += (W[i, k] / w_norm)**2 * SS[k]
        
        if SS_total != 0:
            VIP[i] = np.sqrt(m * numerador / SS_total)
        else:
            VIP[i] = 0
    
    return VIP

def plotar_vip_bandas(modelo, colunas_X, X, titulo="VIP das Bandas Espectrais"):
    """Plota o VIP de cada banda do modelo PLSR."""
    vip = calcular_vip(modelo, X)
    
    # Criar DataFrame para visualização
    vip_df = pd.DataFrame({
        'Banda': [c.replace('d1_Band_', '').replace('Band_', '').replace('nm', '') for c in colunas_X],
        'VIP': vip
    })
    
    # Ordenar por VIP (descendente para mostrar os mais importantes no topo)
    vip_df = vip_df.sort_values(by='VIP', ascending=True)
    
    plt.figure(figsize=(10, 8))
    cores = ['red' if v > 1 else 'orange' if v > 0.8 else 'lightblue' for v in vip_df['VIP']]
    plt.barh(vip_df['Banda'], vip_df['VIP'], color=cores, edgecolor='black', linewidth=0.5)
    plt.axvline(x=1, color='red', linestyle='--', linewidth=2, label='Threshold VIP=1')
    plt.xlabel('VIP Score', fontsize=12, fontweight='bold')
    plt.ylabel('Bandas Espectrais (nm)', fontsize=12, fontweight='bold')
    plt.title(titulo, fontsize=13, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('vip_bandas.png', dpi=300)
    plt.show()
    
    return vip_df

# ================= FUNÇÕES DE AVALIAÇÃO =================

def avaliar_combinacao(X_train, X_test, y_train, y_test, n_lvs=3):
    """
    Treina e avalia um modelo PLSR para uma combinação de bandas.
    Retorna R² no conjunto de teste.
    """
    try:
        # Validar dados
        if len(X_train) == 0 or len(X_test) == 0:
            return -np.inf, np.inf, None, None
        
        if np.isnan(y_train.values).any() or np.isnan(y_test.values).any():
            return -np.inf, np.inf, None, None
        
        # Ajustar n_lvs se necessário
        n_lvs_ajustado = min(n_lvs, X_train.shape[1], len(X_train) - 1)
        if n_lvs_ajustado < 1:
            n_lvs_ajustado = 1
        
        model = PLSRegression(n_components=n_lvs_ajustado, scale=True)
        model.fit(X_train, y_train.values.reshape(-1, 1))
        y_pred = model.predict(X_test).flatten()
        
        # Validar predição
        if len(y_pred) != len(y_test) or np.isnan(y_pred).any():
            return -np.inf, np.inf, None, None
        
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        return r2, rmse, model, y_pred
    except Exception as e:
        return -np.inf, np.inf, None, None

def otimizar_bandas_por_r2(X, y, colunas_bandas, min_bandas=3, max_bandas=None, 
                            test_size=0.30, n_lvs=3, verbose=True):
    """
    Testa todas as combinações de bandas de min_bandas até max_bandas
    e retorna o subconjunto com melhor R² no conjunto de teste.
    """
    
    if max_bandas is None:
        max_bandas = len(colunas_bandas)
    
    max_bandas = min(max_bandas, len(colunas_bandas))
    
    # Divisão treino/teste (fixa para todas as combinações)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=SEED
    )
    
    resultados = []
    total_combinacoes = sum(
        len(list(combinations(range(len(colunas_bandas)), r))) 
        for r in range(min_bandas, max_bandas + 1)
    )
    
    contador = 0
    
    # Iterar sobre diferentes números de bandas
    for n_bandas in range(min_bandas, max_bandas + 1):
        if verbose:
            print(f"\nTestando combinações com {n_bandas} bandas...")
        
        # Iterar sobre todas as combinações possíveis
        for combo_indices in combinations(range(len(colunas_bandas)), n_bandas):
            contador += 1
            
            # Selecionar bandas
            X_train_subset = X_train.iloc[:, list(combo_indices)].reset_index(drop=True)
            X_test_subset = X_test.iloc[:, list(combo_indices)].reset_index(drop=True)
            y_train_reset = y_train.reset_index(drop=True)
            y_test_reset = y_test.reset_index(drop=True)
            
            # Avaliar
            r2, rmse, modelo, y_pred = avaliar_combinacao(
                X_train_subset, X_test_subset, y_train_reset, y_test_reset, n_lvs
            )
            
            # Armazenar resultado
            bandas_combo = [colunas_bandas[i] for i in combo_indices]
            wavelengths = extrair_wavelengths(bandas_combo)
            
            resultados.append({
                'n_bandas': n_bandas,
                'indices': combo_indices,
                'bandas': bandas_combo,
                'wavelengths': wavelengths,
                'r2': r2,
                'rmse': rmse,
                'modelo': modelo,
                'X_train': X_train_subset,
                'X_test': X_test_subset,
                'y_pred': y_pred,
                'y_test': y_test_reset.values
            })
            
            if verbose and contador % max(1, total_combinacoes // 20) == 0:
                progresso = (contador / total_combinacoes) * 100
                print(f"  Progresso: {progresso:.1f}% | Melhor R² até agora: {max([r['r2'] for r in resultados]):.4f}")
    
    # Converter para DataFrame
    df_resultados = pd.DataFrame(resultados)
    
    # Encontrar melhor combinação
    idx_melhor = df_resultados['r2'].idxmax()
    melhor_linha = df_resultados.loc[idx_melhor]
    
    melhor_combo = melhor_linha['indices']
    melhor_modelo = melhor_linha['modelo']
    melhor_r2 = melhor_linha['r2']
    melhor_rmse = melhor_linha['rmse']
    melhor_bandas = melhor_linha['bandas']
    melhor_wavelengths = melhor_linha['wavelengths']
    melhor_X_test = melhor_linha['X_test']
    
    if verbose:
        print(f"\n{'='*70}")
        print(f" MELHOR COMBINAÇÃO ENCONTRADA")
        print(f"{'='*70}")
        print(f"Número de bandas: {len(melhor_bandas)}")
        print(f"Bandas: {melhor_bandas}")
        print(f"Wavelengths (nm): {sorted([w for w in melhor_wavelengths if w is not None])}")
        print(f"R² (Teste): {melhor_r2:.4f}")
        print(f"RMSE (Teste): {melhor_rmse:.4f}")
        print(f"{'='*70}\n")
    
    return df_resultados, melhor_combo, melhor_modelo, melhor_r2, melhor_rmse, melhor_bandas, melhor_X_test

def plotar_evolucao_r2(df_resultados):
    """Plota a evolução do R² em relação ao número de bandas."""
    resumo = df_resultados.groupby('n_bandas').agg({
        'r2': ['max', 'mean', 'min']
    }).reset_index()
    
    resumo.columns = ['n_bandas', 'r2_max', 'r2_mean', 'r2_min']
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(resumo['n_bandas'], resumo['r2_max'], 'g-o', linewidth=2.5, 
             markersize=8, label='Melhor R²')
    plt.fill_between(resumo['n_bandas'], resumo['r2_min'], resumo['r2_max'], 
                     alpha=0.2, color='green', label='Range (Min-Max)')
    plt.plot(resumo['n_bandas'], resumo['r2_mean'], 'b--s', linewidth=1.5, 
             markersize=6, label='R² Médio')
    
    plt.xlabel('Número de Bandas', fontsize=12)
    plt.ylabel('R² (Correlação)', fontsize=12)
    plt.title('Evolução do R² com Número de Bandas\n(Teste Set)', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.xticks(resumo['n_bandas'])
    plt.tight_layout()
    plt.savefig('evolucao_r2_por_bandas.png', dpi=300)
    plt.show()

def plotar_scatter_melhor_modelo(melhor_linha):
    """Plota dispersão para o melhor modelo."""
    y_test = melhor_linha['y_test']
    y_pred = melhor_linha['y_pred']
    r2 = melhor_linha['r2']
    rmse = melhor_linha['rmse']
    
    # Validar dados
    if y_pred is None or len(y_pred) == 0 or np.isnan(y_pred).any():
        print("⚠ Aviso: Melhor modelo não possui predições válidas.")
        return
    
    plt.figure(figsize=(9, 7))
    plt.scatter(y_test, y_pred, alpha=0.7, color='darkgreen', s=100, edgecolors='k', linewidth=0.5)
    
    # Linha de Perfeição
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2.5, label='1:1 (Ideal)')
    
    plt.title(f"Estimativa de Clorofila com Bandas Ótimas (PLSR)\nR² = {r2:.4f} | RMSE = {rmse:.4f}", 
              fontsize=13, fontweight='bold')
    plt.xlabel("Clorofila Real (Campo)", fontsize=11)
    plt.ylabel("Clorofila Predita (IA)", fontsize=11)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('scatter_melhor_modelo.png', dpi=300)
    plt.show()

def plotar_top_combinacoes(df_resultados, top_n=10):
    """Plota as top N combinações por R²."""
    top_df = df_resultados.nlargest(top_n, 'r2').copy()
    top_df['combo_label'] = top_df['wavelengths'].apply(
        lambda x: ', '.join([f'{w:.0f}' for w in sorted(x) if w is not None])
    )
    
    plt.figure(figsize=(14, 7))
    bars = plt.bar(range(len(top_df)), top_df['r2'], color=plt.cm.viridis(np.linspace(0, 1, len(top_df))))
    plt.xticks(range(len(top_df)), top_df['combo_label'], rotation=45, ha='right')
    plt.ylabel('R² (Teste)', fontsize=12)
    plt.xlabel('Combinação de Bandas (nm)', fontsize=12)
    plt.title(f'Top {top_n} Combinações de Bandas por R²', fontsize=13, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('top_combinacoes_bandas.png', dpi=300)
    plt.show()

# ================= EXECUÇÃO PRINCIPAL =================

if __name__ == "__main__":
    print("=== OTIMIZAÇÃO DE BANDAS PARA ESTIMATIVA DE CLOROFILA COM PLSR ===\n")
    
    # 1. Carregar Dados
    if not os.path.exists(ARQUIVO_DADOS):
        sys.exit(f"ERRO: Arquivo {ARQUIVO_DADOS} não encontrado.")
    
    df = pd.read_csv(ARQUIVO_DADOS, sep=';', decimal=',')
    
    # 2. Limpeza
    df_cloro = df.dropna(subset=['Y_Clorofila']).copy()
    
    # 3. Seleção das Bandas Específicas
    cols_totais = [c for c in df_cloro.columns if 'Band_' in c]
    cols_selecionadas, bandas_encontradas = selecionar_colunas_por_lista(cols_totais, BANDAS_ALVO)
    
    if len(cols_selecionadas) == 0:
        sys.exit("ERRO: Nenhuma das bandas solicitadas foi encontrada nas colunas do CSV.")

    print(f"Amostras disponíveis: {len(df_cloro)}")
    print(f"Bandas mapeadas do sensor: {len(cols_selecionadas)}")
    print(f"Bandas encontradas: {sorted([extrair_valor_onda(c) for c in cols_selecionadas])}\n")

    # Preparar matrizes
    X = df_cloro[cols_selecionadas].apply(pd.to_numeric, errors='coerce').fillna(0)
    y = pd.to_numeric(df_cloro['Y_Clorofila'], errors='coerce')
    
    # Diagnóstico
    diagnosticar_dados(X, y, cols_selecionadas)

    # 4. OTIMIZAÇÃO: Testar todas as combinações
    print("Iniciando busca exaustiva por combinações ótimas de bandas...")
    print("Isso pode levar alguns minutos...\n")
    
    df_resultados, melhor_combo, melhor_modelo, melhor_r2, melhor_rmse, melhor_bandas, melhor_X_test = otimizar_bandas_por_r2(
        X, y, cols_selecionadas,
        min_bandas=3,
        max_bandas=len(cols_selecionadas),
        test_size=0.30,
        n_lvs=3,
        verbose=True
    )

    # 5. Visualizações
    print("Gerando visualizações...\n")
    plotar_evolucao_r2(df_resultados)
    
    melhor_idx = df_resultados['r2'].idxmax()
    
    # Verificar se há resultados válidos
    if df_resultados['r2'].max() > -np.inf:
        plotar_scatter_melhor_modelo(df_resultados.loc[melhor_idx])
        plotar_top_combinacoes(df_resultados, top_n=15)
        
        # Plotar VIP do melhor modelo
        print("\nAnalisando importância das bandas (VIP)...\n")
        melhor_bandas_lista = df_resultados.loc[melhor_idx, 'bandas']
        vip_df = plotar_vip_bandas(melhor_modelo, melhor_bandas_lista, melhor_X_test,
                                    titulo=f"VIP das Bandas - Melhor Modelo (R²={melhor_r2:.4f})")
        
        print("\nRanking VIP das Bandas:")
        print(vip_df.sort_values('VIP', ascending=False).to_string(index=False))
    else:
        print("⚠ AVISO: Nenhuma combinação produziu um modelo válido.")
        print("  Possíveis causas:")
        print("  - Dataset muito pequeno (76 amostras)")
        print("  - Valores de clorofila com pouca variância")
        print("  - Dados contêm NaN não detectados")
        print("\nVerifique os dados de entrada.")
    
    # 6. Relatório Detalhado
    print(f"\n{'='*70}")
    print(" RELATÓRIO FINAL DE OTIMIZAÇÃO")
    print(f"{'='*70}")
    print(f"\nMELHOR CONFIGURAÇÃO ENCONTRADA:")
    print(f"  • Número de bandas: {len(melhor_bandas)}")
    print(f"  • Bandas selecionadas: {[extrair_valor_onda(b) for b in melhor_bandas]}")
    print(f"  • R² (Teste): {melhor_r2:.4f}")
    print(f"  • RMSE (Teste): {melhor_rmse:.4f}")
    
    print(f"\nESTATÍSTICAS GERAIS:")
    print(f"  • Total de combinações testadas: {len(df_resultados)}")
    print(f"  • R² máximo encontrado: {df_resultados['r2'].max():.4f}")
    print(f"  • R² mínimo encontrado: {df_resultados['r2'].min():.4f}")
    print(f"  • R² médio: {df_resultados['r2'].mean():.4f}")
    
    print(f"\n{'='*70}\n")
    print("Processo concluído com sucesso!")