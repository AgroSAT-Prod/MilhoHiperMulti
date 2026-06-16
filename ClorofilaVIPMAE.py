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
ARQUIVO_DADOS = os.path.join(PASTA_DO_SCRIPT, 'DATASET_IA_PROCESSADO_JAN2026_VALIDO.csv')
# 'DATASET_IA_PROCESSADO_JAN2026_VALIDO.csv'
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
    """Calcula o VIP (Variable Importance in Projection)."""
    T = modelo.x_scores_
    W = modelo.x_weights_
    Q = modelo.y_loadings_.flatten()
    
    m = X.shape[1]
    n_comp = T.shape[1]
    
    VIP = np.zeros(m)
    SS = np.sum(T**2, axis=0) * (Q**2)
    SS_total = np.sum(SS)
    
    for i in range(m):
        numerador = 0
        for k in range(n_comp):
            w_norm = np.linalg.norm(W[:, k])
            if w_norm > 0:
                numerador += (W[i, k] / w_norm)**2 * SS[k]
        
        if SS_total != 0:
            VIP[i] = np.sqrt(m * numerador / SS_total)
        else:
            VIP[i] = 0
    
    return VIP

def plotar_vip_bandas(modelo, colunas_X, X, titulo="VIP das Bandas Espectrais"):
    """Plota o VIP de cada banda do modelo PLSR."""
    vip = calcular_vip(modelo, X)
    
    vip_df = pd.DataFrame({
        'Banda': [c.replace('d1_Band_', '').replace('Band_', '').replace('nm', '') for c in colunas_X],
        'VIP': vip
    })
    
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

def plotar_coeficientes_regressao(modelo, colunas_X, titulo="Coeficientes de Regressão (Beta)"):
    """
    Plota os coeficientes de regressão do modelo PLSR.
    Indica a direção e magnitude da influência de cada banda.
    """
    # Em modelos PLSR do scikit-learn, os coeficientes ficam em modelo.coef_
    coefs = modelo.coef_.flatten()
    
    coef_df = pd.DataFrame({
        'Banda': [c.replace('d1_Band_', '').replace('Band_', '').replace('nm', '') for c in colunas_X],
        'Coeficiente': coefs
    })
    
    # Ordenar por magnitude absoluta ou valor real (aqui ordenamos por valor real para visualização)
    coef_df = coef_df.sort_values(by='Coeficiente', ascending=True)
    
    plt.figure(figsize=(10, 8))
    
    # Cores: Verde para positivo (aumenta clorofila), Vermelho para negativo (diminui clorofila)
    cores = ['green' if c > 0 else 'red' for c in coef_df['Coeficiente']]
    
    plt.barh(coef_df['Banda'], coef_df['Coeficiente'], color=cores, edgecolor='black', linewidth=0.5, alpha=0.7)
    plt.axvline(x=0, color='black', linestyle='-', linewidth=1)
    
    plt.xlabel('Valor do Coeficiente (Beta)', fontsize=12, fontweight='bold')
    plt.ylabel('Bandas Espectrais (nm)', fontsize=12, fontweight='bold')
    plt.title(titulo, fontsize=13, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    
    # Adicionar anotação explicativa
    plt.figtext(0.02, 0.02, 
                "Verde (>0): Correlação Positiva | Vermelho (<0): Correlação Negativa", 
                fontsize=9, style='italic', bbox={"facecolor":"white", "alpha":0.8, "pad":5})
    
    plt.tight_layout()
    plt.savefig('coeficientes_regressao.png', dpi=300)
    plt.show()
    
    return coef_df

# ================= FUNÇÕES DE AVALIAÇÃO =================

def avaliar_combinacao(X_train, X_test, y_train, y_test, n_lvs=3):
    """
    Treina e avalia um modelo PLSR.
    Retorna todas as métricas: R², RMSE, MAE, MSE
    """
    try:
        # Validar dados
        if len(X_train) == 0 or len(X_test) == 0:
            return -np.inf, np.inf, np.inf, np.inf, None, None
        
        if np.isnan(y_train.values).any() or np.isnan(y_test.values).any():
            return -np.inf, np.inf, np.inf, np.inf, None, None
        
        n_lvs_ajustado = min(n_lvs, X_train.shape[1], len(X_train) - 1)
        if n_lvs_ajustado < 1:
            n_lvs_ajustado = 1
        
        model = PLSRegression(n_components=n_lvs_ajustado, scale=True)
        model.fit(X_train, y_train.values.reshape(-1, 1))
        y_pred = model.predict(X_test).flatten()
        
        if len(y_pred) != len(y_test) or np.isnan(y_pred).any():
            return -np.inf, np.inf, np.inf, np.inf, None, None
        
        # Calcular todas as métricas
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        
        return r2, rmse, mae, mse, model, y_pred
    except Exception as e:
        return -np.inf, np.inf, np.inf, np.inf, None, None

def otimizar_bandas_por_r2(X, y, colunas_bandas, min_bandas=3, max_bandas=None, 
                            test_size=0.30, n_lvs=3, verbose=True):
    """
    Testa combinações e retorna a melhor baseada em R².
    Calcula todas as métricas: R², RMSE, MAE, MSE
    """
    
    if max_bandas is None:
        max_bandas = len(colunas_bandas)
    
    max_bandas = min(max_bandas, len(colunas_bandas))
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=SEED
    )
    
    resultados = []
    total_combinacoes = sum(
        len(list(combinations(range(len(colunas_bandas)), r))) 
        for r in range(min_bandas, max_bandas + 1)
    )
    
    contador = 0
    
    for n_bandas in range(min_bandas, max_bandas + 1):
        if verbose:
            print(f"\nTestando combinações com {n_bandas} bandas...")
        
        for combo_indices in combinations(range(len(colunas_bandas)), n_bandas):
            contador += 1
            
            X_train_subset = X_train.iloc[:, list(combo_indices)].reset_index(drop=True)
            X_test_subset = X_test.iloc[:, list(combo_indices)].reset_index(drop=True)
            y_train_reset = y_train.reset_index(drop=True)
            y_test_reset = y_test.reset_index(drop=True)
            
            # Avaliar com todas as métricas
            r2, rmse, mae, mse, modelo, y_pred = avaliar_combinacao(
                X_train_subset, X_test_subset, y_train_reset, y_test_reset, n_lvs
            )
            
            bandas_combo = [colunas_bandas[i] for i in combo_indices]
            wavelengths = extrair_wavelengths(bandas_combo)
            
            resultados.append({
                'n_bandas': n_bandas,
                'indices': combo_indices,
                'bandas': bandas_combo,
                'wavelengths': wavelengths,
                'r2': r2,
                'rmse': rmse,
                'mae': mae,
                'mse': mse,
                'modelo': modelo,
                'X_train': X_train_subset,
                'X_test': X_test_subset,
                'y_pred': y_pred,
                'y_test': y_test_reset.values
            })
            
            if verbose and contador % max(1, total_combinacoes // 20) == 0:
                progresso = (contador / total_combinacoes) * 100
                print(f"  Progresso: {progresso:.1f}% | Melhor R² até agora: {max([r['r2'] for r in resultados]):.4f}")
    
    df_resultados = pd.DataFrame(resultados)
    
    idx_melhor = df_resultados['r2'].idxmax()
    melhor_linha = df_resultados.loc[idx_melhor]
    
    melhor_combo = melhor_linha['indices']
    melhor_modelo = melhor_linha['modelo']
    melhor_r2 = melhor_linha['r2']
    melhor_rmse = melhor_linha['rmse']
    melhor_mae = melhor_linha['mae']
    melhor_mse = melhor_linha['mse']
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
        print(f"\nMÉTRICAS DE DESEMPENHO (Teste):")
        print(f"  • R²:   {melhor_r2:.4f}")
        print(f"  • RMSE: {melhor_rmse:.4f}")
        print(f"  • MAE:  {melhor_mae:.4f}")
        print(f"  • MSE:  {melhor_mse:.4f}")
        print(f"{'='*70}\n")
    
    return (df_resultados, melhor_combo, melhor_modelo, melhor_r2, melhor_rmse, 
            melhor_mae, melhor_mse, melhor_bandas, melhor_X_test)

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
    """Plota dispersão para o melhor modelo com todas as métricas."""
    y_test = melhor_linha['y_test']
    y_pred = melhor_linha['y_pred']
    r2 = melhor_linha['r2']
    rmse = melhor_linha['rmse']
    mae = melhor_linha['mae']
    mse = melhor_linha['mse']
    
    if y_pred is None or len(y_pred) == 0 or np.isnan(y_pred).any():
        print("⚠ Aviso: Melhor modelo não possui predições válidas.")
        return
    
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred, alpha=0.7, color='darkgreen', s=100, edgecolors='k', linewidth=0.5)
    
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2.5, label='1:1 (Ideal)')
    
    # Título com todas as métricas
    titulo = (f"Estimativa de Clorofila com Bandas Ótimas (PLSR)\n"
              f"R² = {r2:.4f} | RMSE = {rmse:.4f} | MAE = {mae:.4f} | MSE = {mse:.4f}")
    
    plt.title(titulo, fontsize=12, fontweight='bold')
    plt.xlabel("Clorofila Real (Campo)", fontsize=11)
    plt.ylabel("Clorofila Predita (IA)", fontsize=11)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('scatter_melhor_modelo.png', dpi=300)
    plt.show()

def plotar_metricas_comparativas(df_resultados):
    """Plota comparação entre todas as métricas por número de bandas."""
    resumo = df_resultados.groupby('n_bandas').agg({
        'r2': 'max',
        'rmse': 'min',
        'mae': 'min',
        'mse': 'min'
    }).reset_index()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # R²
    axes[0, 0].plot(resumo['n_bandas'], resumo['r2'], 'o-', linewidth=2, markersize=8, color='green')
    axes[0, 0].set_xlabel('Número de Bandas', fontsize=11)
    axes[0, 0].set_ylabel('R² (máximo)', fontsize=11)
    axes[0, 0].set_title('Coeficiente de Determinação (R²)', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xticks(resumo['n_bandas'])
    
    # RMSE
    axes[0, 1].plot(resumo['n_bandas'], resumo['rmse'], 'o-', linewidth=2, markersize=8, color='red')
    axes[0, 1].set_xlabel('Número de Bandas', fontsize=11)
    axes[0, 1].set_ylabel('RMSE (mínimo)', fontsize=11)
    axes[0, 1].set_title('Root Mean Squared Error (RMSE)', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xticks(resumo['n_bandas'])
    
    # MAE
    axes[1, 0].plot(resumo['n_bandas'], resumo['mae'], 'o-', linewidth=2, markersize=8, color='blue')
    axes[1, 0].set_xlabel('Número de Bandas', fontsize=11)
    axes[1, 0].set_ylabel('MAE (mínimo)', fontsize=11)
    axes[1, 0].set_title('Mean Absolute Error (MAE)', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xticks(resumo['n_bandas'])
    
    # MSE
    axes[1, 1].plot(resumo['n_bandas'], resumo['mse'], 'o-', linewidth=2, markersize=8, color='orange')
    axes[1, 1].set_xlabel('Número de Bandas', fontsize=11)
    axes[1, 1].set_ylabel('MSE (mínimo)', fontsize=11)
    axes[1, 1].set_title('Mean Squared Error (MSE)', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xticks(resumo['n_bandas'])
    
    plt.tight_layout()
    plt.savefig('metricas_comparativas.png', dpi=300)
    plt.show()

def plotar_distribuicao_metricas(df_resultados):
    """
    Plota a distribuição das métricas usando histogramas e boxplots.
    Visualização mais clara para grandes quantidades de combinações.
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    metricas = ['r2', 'rmse', 'mae', 'mse']
    titulos = ['R² (Coeficiente de Determinação)', 'RMSE', 'MAE', 'MSE']
    cores = ['green', 'red', 'blue', 'orange']
    
    for idx, (metrica, titulo, cor) in enumerate(zip(metricas, titulos, cores)):
        # Histograma
        ax_hist = fig.add_subplot(gs[0, idx])
        valores = df_resultados[metrica]
        ax_hist.hist(valores, bins=50, color=cor, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Adicionar linha vertical no melhor valor
        if metrica == 'r2':
            melhor = valores.max()
            ax_hist.axvline(melhor, color='darkred', linestyle='--', linewidth=2, label=f'Melhor: {melhor:.4f}')
        else:
            melhor = valores.min()
            ax_hist.axvline(melhor, color='darkred', linestyle='--', linewidth=2, label=f'Melhor: {melhor:.4f}')
        
        # Média
        media = valores.mean()
        ax_hist.axvline(media, color='black', linestyle=':', linewidth=2, label=f'Média: {media:.4f}')
        
        ax_hist.set_xlabel(titulo, fontsize=10, fontweight='bold')
        ax_hist.set_ylabel('Frequência', fontsize=10)
        ax_hist.legend(fontsize=8)
        ax_hist.grid(True, alpha=0.3)
        
        # Boxplot
        ax_box = fig.add_subplot(gs[1, idx])
        bp = ax_box.boxplot(valores, vert=True, patch_artist=True, widths=0.6)
        bp['boxes'][0].set_facecolor(cor)
        bp['boxes'][0].set_alpha(0.7)
        ax_box.set_ylabel(titulo, fontsize=10, fontweight='bold')
        ax_box.grid(True, alpha=0.3, axis='y')
        ax_box.set_xticklabels([''])
        
    # Gráfico combinado: R² vs RMSE (scatter mais informativo)
    ax_scatter = fig.add_subplot(gs[2, :2])
    scatter = ax_scatter.scatter(df_resultados['r2'], df_resultados['rmse'], 
                                 c=df_resultados['n_bandas'], cmap='viridis',
                                 alpha=0.5, s=20, edgecolors='none')
    ax_scatter.set_xlabel('R² (Coeficiente de Determinação)', fontsize=11, fontweight='bold')
    ax_scatter.set_ylabel('RMSE', fontsize=11, fontweight='bold')
    ax_scatter.set_title('Relação entre R² e RMSE', fontsize=12, fontweight='bold')
    ax_scatter.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax_scatter)
    cbar.set_label('Número de Bandas', fontsize=10)
    
    # Marcar o melhor ponto
    melhor_idx = df_resultados['r2'].idxmax()
    melhor_r2 = df_resultados.loc[melhor_idx, 'r2']
    melhor_rmse = df_resultados.loc[melhor_idx, 'rmse']
    ax_scatter.scatter(melhor_r2, melhor_rmse, color='red', s=200, marker='*', 
                      edgecolors='black', linewidth=2, label='Melhor Modelo', zorder=5)
    ax_scatter.legend(fontsize=10)
    
    # Gráfico combinado: MAE vs MSE
    ax_scatter2 = fig.add_subplot(gs[2, 2:])
    scatter2 = ax_scatter2.scatter(df_resultados['mae'], df_resultados['mse'], 
                                   c=df_resultados['n_bandas'], cmap='viridis',
                                   alpha=0.5, s=20, edgecolors='none')
    ax_scatter2.set_xlabel('MAE (Mean Absolute Error)', fontsize=11, fontweight='bold')
    ax_scatter2.set_ylabel('MSE (Mean Squared Error)', fontsize=11, fontweight='bold')
    ax_scatter2.set_title('Relação entre MAE e MSE', fontsize=12, fontweight='bold')
    ax_scatter2.grid(True, alpha=0.3)
    cbar2 = plt.colorbar(scatter2, ax=ax_scatter2)
    cbar2.set_label('Número de Bandas', fontsize=10)
    
    # Marcar o melhor ponto
    melhor_mae = df_resultados.loc[melhor_idx, 'mae']
    melhor_mse = df_resultados.loc[melhor_idx, 'mse']
    ax_scatter2.scatter(melhor_mae, melhor_mse, color='red', s=200, marker='*', 
                       edgecolors='black', linewidth=2, label='Melhor Modelo', zorder=5)
    ax_scatter2.legend(fontsize=10)
    
    plt.suptitle(f'Distribuição de Métricas - {len(df_resultados)} Combinações Testadas', 
                 fontsize=14, fontweight='bold')
    plt.savefig('distribuicao_metricas.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Estatísticas resumidas
    print(f"\n{'='*70}")
    print("ESTATÍSTICAS DAS COMBINAÇÕES TESTADAS")
    print(f"{'='*70}")
    print(f"Total de combinações: {len(df_resultados)}")
    
    for metrica, nome in zip(['r2', 'rmse', 'mae', 'mse'], ['R²', 'RMSE', 'MAE', 'MSE']):
        valores = df_resultados[metrica]
        if metrica == 'r2':
            print(f"\n{nome}:")
            print(f"  • Melhor (máximo):  {valores.max():.4f}")
            print(f"  • Pior (mínimo):    {valores.min():.4f}")
        else:
            print(f"\n{nome}:")
            print(f"  • Melhor (mínimo):  {valores.min():.4f}")
            print(f"  • Pior (máximo):    {valores.max():.4f}")
        print(f"  • Média:            {valores.mean():.4f}")
        print(f"  • Mediana:          {valores.median():.4f}")
        print(f"  • Desvio padrão:    {valores.std():.4f}")
        print(f"  • Quartil 25%:      {valores.quantile(0.25):.4f}")
        print(f"  • Quartil 75%:      {valores.quantile(0.75):.4f}")
    
    print(f"{'='*70}\n")

def plotar_metricas_por_n_bandas_detalhado(df_resultados):
    """
    Plota violinplot/boxplot das métricas agrupadas por número de bandas.
    Mostra a distribuição de desempenho para cada quantidade de bandas.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    metricas = ['r2', 'rmse', 'mae', 'mse']
    titulos = ['R²', 'RMSE', 'MAE', 'MSE']
    cores_paleta = ['Greens', 'Reds', 'Blues', 'Oranges']
    
    for idx, (ax, metrica, titulo, paleta) in enumerate(zip(axes.flat, metricas, titulos, cores_paleta)):
        # Criar violinplot
        parts = ax.violinplot([df_resultados[df_resultados['n_bandas'] == n][metrica].values 
                               for n in sorted(df_resultados['n_bandas'].unique())],
                              positions=sorted(df_resultados['n_bandas'].unique()),
                              widths=0.7, showmeans=True, showmedians=True)
        
        # Colorir os violins
        for pc in parts['bodies']:
            pc.set_facecolor(plt.cm.get_cmap(paleta)(0.6))
            pc.set_alpha(0.7)
        
        # Adicionar linha do melhor valor
        if metrica == 'r2':
            melhor = df_resultados[metrica].max()
            label_melhor = f'Melhor R²: {melhor:.4f}'
        else:
            melhor = df_resultados[metrica].min()
            label_melhor = f'Melhor {titulo}: {melhor:.4f}'
        
        ax.axhline(melhor, color='red', linestyle='--', linewidth=2, label=label_melhor, alpha=0.8)
        
        ax.set_xlabel('Número de Bandas', fontsize=11, fontweight='bold')
        ax.set_ylabel(titulo, fontsize=11, fontweight='bold')
        ax.set_title(f'Distribuição de {titulo} por Número de Bandas', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=9)
        ax.set_xticks(sorted(df_resultados['n_bandas'].unique()))
    
    plt.suptitle('Análise de Desempenho por Número de Bandas', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('metricas_por_n_bandas_violinplot.png', dpi=300, bbox_inches='tight')
    plt.show()

def plotar_top_combinacoes(df_resultados, top_n=10):
    """Plota as top N combinações por R² com todas as métricas."""
    top_df = df_resultados.nlargest(top_n, 'r2').copy()
    top_df['combo_label'] = top_df['wavelengths'].apply(
        lambda x: ', '.join([f'{w:.0f}' for w in sorted(x) if w is not None])
    )
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    x_pos = range(len(top_df))
    
    # R²
    axes[0, 0].bar(x_pos, top_df['r2'], color=plt.cm.Greens(np.linspace(0.4, 1, len(top_df))))
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(top_df['combo_label'], rotation=45, ha='right', fontsize=8)
    axes[0, 0].set_ylabel('R²', fontsize=11)
    axes[0, 0].set_title(f'Top {top_n} - R² (maior é melhor)', fontsize=12, fontweight='bold')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # RMSE
    axes[0, 1].bar(x_pos, top_df['rmse'], color=plt.cm.Reds(np.linspace(0.4, 1, len(top_df))))
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(top_df['combo_label'], rotation=45, ha='right', fontsize=8)
    axes[0, 1].set_ylabel('RMSE', fontsize=11)
    axes[0, 1].set_title(f'Top {top_n} - RMSE (menor é melhor)', fontsize=12, fontweight='bold')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # MAE
    axes[1, 0].bar(x_pos, top_df['mae'], color=plt.cm.Blues(np.linspace(0.4, 1, len(top_df))))
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(top_df['combo_label'], rotation=45, ha='right', fontsize=8)
    axes[1, 0].set_ylabel('MAE', fontsize=11)
    axes[1, 0].set_title(f'Top {top_n} - MAE (menor é melhor)', fontsize=12, fontweight='bold')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # MSE
    axes[1, 1].bar(x_pos, top_df['mse'], color=plt.cm.Oranges(np.linspace(0.4, 1, len(top_df))))
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(top_df['combo_label'], rotation=45, ha='right', fontsize=8)
    axes[1, 1].set_ylabel('MSE', fontsize=11)
    axes[1, 1].set_title(f'Top {top_n} - MSE (menor é melhor)', fontsize=12, fontweight='bold')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('top_combinacoes_todas_metricas.png', dpi=300)
    plt.show()

def gerar_tabela_top_resultados(df_resultados, top_n=20):
    """Gera e salva tabela com os top N resultados."""
    top_df = df_resultados.nlargest(top_n, 'r2').copy()
    
    top_df['wavelengths_str'] = top_df['wavelengths'].apply(
        lambda x: ', '.join([f'{w:.0f}' for w in sorted(x) if w is not None])
    )
    
    tabela = top_df[['n_bandas', 'wavelengths_str', 'r2', 'rmse', 'mae', 'mse']].copy()
    tabela.columns = ['N_Bandas', 'Bandas (nm)', 'R²', 'RMSE', 'MAE', 'MSE']
    tabela = tabela.reset_index(drop=True)
    tabela.index = tabela.index + 1
    
    print(f"\n{'='*90}")
    print(f"TOP {top_n} MELHORES COMBINAÇÕES DE BANDAS")
    print(f"{'='*90}")
    print(tabela.to_string())
    print(f"{'='*90}\n")
    
    tabela.to_csv('top_combinacoes_metricas.csv', index=True, index_label='Rank')
    print("Tabela salva em: top_combinacoes_metricas.csv\n")
    
    return tabela

# ================= EXECUÇÃO PRINCIPAL =================

if __name__ == "__main__":
    print("=== OTIMIZAÇÃO DE BANDAS PARA ESTIMATIVA DE CLOROFILA COM PLSR ===\n")
    
    if not os.path.exists(ARQUIVO_DADOS):
        sys.exit(f"ERRO: Arquivo {ARQUIVO_DADOS} não encontrado.")
    
    df = pd.read_csv(ARQUIVO_DADOS, sep=';', decimal=',')
    
    df_cloro = df.dropna(subset=['Y_Clorofila']).copy()
    
    cols_totais = [c for c in df_cloro.columns if 'Band_' in c]
    cols_selecionadas, bandas_encontradas = selecionar_colunas_por_lista(cols_totais, BANDAS_ALVO)
    
    if len(cols_selecionadas) == 0:
        sys.exit("ERRO: Nenhuma das bandas solicitadas foi encontrada nas colunas do CSV.")

    print(f"Amostras disponíveis: {len(df_cloro)}")
    print(f"Bandas mapeadas do sensor: {len(cols_selecionadas)}")
    print(f"Bandas encontradas: {sorted([extrair_valor_onda(c) for c in cols_selecionadas])}\n")

    X = df_cloro[cols_selecionadas].apply(pd.to_numeric, errors='coerce').fillna(0)
    y = pd.to_numeric(df_cloro['Y_Clorofila'], errors='coerce')
    
    diagnosticar_dados(X, y, cols_selecionadas)

    print("Iniciando busca exaustiva por combinações ótimas de bandas...")
    print("Isso pode levar alguns minutos...\n")
    
    # Otimização com todas as métricas
    (df_resultados, melhor_combo, melhor_modelo, melhor_r2, melhor_rmse, 
     melhor_mae, melhor_mse, melhor_bandas, melhor_X_test) = otimizar_bandas_por_r2(
        X, y, cols_selecionadas,
        min_bandas=3,
        max_bandas=len(cols_selecionadas),
        test_size=0.30,
        n_lvs=3,
        verbose=True
    )

    print("Gerando visualizações...\n")
    
    # Gráficos
    plotar_evolucao_r2(df_resultados)
    plotar_metricas_comparativas(df_resultados)
    plotar_distribuicao_metricas(df_resultados)  # NOVA FUNÇÃO - Histogramas e Boxplots
    plotar_metricas_por_n_bandas_detalhado(df_resultados)  # NOVA FUNÇÃO - Violinplots
    
    melhor_idx = df_resultados['r2'].idxmax()
    
    if df_resultados['r2'].max() > -np.inf:
        plotar_scatter_melhor_modelo(df_resultados.loc[melhor_idx])
        plotar_top_combinacoes(df_resultados, top_n=15)
        
        # Tabela de resultados
        gerar_tabela_top_resultados(df_resultados, top_n=20)
        
        print("\nAnalisando importância das bandas (VIP)...\n")
        melhor_bandas_lista = df_resultados.loc[melhor_idx, 'bandas']
        vip_df = plotar_vip_bandas(melhor_modelo, melhor_bandas_lista, melhor_X_test,
                                   titulo=f"VIP das Bandas - Melhor Modelo (R²={melhor_r2:.4f})")
        
        print("\nRanking VIP das Bandas:")
        print(vip_df.sort_values('VIP', ascending=False).to_string(index=False))
        
        # --- NOVO BLOCO: COEFICIENTES DE REGRESSÃO ---
        print("\nAnalisando Coeficientes de Regressão (Betas)...\n")
        coef_df = plotar_coeficientes_regressao(melhor_modelo, melhor_bandas_lista,
                                              titulo=f"Coeficientes de Regressão - Melhor Modelo (R²={melhor_r2:.4f})")
        
        print("\nTabela de Coeficientes:")
        print(coef_df.sort_values('Coeficiente', ascending=False).to_string(index=False))
        
        # Salvar tabela de coeficientes
        coef_df.to_csv('coeficientes_regressao.csv', index=False)
        print("\nTabela salva em: coeficientes_regressao.csv")
        # ---------------------------------------------
    else:
        print("⚠ AVISO: Nenhuma combinação produziu um modelo válido.")
    
    print(f"\n{'='*70}")
    print(" RELATÓRIO FINAL DE OTIMIZAÇÃO")
    print(f"{'='*70}")
    print(f"\nMELHOR CONFIGURAÇÃO ENCONTRADA:")
    print(f"  • Número de bandas: {len(melhor_bandas)}")
    print(f"  • Bandas selecionadas: {[extrair_valor_onda(b) for b in melhor_bandas]}")
    print(f"\nMÉTRICAS DE DESEMPENHO (Conjunto de Teste):")
    print(f"  • R² (Coeficiente de Determinação): {melhor_r2:.4f}")
    print(f"  • RMSE (Root Mean Squared Error):   {melhor_rmse:.4f}")
    print(f"  • MAE (Mean Absolute Error):        {melhor_mae:.4f}")
    print(f"  • MSE (Mean Squared Error):         {melhor_mse:.4f}")
    
    print(f"\nESTATÍSTICAS GERAIS:")
    print(f"  • Total de combinações testadas: {len(df_resultados)}")
    print(f"  • R² máximo encontrado: {df_resultados['r2'].max():.4f}")
    print(f"  • R² mínimo encontrado: {df_resultados['r2'].min():.4f}")
    print(f"  • R² médio: {df_resultados['r2'].mean():.4f}")
    print(f"  • RMSE mínimo encontrado: {df_resultados['rmse'].min():.4f}")
    print(f"  • MAE mínimo encontrado: {df_resultados['mae'].min():.4f}")
    
    print(f"\n{'='*70}\n")
    print("Processo concluído com sucesso!")
    print("\nArquivos gerados:")
    print("  • vip_bandas.png")
    print("  • evolucao_r2_por_bandas.png")
    print("  • metricas_comparativas.png")
    print("  • distribuicao_metricas.png")
    print("  • metricas_por_n_bandas_violinplot.png")
    print("  • scatter_melhor_modelo.png")
    print("  • top_combinacoes_todas_metricas.png")
    print("  • top_combinacoes_metricas.csv")
    print("  • coeficientes_regressao.png")
    print("  • coeficientes_regressao.csv")