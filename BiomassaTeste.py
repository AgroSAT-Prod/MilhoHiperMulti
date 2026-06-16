import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import warnings
import optuna  # <--- IMPORTANTE: Importando Optuna

# Importações de Machine Learning
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score

# Ignorar avisos
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING) # Limpa o log do console, mostrando apenas erros ou warnings

# ================= CONFIGURAÇÕES =================
ARQUIVO_DADOS = r"C:\Users\muril\OneDrive\Documentos\Experiments-Prod\ProjetoMilho\VontadeDeMorrer.csv" 

INDICES_INTERESSE = [
    'NDRE', 'RDVI'
]

# ================= FUNÇÕES AUXILIARES =================

def treinar_otimizado(X, y_transformado):
    print(f" -> Iniciando Otimização Bayesiana com Optuna...")
    print(f" -> Buscando melhores hiperparâmetros (N_trees, Depth, Split, Leaf)...")

    loo = LeaveOneOut()

    # Função Objetivo para o Optuna minimizar
    def objective(trial):
        # Definição do espaço de busca dos hiperparâmetros
        n_estimators = trial.suggest_int('n_estimators', 50, 1000, step=50)
        
        # Max depth: None ou inteiro. Tratamos 0 como None (profundidade ilimitada) ou buscamos inteiros.
        # Aqui vamos buscar inteiros entre 5 e 50. Árvores muito profundas causam overfitting.
        max_depth = trial.suggest_int('max_depth', 3, 50)
        
        # Adicionando parâmetros extras para evitar overfitting
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)

        model = RandomForestRegressor(
            n_estimators=n_estimators, 
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42, 
            n_jobs=-1
        )
        
        # Cross-validation com LeaveOneOut
        # Retorna as predições para cada amostra quando ela estava no teste
        y_cv_pred = cross_val_predict(model, X, y_transformado, cv=loo, n_jobs=-1)
        
        # Métrica a ser minimizada (RMSE dos dados transformados)
        rmse_trans = np.sqrt(mean_squared_error(y_transformado, y_cv_pred))
        
        return rmse_trans

    # Criar o estudo
    study = optuna.create_study(direction='minimize')
    
    # Executar a otimização (n_trials define quantas combinações testar)
    # Aumente n_trials se quiser uma busca mais exaustiva (ex: 50 ou 100)
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    # Recuperar os melhores parâmetros
    best_params = study.best_params
    best_rmse = study.best_value

    print(f"\n -> Melhores Parâmetros Encontrados: {best_params}")
    print(f" -> Melhor RMSE (Transf): {best_rmse:.4f}")

    # Recriar o melhor modelo para treinar e retornar as predições finais
    best_model = RandomForestRegressor(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf'],
        random_state=42,
        n_jobs=-1
    )

    # Gerar as predições CV do melhor modelo para plotagem e métricas
    best_y_pred_trans = cross_val_predict(best_model, X, y_transformado, cv=loo, n_jobs=-1)
    
    # Treinar o modelo final em todos os dados (para uso futuro/feature importance)
    best_model.fit(X, y_transformado)

    return best_model, best_y_pred_trans, best_params

# ================= EXECUÇÃO PRINCIPAL =================

if __name__ == "__main__":
    print("=== PREDIÇÃO DE BIOMASSA (RF + OPTUNA + LOG + NORMALIZAÇÃO) ===\n")

    # 1. Carregar CSV
    if not os.path.exists(ARQUIVO_DADOS):
        print(f"ERRO: Arquivo '{ARQUIVO_DADOS}' não encontrado.")
        sys.exit(1)

    try:
        df = pd.read_csv(ARQUIVO_DADOS, sep=';', decimal=',', thousands='.', na_values=['-', ' -', '- ', 'nan', 'NaN'])
        df.columns = df.columns.str.strip()
        print(f"✓ Arquivo carregado: {len(df)} linhas totais.")
    except Exception as e:
        sys.exit(f"Erro ao ler CSV: {e}")

    # 2. Localizar Coluna de Biomassa
    colunas_biomassa = [c for c in df.columns if 'kg/ha' in str(c)]
    
    if len(colunas_biomassa) == 0:
        print("⚠ ERRO: Não encontrei a coluna alvo (kg/ha). Verifique o CSV.")
        sys.exit(1)
        
    NOME_COLUNA_ALVO = colunas_biomassa[0]
    print(f"-> Coluna Alvo detectada: '{NOME_COLUNA_ALVO}'")

    # 3. Filtragem e Limpeza Inteligente
    
    # Passo A: Remover linhas onde a Biomassa é nula (NaN)
    df_temp = df.dropna(subset=[NOME_COLUNA_ALVO]).copy()
    
    # Força a conversão para numérico caso o Pandas tenha lido como texto
    if df_temp[NOME_COLUNA_ALVO].dtype == 'object':
        print("-> Convertendo coluna de Biomassa de Texto para Número...")
        df_temp[NOME_COLUNA_ALVO] = df_temp[NOME_COLUNA_ALVO].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
    
    # Converte para float (valores inválidos viram NaN)
    df_temp[NOME_COLUNA_ALVO] = pd.to_numeric(df_temp[NOME_COLUNA_ALVO], errors='coerce')
    
    # Remove NaNs gerados pela conversão
    df_temp = df_temp.dropna(subset=[NOME_COLUNA_ALVO])
    
    # Agora sim, filtra maior que 0
    df_temp = df_temp[df_temp[NOME_COLUNA_ALVO] > 0]
    
    print(f"-> Linhas com Biomassa válida: {len(df_temp)}")

    if len(df_temp) < 3:
        print("ERRO: Menos de 3 amostras possuem valor de biomassa. Verifique a coluna '(kg/ha)'.")
        sys.exit(1)

    # Passo B: Identificar colunas de interesse
    colunas_presentes = [col for col in INDICES_INTERESSE if col in df_temp.columns]
    
    if not colunas_presentes:
        print("ERRO: Nenhum índice espectral encontrado.")
        sys.exit(1)

    # Passo C: Selecionar apenas colunas que NÃO estão vazias PARA ESSAS LINHAS
    colunas_validas_para_modelo = []
    colunas_rejeitadas = []

    for col in colunas_presentes:
        # Garante que a coluna preditora também seja numérica
        df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce')
        validos = df_temp[col].notna().sum()
        
        if validos == 0:
            colunas_rejeitadas.append(col)
        else:
            colunas_validas_para_modelo.append(col)
            
    print(f"-> Variáveis REJEITADAS (vazias): {colunas_rejeitadas}")
    print(f"-> Variáveis ACEITAS: {colunas_validas_para_modelo}")

    if not colunas_validas_para_modelo:
        print("ERRO: Todas as variáveis preditoras estão vazias nas linhas onde existe biomassa.")
        sys.exit(1)

    # Passo D: Criar DataFrame Final
    df_clean = df_temp[colunas_validas_para_modelo + [NOME_COLUNA_ALVO]].dropna().reset_index(drop=True)

    print(f"-> Amostras Finais para Treinamento: {len(df_clean)}")
    
    if len(df_clean) < 3:
        print("ERRO: Poucas amostras restantes após limpeza final.")
        sys.exit(1)

    X = df_clean[colunas_validas_para_modelo]
    y_real = df_clean[NOME_COLUNA_ALVO].values
    
    # === [ETAPA 1] TRANSFORMAÇÃO LOGARÍTMICA ===
    print("-> Aplicando transformação Log Neperiano (Ln)...")
    y_log = np.log(y_real) 
    
    # === [ETAPA 2] NORMALIZAÇÃO (Z-score) ===
    print("-> Aplicando Normalização (Z-score) nos dados Log...")
    mean_log = np.mean(y_log)
    std_log = np.std(y_log)
    y_nl = (y_log - mean_log) / std_log

    # 4. Regressão Otimizada (Agora com OPTUNA)
    model, y_pred_nl, params = treinar_otimizado(X, y_nl)

    # === [ETAPA 3] INVERSÃO DAS TRANSFORMAÇÕES ===
    y_pred_log = (y_pred_nl * std_log) + mean_log
    y_pred_final = np.exp(y_pred_log)

    # 5. Cálculo de Métricas
    r2 = r2_score(y_real, y_pred_final)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred_final))
    media_obs = np.mean(y_real)
    rmse_perc = (rmse / media_obs) * 100

    # 6. Resultados Gerais
    print(f"\n{'='*60}")
    print(f"RESULTADO FINAL (Otimizado via Optuna)")
    print(f"Parâmetros: {params}")
    print(f"{'='*60}")
    print(f"R² (CV - Escala Original): {r2:.4f}")
    print(f"RMSE (Escala Original):     {rmse:.2f} kg/ha")
    print(f"RMSE% (Relativo):          {rmse_perc:.2f}%")
    print(f"{'='*60}\n")
    
    # GRÁFICOS
    if r2 > -10: 
        plt.figure(figsize=(14, 6))
        
        plt.subplot(1, 2, 1)
        plt.scatter(y_real, y_pred_final, color='royalblue', alpha=0.7, s=80, edgecolors='k')
        min_v, max_v = min(y_real.min(), y_pred_final.min()), max(y_real.max(), y_pred_final.max())
        plt.plot([min_v, max_v], [min_v, max_v], 'r--', linewidth=2, label='1:1')
        plt.xlabel('Observado (kg/ha)')
        plt.ylabel('Predito (kg/ha)')
        plt.title(f'Predição Optuna (N={params["n_estimators"]}, D={params["max_depth"]})\nR²={r2:.3f} | RMSE%={rmse_perc:.1f}%')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if model is not None:
            plt.subplot(1, 2, 2)
            importances = model.feature_importances_
            feat_imp_df = pd.DataFrame({'Índice': colunas_validas_para_modelo, 'Importancia': importances})
            feat_imp_df = feat_imp_df.sort_values('Importancia', ascending=True)
            plt.barh(feat_imp_df['Índice'], feat_imp_df['Importancia'], color='forestgreen')
            plt.xlabel('Importância Relativa (0 a 1)')
            plt.title('Importância das Variáveis')
        
        plt.tight_layout()

    # OUTLIERS
    residuos = y_real - y_pred_final
    media_res = np.mean(residuos)
    std_res = np.std(residuos)
    limiar_superior = media_res + 2 * std_res
    limiar_inferior = media_res - 2 * std_res
    outliers_mask = (residuos > limiar_superior) | (residuos < limiar_inferior)
    indices_outliers = np.where(outliers_mask)[0]
    
    if len(indices_outliers) > 0:
        print(f"\nOutliers Detectados (>2 sigma): {len(indices_outliers)}")
        print(f"{'Índice':<10} | {'Real':<10} | {'Predito':<10} | {'Resíduo':<10}")
        for i in indices_outliers:
            print(f"{i:<10} | {y_real[i]:.2f}       | {y_pred_final[i]:.2f}       | {residuos[i]:.2f}")

    plt.show()