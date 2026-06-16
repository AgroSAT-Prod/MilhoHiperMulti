import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Importações para Machine Learning e SPA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Importações para lidar com desbalanceamento
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# ================= CONFIGURAÇÕES =================
# Obtém o diretório onde o script está localizado
PASTA_DO_SCRIPT = os.path.dirname(os.path.abspath(__file__))
# Define o caminho completo do arquivo CSV contendo o dataset processado
ARQUIVO_DADOS = os.path.join(PASTA_DO_SCRIPT, 'DATASET_IA_PROCESSADO.csv')

# Seed para reprodutibilidade dos resultados aleatórios
SEED = 42

# 1. Definição das bandas ruidosas (wavelengths que devem ser removidas do dataset)
BANDAS_PARA_REMOVER = [558.50] 
# Tolerância para considerar uma banda como "ruim" (em nanômetros)
TOLERANCIA_REMOCAO = 1.0 

# 2. Configurações do algoritmo SPA (Successive Projections Algorithm)
# Número máximo de variáveis que o SPA pode selecionar
SPA_MAX_VARIAVEIS = 12   
# Proporção de dados para validação durante a seleção de features
SPA_TEST_SIZE = 0.3      

# 3. Intervalos de interesse em nanômetros (regiões espectrais relevantes)
INTERVALOS_DE_INTERESSE = [
    (520, 580),   # Pico do Verde (Green Peak)
    (690, 760)    # Red Edge (região sensível ao teor de clorofila)
] 

# ================= CLASSE SPA (ALGORITMO DE SELEÇÃO) =================
class SPA_Selector:
    """
    Implementação do algoritmo SPA (Successive Projections Algorithm) para seleção
    automática de variáveis espectrais relevantes para predição.
    O algoritmo funciona em duas fases: projeções sucessivas e avaliação via regressão linear.
    """
    def __init__(self, max_vars=20, test_size=0.3):
        """
        Inicializa o seletor SPA.
        
        Args:
            max_vars (int): Número máximo de bandas a selecionar
            test_size (float): Proporção de dados para validação (0 a 1)
        """
        self.max_vars = max_vars
        self.test_size = test_size
        self.selected_cols = []
        
    def fit(self, X, y):
        """
        Executa o algoritmo SPA para encontrar as melhores bandas espectrais.
        
        Args:
            X (DataFrame): Matriz de features (bandas espectrais)
            y (Series): Vetor alvo (variável a predizer)
            
        Returns:
            list: Lista dos nomes das colunas selecionadas
        """
        print(f" > [SPA] Iniciando seleção (Buscando as {self.max_vars} bandas mais distintas)...")
        
        # Divide dados em calibração (70%) e validação (30%)
        X_cal, X_val, y_cal, y_val = train_test_split(X, y, test_size=self.test_size, random_state=SEED)
        
        # Converte DataFrames para arrays NumPy para operações matriciais
        X_cal_mat = np.array(X_cal)
        X_val_mat = np.array(X_val)
        y_cal_mat = np.array(y_cal)
        y_val_mat = np.array(y_val)
        
        # Obtém informações sobre dimensões
        n_cols = X_cal_mat.shape[1]
        feature_names = X.columns.tolist()
        # Limita k_max ao máximo de variáveis ou número de colunas
        k_max = min(self.max_vars, n_cols - 1)
        
        # Variáveis para rastrear a melhor solução encontrada
        best_chain = None
        min_rmse_global = float('inf')
        
        # --- FASE 1: PROJEÇÕES SUCESSIVAS ---
        # Este dicionário armazena cadeias de índices para cada ponto de partida
        chains = {} 
        
        # Para cada coluna como ponto inicial
        for k0 in range(n_cols):
            # Cria cópia da matriz para projeções
            x_projected = X_cal_mat.copy()
            # Rastreia os índices das bandas selecionadas
            selected_indices = [k0]
            
            # Itera para construir uma cadeia de variáveis
            for k in range(1, k_max):
                # Obtém o vetor da última variável selecionada
                last_idx = selected_indices[-1]
                v_last = x_projected[:, last_idx].reshape(-1, 1)
                
                # Calcula a norma quadrada do vetor
                norm_sq = np.dot(v_last.T, v_last)
                # Se a norma é muito pequena, interrompe (evita divisão por zero)
                if norm_sq < 1e-10: break 
                
                # Projeta todos os outros vetores sobre o vetor atual
                proj_factor = np.dot(v_last.T, x_projected) / norm_sq
                # Remove a componente projetada (deflação)
                x_projected = x_projected - np.dot(v_last, proj_factor)
                
                # Calcula normas dos vetores residuais
                norms = np.sum(x_projected**2, axis=0)
                # Marca variáveis já selecionadas com valor negativo (não podem ser selecionadas novamente)
                norms[selected_indices] = -1 
                # Seleciona a variável com maior norma residual
                next_idx = np.argmax(norms)
                selected_indices.append(next_idx)
            
            # Armazena a cadeia de projeções para este ponto inicial
            chains[k0] = selected_indices

        # --- FASE 2: AVALIAÇÃO VIA REGRESSÃO LINEAR MÚLTIPLA (MLR) ---
        # Instancia o modelo de regressão linear
        lr = LinearRegression()
        
        # Para cada cadeia de variáveis gerada
        for k0, indices_chain in chains.items():
            # Testa subconjuntos de diferentes tamanhos da cadeia
            for n_vars in range(1, k_max + 1):
                # Obtém as primeiras n_vars variáveis da cadeia
                subset = indices_chain[:n_vars]
                # Treina regressão linear com esse subconjunto
                lr.fit(X_cal_mat[:, subset], y_cal_mat)
                # Realiza predição nos dados de validação
                y_pred = lr.predict(X_val_mat[:, subset])
                # Calcula RMSE (raiz do erro quadrático médio)
                rmse = np.sqrt(np.mean((y_val_mat - y_pred)**2))
                
                # Se este é o melhor resultado até agora, atualiza best_chain
                if rmse < min_rmse_global:
                    min_rmse_global = rmse
                    best_chain = subset
        
        # Converte índices para nomes de colunas
        self.selected_cols = [feature_names[i] for i in best_chain]
        return self.selected_cols

# ================= FUNÇÕES AUXILIARES =================

def extrair_valor_onda(nome_coluna):
    """
    Extrai o valor numérico do comprimento de onda (em nm) do nome da coluna.
    
    Args:
        nome_coluna (str): Nome da coluna no formato 'd1_Band_XXX.XXnm'
        
    Returns:
        float: Valor do comprimento de onda, ou None se não conseguir extrair
    """
    try:
        # Remove prefixos e sufixos do nome da coluna
        limpo = nome_coluna.replace('d1_', '').replace('Band_', '').replace('nm', '')
        # Converte para float
        return float(limpo)
    except:
        # Retorna None se não conseguir fazer a conversão
        return None

def filtrar_e_limpar_bandas(colunas, intervalos, bandas_ruins):
    """
    Filtra as bandas espectrais, mantendo apenas aquelas dentro dos intervalos
    de interesse e removendo as bandas ruidosas definidas.
    
    Args:
        colunas (list): Lista de nomes de colunas
        intervalos (list): Lista de tuplas (início, fim) em nm
        bandas_ruins (list): Lista de wavelengths a remover
        
    Returns:
        list: Colunas filtradas e limpas
    """
    cols_boas = []
    print(f"\nFiltrando bandas e removendo ruídos...")
    # Itera sobre cada coluna
    for col in colunas:
        # Extrai o valor numérico do comprimento de onda
        wl = extrair_valor_onda(col)
        if wl is not None:
            # Verifica se esta banda é considerada "ruim" (ruidosa)
            e_ruim = any(abs(wl - ruim) < TOLERANCIA_REMOCAO for ruim in bandas_ruins)
            # Se é ruim, pula para próxima iteração
            if e_ruim: continue
            # Verifica se a banda está dentro de algum intervalo de interesse
            for (inicio, fim) in intervalos:
                if inicio <= wl <= fim:
                    cols_boas.append(col)
                    break
    return cols_boas

def plotar_espectro_selecionado(dataframe, todas_cols, cols_spa, coluna_alvo, titulo):
    """
    Plota o espectro médio para cada classe, destacando as bandas selecionadas pelo SPA.
    
    Args:
        dataframe (DataFrame): Dataset completo
        todas_cols (list): Todas as colunas de banda
        cols_spa (list): Colunas selecionadas pelo SPA
        coluna_alvo (str): Nome da coluna com a classe/dose
        titulo (str): Título do gráfico
    """
    # Ordena todas as colunas por comprimento de onda
    todas_ordenadas = sorted(todas_cols, key=lambda x: extrair_valor_onda(x))
    # Extrai valores numéricos de comprimento de onda
    wls_full = [extrair_valor_onda(c) for c in todas_ordenadas]
    wls_spa = [extrair_valor_onda(c) for c in cols_spa]
    
    # Calcula a média espectral para cada classe/dose
    grupos = dataframe.groupby(coluna_alvo)[todas_ordenadas].mean()
    
    # Cria figura e eixos
    plt.figure(figsize=(12, 6))
    # Define cores diferentes para cada classe
    cores = plt.cm.viridis(np.linspace(0, 1, len(grupos)))

    # Plota o espectro médio para cada classe
    for i, (classe, linha_media) in enumerate(grupos.iterrows()):
        plt.plot(wls_full, linha_media, label=f"{classe}", color=cores[i], linewidth=2)

    # Adiciona linhas verticais para as bandas selecionadas pelo SPA
    for j, wl in enumerate(wls_spa):
        # Adiciona label apenas uma vez para evitar duplicação na legenda
        lbl = 'Bandas SPA' if j == 0 else ""
        plt.axvline(x=wl, color='red', linestyle='--', alpha=0.7, label=lbl)

    # Configurações do gráfico
    plt.title(f"Bandas Selecionadas SPA - {titulo}", fontsize=14)
    plt.xlabel("Comprimento de Onda (nm)")
    plt.ylabel("1ª Derivada")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def validar_bandas_biologicamente(df, cols_selecionadas, target_col='Dose_N'):
    """
    Gera visualizações para validar biologicamente as bandas selecionadas.
    Inclui: análise de correlação de Pearson e boxplots para visualizar a separação das classes.
    
    Args:
        df (DataFrame): Dataset completo
        cols_selecionadas (list): Bandas selecionadas pelo SPA
        target_col (str): Nome da coluna alvo
    """
    print("\n" + "="*40)
    print(" 3. VALIDAÇÃO BIOLÓGICA DAS BANDAS")
    print("="*40)

    # 1. PREPARAÇÃO DOS DADOS
    # Seleciona apenas colunas com derivada de banda
    cols_derivada = [c for c in df.columns if c.startswith('d1_Band_')]
    
    # Extrai o valor numérico da Dose de forma robusta
    y_corr = pd.to_numeric(df[target_col], errors='coerce').fillna(0)
    
    # 2. CÁLCULO DA CORRELAÇÃO DE PEARSON (para todo o espectro)
    correlacoes = []
    wls = []
    
    print("Calculando correlação espectral...")
    # Para cada banda no espectro completo
    for col in cols_derivada:
        wl = extrair_valor_onda(col)
        if wl is not None:
            # Calcula correlação de Pearson entre a banda e a dose de N
            corr = df[col].corr(y_corr)
            correlacoes.append(corr)
            wls.append(wl)
            
    # Ordena por comprimento de onda para plotagem correta
    ordem = np.argsort(wls)
    wls = np.array(wls)[ordem]
    correlacoes = np.array(correlacoes)[ordem]

    # 3. PLOT DO ESPECTRO DE CORRELAÇÃO
    plt.figure(figsize=(12, 6))
    
    # Plota a linha contínua de correlação
    plt.plot(wls, np.abs(correlacoes), label='Correlação com N (|r|)', color='gray', alpha=0.6)
    # Preenchimento para melhor visualização
    plt.fill_between(wls, 0, np.abs(correlacoes), color='gray', alpha=0.1)
    
    # Destaca as bandas selecionadas pelo SPA em vermelho
    wls_spa = [extrair_valor_onda(c) for c in cols_selecionadas]
    
    # Para cada banda selecionada
    for i, wl in enumerate(wls_spa):
        # Encontra o índice mais próximo no espectro completo
        idx_prox = (np.abs(wls - wl)).argmin()
        # Obtém o valor de correlação naquela banda
        r_val = np.abs(correlacoes[idx_prox])
        
        # Marca a banda selecionada com um ponto vermelho
        plt.scatter(wl, r_val, color='red', s=100, zorder=5)
        # Adiciona anotação com valor de comprimento de onda e correlação
        plt.text(wl, r_val + 0.05, f"{wl}nm\n(r={r_val:.2f})", 
                 ha='center', fontsize=9, fontweight='bold', color='darkred')
        # Adiciona linha vertical auxiliar
        plt.axvline(x=wl, color='red', linestyle='--', alpha=0.3)

    # Configurações do gráfico
    plt.title("Validação: As bandas escolhidas coincidem com os picos de informação?", fontsize=14)
    plt.xlabel("Comprimento de Onda (nm)")
    plt.ylabel("Correlação Absoluta de Pearson (|r|)")
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 4. BOXPLOTS: O COMPORTAMENTO É LÓGICO?
    # Cria figura com múltiplos subplots (até 3 bandas)
    plt.figure(figsize=(14, 5))
    
    # Ordena o dataset pela dose para melhor visualização
    df_sorted = df.sort_values(by=target_col, key=lambda col: pd.to_numeric(col, errors='coerce'))
    
    # Plota boxplot para as 3 primeiras bandas selecionadas
    for i, col in enumerate(cols_selecionadas):
        if i >= 3: break # Limita a 3 para evitar poluição visual
        
        # Cria subplot
        plt.subplot(1, 3, i+1)
        # Plota boxplot com separação por dose
        sns.boxplot(x=target_col, y=col, data=df_sorted, palette="viridis")
        # Extrai comprimento de onda para título
        wl = extrair_valor_onda(col)
        plt.title(f"Banda: {wl} nm")
        plt.ylabel("Valor da 1ª Derivada")
        plt.xlabel("Dose de N")
        
    # Título geral
    plt.suptitle("Separação Visual das Classes nas Bandas Selecionadas", fontsize=16)
    plt.tight_layout()
    plt.show()

def avaliar_modelo(y_true, y_pred, titulo, labels_nomes=None):
    """
    Avalia o desempenho do modelo de classificação exibindo relatório
    de classificação e matriz de confusão.
    
    Args:
        y_true (array): Valores verdadeiros
        y_pred (array): Valores preditos
        titulo (str): Título para os resultados
        labels_nomes (list): Nomes das classes para melhor legibilidade
    """
    # Imprime relatório detalhado de classificação
    print(f"\n[{titulo}] Relatório de Classificação:")
    print(classification_report(y_true, y_pred, target_names=labels_nomes, zero_division=0))
    
    # Calcula matriz de confusão
    cm = confusion_matrix(y_true, y_pred)
    # Cria visualização da matriz de confusão
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels_nomes if labels_nomes else "auto",
                yticklabels=labels_nomes if labels_nomes else "auto")
    plt.title(f"Matriz de Confusão: {titulo}")
    plt.xlabel("Predito")
    plt.ylabel("Verdadeiro")
    plt.tight_layout()
    plt.show()

def plotar_importancia(modelo, colunas_X, titulo):
    """
    Plota a importância de cada feature no modelo Random Forest.
    
    Args:
        modelo (RandomForestClassifier ou Pipeline): Modelo treinado
        colunas_X (list): Nomes das colunas de features
        titulo (str): Título do gráfico
    """
    # Se o modelo é um Pipeline, extrai o Random Forest
    if isinstance(modelo, Pipeline):
        rf_model = modelo.named_steps['rf']
    else:
        rf_model = modelo
        
    # Obtém as importâncias de cada feature
    importancias = rf_model.feature_importances_
    # Ordena em ordem decrescente
    indices = np.argsort(importancias)[::-1]
    
    # Cria gráfico de barras
    plt.figure(figsize=(10, 5))
    plt.title(f"Feature Importance Random Forest - {titulo}")
    plt.bar(range(len(colunas_X)), importancias[indices], align="center", color='#2ca02c')
    # Define rótulos das colunas no eixo X com rotação para legibilidade
    plt.xticks(range(len(colunas_X)), [colunas_X[i] for i in indices], rotation=45)
    plt.tight_layout()
    plt.show()

# ================= EXECUÇÃO PRINCIPAL =================

if __name__ == "__main__":
    # Verifica se o arquivo de dados existe
    if not os.path.exists(ARQUIVO_DADOS):
        sys.exit("ERRO: Arquivo PROCESSADO não encontrado.")
        
    # Carrega o dataset com separador ';' e decimal ','
    df = pd.read_csv(ARQUIVO_DADOS, sep=';', decimal=',')
    print(f"Dataset carregado: {len(df)} amostras.")

    # 1. SELEÇÃO E LIMPEZA DE COLUNAS
    # Filtra apenas colunas de primeira derivada
    cols_derivada = [c for c in df.columns if c.startswith('d1_Band_')]
    # Remove bandas ruidosas e mantém apenas intervalos de interesse
    cols_limpas = filtrar_e_limpar_bandas(cols_derivada, INTERVALOS_DE_INTERESSE, BANDAS_PARA_REMOVER)
    
    # Verifica se pelo menos uma banda restou
    if len(cols_limpas) == 0: sys.exit("ERRO: Nenhuma banda restou.")

    # Prepara dados para o SPA: converte para numérico e preenche valores faltantes
    X_full = df[cols_limpas].apply(pd.to_numeric, errors='coerce').fillna(0)
    # Extrai valores numéricos da dose de N
    y_target_numeric = pd.to_numeric(df['Dose_N'], errors='coerce').fillna(0)

    # 2. EXECUTA ALGORITMO SPA
    print("\n" + "="*40)
    print(" 1. SELEÇÃO DE BANDAS ÓTIMAS (SPA)")
    print("="*40)
    
    # Instancia o seletor SPA com parâmetros definidos
    spa = SPA_Selector(max_vars=SPA_MAX_VARIAVEIS, test_size=SPA_TEST_SIZE)
    # Executa a seleção e obtém as colunas finais
    cols_finais = spa.fit(X_full, y_target_numeric)
    
    # Extrai e ordena os valores de comprimento de onda das bandas selecionadas
    wls_finais = sorted([extrair_valor_onda(c) for c in cols_finais])
    print(f"\nRESULTADO FINAL SPA: {len(cols_finais)} bandas selecionadas -> {wls_finais}")

    # 3. VALIDAÇÃO BIOLÓGICA (CORRELAÇÃO DE PEARSON + BOXPLOTS)
    # Valida se as bandas escolhidas têm sentido biológico
    validar_bandas_biologicamente(df, cols_finais, 'Dose_N')

    # Seleciona apenas as colunas com as bandas selecionadas pelo SPA
    X_selected = X_full[cols_finais]
    # Instancia o método Leave-One-Out Cross-Validation
    loo = LeaveOneOut()

    # 4. TREINAMENTO E CLASSIFICAÇÃO
    print("\n" + "="*40)
    print(" 4. CLASSIFICAÇÃO (SMOTE + Class Weights)")
    print("="*40)

    # --- Análise MULTI-CLASSE por DOSE ---
    # Converte as doses para string para tratamento como classe
    y_dose = df['Dose_N'].astype(str)
    # Ordena as classes de forma lógica (como números)
    classes_dose = sorted(y_dose.unique(), key=lambda x: float(x))
    
    print(f"\n> Classificando por DOSE ({len(classes_dose)} classes)...")
    # Plota o espectro separado por dose
    plotar_espectro_selecionado(df, cols_limpas, cols_finais, 'Dose_N', "Doses")
    
    # Cria pipeline com SMOTE para balanceamento e Random Forest para classificação
    # SMOTE gera amostras sintéticas das classes minoritárias
    # k_neighbors=1 e n_jobs=1 evitam erros de threading/memória
    pipeline_dose = Pipeline([
        ('smote', SMOTE(random_state=SEED, k_neighbors=1)), 
        ('rf', RandomForestClassifier(n_estimators=100, 
                                      class_weight='balanced', 
                                      random_state=SEED, 
                                      n_jobs=1)) # n_jobs=1 para evitar paralelização
    ])
    
    # Realiza validação cruzada Leave-One-Out
    y_pred_dose = cross_val_predict(pipeline_dose, X_selected, y_dose, cv=loo, n_jobs=1)
    
    # Avalia o modelo e exibe métricas
    avaliar_modelo(y_dose, y_pred_dose, "Classificação Doses", labels_nomes=classes_dose)
    
    # Treina o pipeline com todos os dados para calcular feature importance
    pipeline_dose.fit(X_selected, y_dose)
    # Plota a importância de cada feature
    plotar_importancia(pipeline_dose, cols_finais, "Doses")

    # --- Análise BINÁRIA (Com N vs Sem N) ---
    print(f"\n> Classificando BINÁRIO (Com N vs Sem N)...")
    # Cria variável binária: valores > 0 = "Com N", senão = "Sem N"
    y_binario = np.where(y_target_numeric > 0, "Com N", "Sem N")
    
    # Cria pipeline similar para classificação binária
    pipeline_bin = Pipeline([
        ('smote', SMOTE(random_state=SEED, k_neighbors=3)), 
        ('rf', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=SEED, n_jobs=1))
    ])
    
    # Realiza validação cruzada Leave-One-Out para classificação binária
    y_pred_bin = cross_val_predict(pipeline_bin, X_selected, y_binario, cv=loo, n_jobs=1)
    # Avalia o modelo binário
    avaliar_modelo(y_binario, y_pred_bin, "Binário")
    
    # Mensagem de conclusão
    print("\nProcesso concluído.")