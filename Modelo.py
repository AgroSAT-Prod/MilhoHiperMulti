import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ================= CONFIGURAÇÕES =================
PASTA_DO_SCRIPT = os.path.dirname(os.path.abspath(__file__))
ARQUIVO_DADOS = os.path.join(PASTA_DO_SCRIPT, 'DATASET_IA_PROCESSADO.csv')

# Configuração do Modelo
SEED = 42  # Para garantir que os resultados sejam reproduzíveis

# ================= FUNÇÕES DE AVALIAÇÃO =================

def avaliar_modelo(y_teste, y_pred, titulo, labels_nomes=None):
    """Gera matriz de confusão e relatório de métricas."""
    print(f"\n{'='*20} {titulo} {'='*20}")
    
    # Acurácia
    acc = accuracy_score(y_teste, y_pred)
    print(f"Acurácia Global: {acc:.2%}")
    
    # Relatório Detalhado
    print("\nRelatório de Classificação:")
    print(classification_report(y_teste, y_pred, target_names=labels_nomes))
    
    # Matriz de Confusão Visual
    cm = confusion_matrix(y_teste, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels_nomes if labels_nomes else "auto",
                yticklabels=labels_nomes if labels_nomes else "auto")
    plt.title(f"Matriz de Confusão: {titulo}")
    plt.xlabel("Predição da IA")
    plt.ylabel("Real (Ground Truth)")
    plt.tight_layout()
    plt.show()

def plotar_importancia(modelo, colunas_X, titulo):
    """Mostra quais bandas foram mais importantes para a decisão."""
    importancias = modelo.feature_importances_
    indices = np.argsort(importancias)[::-1] # Ordena do maior para o menor
    
    # Pega as TOP 15 bandas mais importantes
    top_n = 15
    indices_top = indices[:top_n]
    
    plt.figure(figsize=(10, 6))
    plt.title(f"Top {top_n} Bandas Mais Importantes - {titulo}")
    plt.bar(range(top_n), importancias[indices_top], align="center", color='#2ca02c')
    plt.xticks(range(top_n), [colunas_X[i].replace('d1_Band_', '').replace('nm', '') + 'nm' for i in indices_top], rotation=45)
    plt.xlabel("Comprimento de Onda")
    plt.ylabel("Importância (Gini)")
    plt.tight_layout()
    plt.show()

def plotar_assinatura_media_por_classe(dataframe, colunas_bandas, coluna_alvo, titulo):
    """
    Plota a média do espectro para cada classe (Dose ou Binário).
    Isso ajuda a explicar POR QUE o modelo escolheu certas bandas.
    """
    plt.figure(figsize=(12, 6))
    
    # Extrai números das bandas para o eixo X
    wls = [float(c.replace('d1_Band_', '').replace('Band_', '').replace('nm', '')) for c in colunas_bandas]
    
    # Agrupa por classe (ex: Doses 0, 90, 180...) e calcula média
    grupos = dataframe.groupby(coluna_alvo)[colunas_bandas].mean()
    
    # Define cores (se for binário usa vermelho/verde, se for doses usa gradiente)
    if len(grupos) == 2:
        cores = ['red', 'green'] # Assumindo 0 (Sem N) e 1 (Com N)
    else:
        cores = plt.cm.viridis(np.linspace(0, 1, len(grupos)))

    for i, (classe, linha_media) in enumerate(grupos.iterrows()):
        plt.plot(wls, linha_media, label=f"{coluna_alvo}: {classe}", color=cores[i], linewidth=2)

    plt.title(f"Assinatura Espectral Média por {titulo}", fontsize=14)
    plt.xlabel("Comprimento de Onda (nm)")
    plt.ylabel("Reflectância (ou 1ª Derivada)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Destaque para regiões típicas de Nitrogênio (Red Edge)
    plt.axvspan(690, 740, color='gray', alpha=0.1, label='Região Red-Edge')
    
    plt.tight_layout()
    plt.show()

# --- ONDE COLOCAR NA EXECUÇÃO PRINCIPAL (EXEMPLO) ---
# Logo após rodar o modelo de Doses:
# 

# Logo após rodar o modelo Binário (precisa criar a coluna no df antes de plotar):
# 

# ================= EXECUÇÃO PRINCIPAL =================

if __name__ == "__main__":
    # 1. Carregar Dados
    if not os.path.exists(ARQUIVO_DADOS):
        print("ERRO: Arquivo processado não encontrado. Rode o script de processamento antes.")
        sys.exit()
        
    df = pd.read_csv(ARQUIVO_DADOS, sep=';', decimal=',')
    print(f"Dataset carregado: {len(df)} amostras.")

    # 2. Preparar Features (X)
    # Seleciona apenas as colunas que são bandas espectrais (já processadas)
    cols_features = [c for c in df.columns if c.startswith('d1_Band_') or c.startswith('Band_')]
    X = df[cols_features]
    
    # Garante que X contém apenas números
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    print(f"Usando {len(cols_features)} bandas espectrais como entrada.")

    # ---------------------------------------------------------
    # MODELO 1: Classificação por DOSE (Multiclasse)
    # ---------------------------------------------------------
    # Alvo (Y): Dose de Nitrogênio (0, 90, 180, etc.)
    y_dose = df['Dose_N'].astype(str) # Converte para texto para ser categoria

    print("\n=== VERIFICAÇÃO DE CLASSES ===")
    doses_unicas = sorted(df['Dose_N'].unique())
    print(f"Doses encontradas no CSV: {doses_unicas}")
    print(f"Quantidade de Classes: {len(doses_unicas)}")
    
    # Conta quantas amostras tem em cada dose
    print("Contagem por Dose:")
    print(df['Dose_N'].value_counts())
    
    # Se tiver menos de 2 amostras em alguma dose, o stratify vai dar erro ou o split vai falhar
    minimo_amostras = df['Dose_N'].value_counts().min()
    if minimo_amostras < 2:
        print("\nALERTA CRÍTICO: Você tem doses com apenas 1 amostra!")
        print("Isso impede a divisão correta entre Treino e Teste.")
    
    # Divisão Treino (70%) / Teste (30%)
    X_train, X_test, y_train, y_test = train_test_split(X, y_dose, test_size=0.3, random_state=SEED, stratify=y_dose)
    
    # Treinamento
    print("\nTreinando Modelo 1 (Doses Específicas)...")
    rf_dose = RandomForestClassifier(n_estimators=100, random_state=SEED)
    rf_dose.fit(X_train, y_train)
    
    # Avaliação
    y_pred_dose = rf_dose.predict(X_test)
    
    # Pega os nomes das classes ordenados para o gráfico
    classes_dose = sorted(rf_dose.classes_, key=lambda x: float(x))
    avaliar_modelo(y_test, y_pred_dose, "Modelo Multiclasse (Doses N)", labels_nomes=classes_dose)
    plotar_importancia(rf_dose, cols_features, "Doses N")
    plotar_assinatura_media_por_classe(df, cols_features, 'Dose_N', "Doses de Nitrogênio")

    # ---------------------------------------------------------
    # MODELO 2: Classificação BINÁRIA (Com N vs Sem N)
    # ---------------------------------------------------------
    print("\nPreparando Modelo 2 (Binário)...")
    
    # Cria a coluna Binária: Se Dose > 0 então "Com N", senão "Sem N"
    # Precisamos converter 'Dose_N' para float primeiro para garantir
    doses_numericas = pd.to_numeric(df['Dose_N'], errors='coerce').fillna(0)
    y_binario = np.where(doses_numericas > 0, "Com Nitrogenio", "Sem Nitrogenio")
    
    # Divisão Treino/Teste
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X, y_binario, test_size=0.3, random_state=SEED, stratify=y_binario)
    
    # Treinamento
    rf_bin = RandomForestClassifier(n_estimators=100, random_state=SEED)
    rf_bin.fit(X_train_b, y_train_b)
    
    # Avaliação
    y_pred_bin = rf_bin.predict(X_test_b)
    avaliar_modelo(y_test_b, y_pred_bin, "Modelo Binário (Com vs Sem N)")
    plotar_importancia(rf_bin, cols_features, "Binário")
    df['Status_N'] = np.where(pd.to_numeric(df['Dose_N'], errors='coerce').fillna(0) > 0, "Com N", "Sem N")
    plotar_assinatura_media_por_classe(df, cols_features, 'Status_N', "Status Nitrogênio")
    
    print("\nProcesso concluído! Os modelos foram avaliados.")