import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import optuna  # <--- IMPORTANTE: Biblioteca de otimização

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, roc_auc_score,
                             roc_curve)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# Para silenciar logs excessivos do Optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ================= CONFIGURAÇÕES =================
ARQUIVO_DADOS = 'spectral_indices_rois.csv'
ARQUIVO_AGRONOMICO = os.path.join('Dataset', 'PlanilhaFiltrada.xlsx')
SEED = 42
N_TRIALS = 20  # Número de tentativas de otimização (aumente para 50 ou 100 para melhores resultados)

BANDAS_ALVO = [430, 450, 460, 500, 520, 550, 600, 625, 640, 660, 685, 720, 722, 740, 990, 995, 970]

# ================= FUNÇÕES AUXILIARES =================

def extrair_valor_onda(nome_coluna):
    """Extrai o valor numérico do comprimento de onda."""
    try:
        limpo = nome_coluna.replace('d1_', '').replace('Band_', '').replace('nm', '')
        return float(limpo)
    except:
        return None

def cruzar_com_excel(df_espectral, arquivo_excel):
    """Cruza dados espectrais com dados agronômicos."""
    try:
        df_agro = pd.read_excel(arquivo_excel, header=2)
        df_agro.columns = [str(c).strip().lower() for c in df_agro.columns]
    except Exception as e:
        print(f"⚠️  Erro ao ler Excel: {e}")
        df_espectral['Tratamento'] = 0
        return df_espectral
    
    col_id = next((c for c in df_agro.columns if 'id' in c and 'parcela' in c), 'id parcela')
    col_alvo = next((c for c in df_agro.columns if 'tratamento' in c), None)
    if col_alvo is None:
        col_alvo = next((c for c in df_agro.columns if 'dose' in c), None)
    
    col_clor_m = next((c for c in df_agro.columns if 'médio' in c and 'clorofila' in c), None)
    col_clor_s = next((c for c in df_agro.columns if 'superior' in c and 'clorofila' in c), None)
    
    df_espectral['ID_Numeric'] = df_espectral['ID_Amostra'].astype(int)
    
    cols_to_merge = [col_id]
    if col_alvo: cols_to_merge.append(col_alvo)
    if col_clor_m: cols_to_merge.append(col_clor_m)
    if col_clor_s: cols_to_merge.append(col_clor_s)

    df_merge = df_espectral.merge(df_agro[cols_to_merge], left_on='ID_Numeric', right_on=col_id, how='left')
    
    if col_clor_m and col_clor_s:
        df_merge['Y_Clorofila'] = np.where(df_merge['Parte'] == 'M', df_merge[col_clor_m], df_merge[col_clor_s])
    else:
        df_merge['Y_Clorofila'] = 0
    
    if col_alvo:
        df_merge.rename(columns={col_alvo: 'Tratamento'}, inplace=True)
    else:
        df_merge['Tratamento'] = 0
    
    cols_finais = list(df_espectral.columns) + ['Tratamento', 'Y_Clorofila']
    cols_existentes = [c for c in cols_finais if c in df_merge.columns]
    return df_merge[cols_existentes]

def selecionar_colunas_por_lista(todas_colunas, bandas_alvo, tolerancia=1.0):
    """Seleciona colunas correspondentes às bandas alvo."""
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

# ================= LÓGICA DO OPTUNA =================

def otimizar_hiperparametros(X_train, y_train, classes_unicas):
    """
    Executa o estudo do Optuna para encontrar os melhores hiperparâmetros
    usando Cross-Validation apenas nos dados de TREINO.
    """
    
    # Verifica o tamanho mínimo de classe para ajustar o limite do SMOTE
    # O k_neighbors do SMOTE não pode ser maior que o número de amostras da menor classe - 1
    contagem_classes = pd.Series(y_train).value_counts()
    min_samples = contagem_classes.min()
    max_k_smote = max(1, min_samples - 1)
    max_k_smote = min(max_k_smote, 7) # Teto de 7 vizinhos

    def objective(trial):
        # 1. Sugestão de Hiperparâmetros
        
        # Random Forest
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        max_depth = trial.suggest_int('max_depth', 5, 30)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
        max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])
        
        # SMOTE (k_neighbors)
        k_neighbors = trial.suggest_int('k_neighbors', 1, max_k_smote)

        # 2. Criação do Pipeline
        pipeline = Pipeline([
            ('smote', SMOTE(random_state=SEED, k_neighbors=k_neighbors)),
            ('rf', RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                max_features=max_features,
                class_weight='balanced',
                random_state=SEED,
                n_jobs=-1
            ))
        ])
        
        # 3. Validação Cruzada (Cross-Validation)
        # Usamos cv=3 para ser mais rápido na demonstração, mas cv=5 é ideal
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        
        # Métrica para otimizar: f1_weighted (bom para classes desbalanceadas)
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1_weighted', n_jobs=-1)
        
        return scores.mean()

    # Cria o estudo e executa
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=N_TRIALS)
    
    return study.best_params

# ================= FUNÇÕES DE CLASSIFICAÇÃO ATUALIZADAS =================

def classificar_multiclasse(X, y, cols_nomes, titulo="Classificação Multi-Classe"):
    
    # 1. Codificar labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # 2. Split treino/teste (O teste fica "escondido" da otimização)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.3, random_state=SEED, stratify=y_encoded
    )
    
    print(f"\n  📊 Treino: {len(X_train)} | Teste: {len(X_test)}")
    print(f"  Classes encontradas: {le.classes_}")
    print(f"  🔍 Iniciando otimização com Optuna ({N_TRIALS} trials)...")
    
    # 3. Buscar Melhores Hiperparâmetros
    best_params = otimizar_hiperparametros(X_train, y_train, np.unique(y_encoded))
    
    print(f"  ✓ Melhores parâmetros encontrados: {best_params}")
    
    # 4. Treinar Modelo Final com Melhores Parâmetros
    # Extrai parametros do SMOTE e do RF separadamente
    k_smote = best_params.pop('k_neighbors')
    
    pipeline = Pipeline([
        ('smote', SMOTE(random_state=SEED, k_neighbors=k_smote)),
        ('rf', RandomForestClassifier(
            **best_params, # Desempacota os params restantes (rf)
            class_weight='balanced',
            random_state=SEED,
            n_jobs=-1
        ))
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    # 5. Métricas e Visualização
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"\n  ✓ Acurácia: {acc:.4f}")
    print(f"  ✓ F1-Score: {f1:.4f}")

    print(f"\n  Relatório por Classe:")
    print(classification_report(y_test, y_pred, target_names=[str(c) for c in le.classes_], zero_division=0))
    
    # Matriz de Confusão (%)
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.1%', cmap='Blues', cbar_kws={'label': 'Porcentagem de Acerto'},
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    ax.set_title(f"Matriz de Confusão (%) - {titulo}\n(Otimizado)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Predito", fontsize=12)
    ax.set_ylabel("Verdadeiro", fontsize=12)
    plt.tight_layout()
    plt.savefig(f'matriz_confusao_{titulo.lower().replace(" ", "_")}.png', dpi=300)
    plt.show()
    
    # Importância das Features
    rf_model = pipeline.named_steps['rf']
    importancias = rf_model.feature_importances_
    indices = np.argsort(importancias)[::-1][:10]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    wls_top = [extrair_valor_onda(cols_nomes[i]) for i in indices]
    ax.barh(range(len(indices)), importancias[indices], color='steelblue', edgecolor='black')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([f'{wl:.0f}nm' if wl else cols_nomes[i] for i, wl in enumerate(wls_top)])
    ax.set_title(f"Top 10 Bandas - {titulo}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'importancia_top10_{titulo.lower().replace(" ", "_")}.png', dpi=300)
    plt.show()
    
    return {'acc': acc, 'f1': f1, 'precision': precision, 'recall': recall}

def classificar_binaria(X, y_numerico, cols_nomes, titulo="Classificação Binária"):
    
    # 1. Criar labels binários
    y_binary = np.where(y_numerico > 0, "Com N", "Sem N")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_binary)
    
    # 2. Split treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.3, random_state=SEED, stratify=y_encoded
    )
    
    print(f"\n  📊 Treino: {len(X_train)} | Teste: {len(X_test)}")
    print(f"  Classes: {le.classes_}")
    print(f"  🔍 Iniciando otimização com Optuna ({N_TRIALS} trials)...")
    
    # 3. Otimização
    best_params = otimizar_hiperparametros(X_train, y_train, np.unique(y_encoded))
    print(f"  ✓ Melhores parâmetros encontrados: {best_params}")
    
    k_smote = best_params.pop('k_neighbors')
    
    # 4. Pipeline Final
    pipeline = Pipeline([
        ('smote', SMOTE(random_state=SEED, k_neighbors=k_smote)),
        ('rf', RandomForestClassifier(
            **best_params,
            class_weight='balanced',
            random_state=SEED,
            n_jobs=-1
        ))
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.named_steps['rf'].predict_proba(X_test)
    
    # 5. Métricas
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    try:
        roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    except:
        roc_auc = 0.5
        
    print(f"\n  ✓ Acurácia: {acc:.4f}")
    print(f"  ✓ ROC-AUC: {roc_auc:.4f}")
    
    print(f"\n  Relatório por Classe:")
    print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))
    
    # Matriz de Confusão (%)
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.heatmap(cm, annot=True, fmt='.1%', cmap='Greens', cbar_kws={'label': 'Porcentagem de Acerto'},
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax1)
    ax1.set_title(f"Matriz de Confusão (%) - {titulo}", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Predito", fontsize=11)
    ax1.set_ylabel("Verdadeiro", fontsize=11)
    
    # ROC
    if len(np.unique(y_test)) > 1:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
        ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Caso Aleatório')
    ax2.set_title('Curva ROC', fontsize=12, fontweight='bold')
    ax2.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f'matriz_roc_{titulo.lower().replace(" ", "_")}.png', dpi=300)
    plt.show()

    # Importância das Features
    rf_model = pipeline.named_steps['rf']
    importancias = rf_model.feature_importances_
    indices = np.argsort(importancias)[::-1][:10]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    wls_top = [extrair_valor_onda(cols_nomes[i]) for i in indices]
    ax.barh(range(len(indices)), importancias[indices], color='forestgreen', edgecolor='black')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([f'{wl:.0f}nm' if wl else cols_nomes[i] for i, wl in enumerate(wls_top)])
    ax.set_title(f"Top 10 Bandas - {titulo}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'importancia_top10_{titulo.lower().replace(" ", "_")}.png', dpi=300)
    plt.show()

    return {'acc': acc, 'f1': f1, 'precision': precision, 'recall': recall, 'roc_auc': roc_auc}

def plotar_comparacao_metricas(resultados_multi, resultados_bin):
    """Plota comparação de métricas entre classificação multi-classe e binária."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metricas = ['acc', 'precision', 'recall', 'f1']
    labels = ['Acurácia', 'Precisão', 'Recall', 'F1-Score']
    valores_multi = [resultados_multi[m] for m in metricas]
    valores_bin = [resultados_bin[m] for m in metricas]
    x = np.arange(len(labels))
    width = 0.35
    
    for idx, (ax, valor_multi, valor_bin, label) in enumerate(zip(axes.flat, valores_multi, valores_bin, labels)):
        bars1 = ax.bar(x[0] - width/2, valor_multi, width, label='Multi-classe', color='steelblue', edgecolor='black')
        bars2 = ax.bar(x[0] + width/2, valor_bin, width, label='Binária', color='forestgreen', edgecolor='black')
        ax.set_ylabel(label, fontsize=11, fontweight='bold')
        ax.set_ylim([0, 1.1])
        # Adicionar valores nas barras
        for bar in bars1 + bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Comparação de Métricas (Modelos Otimizados)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('comparacao_metricas_classificacao.png', dpi=300)
    plt.show()

# ================= EXECUÇÃO =================

if __name__ == "__main__":
    print("="*70)
    print(" CLASSIFICAÇÃO COM OTIMIZAÇÃO DE HIPERPARÂMETROS (OPTUNA)")
    print("="*70)
    
    if not os.path.exists(ARQUIVO_DADOS):
        sys.exit(f"ERRO: {ARQUIVO_DADOS} não encontrado.")
    
    print("\n1. Carregando dados...")
    df_raw = pd.read_csv(ARQUIVO_DADOS, sep=';', decimal=',')
    
    if os.path.exists(ARQUIVO_AGRONOMICO):
        df = cruzar_com_excel(df_raw, ARQUIVO_AGRONOMICO)
    else:
        df = df_raw.copy()
        df['Tratamento'] = 0
    
    print("2. Selecionando bandas...")
    cols_totais = [c for c in df.columns if c.startswith('d1_Band_')]
    cols_selecionadas, _ = selecionar_colunas_por_lista(cols_totais, BANDAS_ALVO)
    
    X = df[cols_selecionadas].apply(pd.to_numeric, errors='coerce').fillna(0)
    y_tratamento = df['Tratamento'].astype(str)
    y_tratamento_num = pd.to_numeric(df['Tratamento'], errors='coerce').fillna(0)
    
    # 3. Multi-Classe
    print("\n" + "="*70)
    print(" CLASSIFICAÇÃO MULTI-CLASSE")
    print("="*70)
    resultados_multi = classificar_multiclasse(X, y_tratamento, cols_selecionadas)
    
    # 4. Binária
    print("\n" + "="*70)
    print(" CLASSIFICAÇÃO BINÁRIA")
    print("="*70)
    resultados_bin = classificar_binaria(X, y_tratamento_num, cols_selecionadas)
    
    # 5. Comparação
    print("\n" + "="*70)
    print(" COMPARAÇÃO FINAL")
    print("="*70)
    plotar_comparacao_metricas(resultados_multi, resultados_bin)
    
    print(f"\n📊 RESUMO:")
    print(f"  Multi-Classe F1: {resultados_multi['f1']:.4f}")
    print(f"  Binária F1:      {resultados_bin['f1']:.4f}")
    print("\n✓ Processo finalizado.")