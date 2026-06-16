import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import optuna
import re

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, roc_auc_score,
                             roc_curve)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# ================= CONFIGURAÇÕES =================
ARQUIVO_DADOS = 'spectral_indices_rois.csv'
ARQUIVO_AGRONOMICO = os.path.join('Dataset', 'PlanilhaFiltrada.xlsx')
SEED = 42

# AUMENTEI PARA 50 POIS O ESPAÇO DE BUSCA FICOU MAIOR
N_TRIALS = 50  

# Configuração do Banco de Dados para o Dashboard
STORAGE_URL = "sqlite:///db_optuna.sqlite3"

BANDAS_ALVO = [430, 450, 460, 500, 520, 550, 600, 625, 640, 660, 685, 720, 722, 740, 990, 995, 970]

# ================= FUNÇÕES AUXILIARES =================

def extrair_valor_onda(nome_coluna):
    try:
        limpo = nome_coluna.replace('d1_', '').replace('Band_', '').replace('nm', '')
        return float(limpo)
    except:
        return None

def extrair_numero_do_texto(valor):
    """
    Tenta encontrar um número dentro de uma string (ex: 'T1' -> 1.0, '150 kg' -> 150.0).
    Se não achar nada (ex: 'Controle'), retorna 0.0.
    """
    if isinstance(valor, (int, float)):
        return float(valor)
    
    texto = str(valor)
    match = re.search(r"(\d+(\.\d+)?)", texto)
    if match:
        return float(match.group(1))
    return 0.0

def cruzar_com_excel(df_espectral, arquivo_excel):
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

# ================= LÓGICA DO OPTUNA (EXPANDIDA) =================

def otimizar_hiperparametros(X_train, y_train, study_name):
    """
    Otimiza hiperparâmetros com espaço de busca expandido.
    """
    
    # Define limite para SMOTE
    contagem_classes = pd.Series(y_train).value_counts()
    min_samples = contagem_classes.min()
    max_k_smote = max(1, min_samples - 1)
    max_k_smote = min(max_k_smote, 7)

    def objective(trial):
        # --- Random Forest Hyperparameters ---
        
        # Número de árvores (Expandido até 500)
        n_estimators = trial.suggest_int('n_estimators', 50, 500)
        
        # Profundidade (Expandido até 50)
        max_depth = trial.suggest_int('max_depth', 5, 50)
        
        # Mínimo para dividir um nó (Expandido até 20)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        
        # [NOVO] Mínimo para ser uma folha (Ajuda a evitar overfitting)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
        
        # Features por split (Adicionado None = usa todas as features)
        max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
        
        # [NOVO] Critério de qualidade
        criterion = trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])
        
        # [NOVO] Bootstrap e Max Samples
        bootstrap = trial.suggest_categorical('bootstrap', [True, False])
        
        # max_samples só funciona se bootstrap=True
        if bootstrap:
            max_samples = trial.suggest_float('max_samples', 0.5, 1.0)
        else:
            max_samples = None

        # --- SMOTE Hyperparameters ---
        k_neighbors = trial.suggest_int('k_neighbors', 1, max_k_smote)

        # Pipeline
        pipeline = Pipeline([
            ('smote', SMOTE(random_state=SEED, k_neighbors=k_neighbors)),
            ('rf', RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf, # Novo
                max_features=max_features,
                criterion=criterion, # Novo
                bootstrap=bootstrap, # Novo
                max_samples=max_samples, # Novo
                class_weight='balanced',
                random_state=SEED,
                n_jobs=-1
            ))
        ])
        
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1_weighted', n_jobs=-1)
        return scores.mean()

    # Cria/Carrega estudo no SQLite
    study = optuna.create_study(
        direction='maximize',
        storage=STORAGE_URL,
        study_name=study_name,
        load_if_exists=True
    )
    
    print(f"  💾 Salvando estudo '{study_name}' em {STORAGE_URL}")
    study.optimize(objective, n_trials=N_TRIALS)
    
    return study.best_params

# ================= FUNÇÕES DE CLASSIFICAÇÃO =================

def classificar_multiclasse(X, y, cols_nomes, titulo="Classificação Multi-Classe"):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.3, random_state=SEED, stratify=y_encoded
    )
    
    print(f"\n  📊 Treino: {len(X_train)} | Teste: {len(X_test)}")
    print(f"  🔍 Iniciando otimização expandida com Optuna ({N_TRIALS} tentativas)...")
    
    nome_estudo = "estudo_multiclasse_v2" # Mudei o nome para não misturar com o estudo anterior
    best_params = otimizar_hiperparametros(X_train, y_train, study_name=nome_estudo)
    
    print(f"  ✓ Melhores parâmetros: {best_params}")
    
    # Remove parametros que não vão direto pro RF
    k_smote = best_params.pop('k_neighbors')
    
    pipeline = Pipeline([
        ('smote', SMOTE(random_state=SEED, k_neighbors=k_smote)),
        ('rf', RandomForestClassifier(**best_params, class_weight='balanced', random_state=SEED, n_jobs=-1))
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"\n  ✓ Acurácia: {acc:.4f} | F1: {f1:.4f}")
    
    # Gráficos
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.1%', cmap='Blues', cbar_kws={'label': 'Porcentagem de Acerto'},
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    ax.set_title(f"Matriz de Confusão (%) - {titulo}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

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
    plt.show()
    
    return {'acc': acc, 'f1': f1, 'precision': precision, 'recall': recall}

def classificar_binaria(X, y_raw, cols_nomes, titulo="Classificação Binária"):
    y_numerico = y_raw.apply(extrair_numero_do_texto)
    
    print(f"\n  🔍 Debug Binário (Amostra):")
    print(f"     Original: {y_raw.unique()[:5]}")
    print(f"     Extraído: {y_numerico.unique()}")

    y_binary = np.where(y_numerico > 0, "Com N", "Sem N")
    
    if len(np.unique(y_binary)) < 2:
        print(f"\n  ❌ ERRO CRÍTICO: Apenas uma classe encontrada: {np.unique(y_binary)}")
        return {'acc': 0, 'f1': 0, 'precision': 0, 'recall': 0, 'roc_auc': 0}

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_binary)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.3, random_state=SEED, stratify=y_encoded
    )
    
    print(f"\n  📊 Treino: {len(X_train)} | Teste: {len(X_test)}")
    print(f"  🔍 Iniciando otimização expandida com Optuna ({N_TRIALS} tentativas)...")
    
    nome_estudo = "estudo_binario_v2" # Mudei o nome para não misturar
    best_params = otimizar_hiperparametros(X_train, y_train, study_name=nome_estudo)
    
    print(f"  ✓ Melhores parâmetros: {best_params}")
    
    k_smote = best_params.pop('k_neighbors')
    pipeline = Pipeline([
        ('smote', SMOTE(random_state=SEED, k_neighbors=k_smote)),
        ('rf', RandomForestClassifier(**best_params, class_weight='balanced', random_state=SEED, n_jobs=-1))
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.named_steps['rf'].predict_proba(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    try:
        roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    except:
        roc_auc = 0.5
        
    print(f"\n  ✓ Acurácia: {acc:.4f} | ROC-AUC: {roc_auc:.4f}")
    
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(cm, annot=True, fmt='.1%', cmap='Greens', cbar_kws={'label': 'Porcentagem de Acerto'},
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax1)
    ax1.set_title(f"Matriz de Confusão (%) - {titulo}", fontsize=12, fontweight='bold')
    
    if len(np.unique(y_test)) > 1:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
        ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Caso Aleatório')
    ax2.set_title('Curva ROC', fontsize=12, fontweight='bold')
    ax2.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

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
    plt.show()

    return {'acc': acc, 'f1': f1, 'precision': precision, 'recall': recall, 'roc_auc': roc_auc}

def plotar_comparacao_metricas(resultados_multi, resultados_bin):
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
        for bar in bars1 + bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Comparação de Métricas (Otimização Expandida)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

# ================= EXECUÇÃO =================

if __name__ == "__main__":
    if not os.path.exists(ARQUIVO_DADOS):
        sys.exit(f"ERRO: {ARQUIVO_DADOS} não encontrado.")
    
    print("\n1. Carregando dados...")
    df_raw = pd.read_csv(ARQUIVO_DADOS, sep=';', decimal=',')
    if os.path.exists(ARQUIVO_AGRONOMICO):
        df = cruzar_com_excel(df_raw, ARQUIVO_AGRONOMICO)
    else:
        df = df_raw.copy(); df['Tratamento'] = 0
    
    cols_totais = [c for c in df.columns if c.startswith('d1_Band_')]
    cols_selecionadas, _ = selecionar_colunas_por_lista(cols_totais, BANDAS_ALVO)
    
    X = df[cols_selecionadas].apply(pd.to_numeric, errors='coerce').fillna(0)
    y_tratamento = df['Tratamento'].astype(str)
    
    resultados_multi = classificar_multiclasse(X, y_tratamento, cols_selecionadas)
    resultados_bin = classificar_binaria(X, df['Tratamento'], cols_selecionadas)
    plotar_comparacao_metricas(resultados_multi, resultados_bin)
    
    print("\n✅ Script Finalizado.")
    print(f"Para ver o Dashboard, execute no terminal:\n  optuna-dashboard {STORAGE_URL}")