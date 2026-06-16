import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, roc_auc_score,
                             roc_curve)
from imblearn.over_sampling import SMOTE

# ================= CONFIGURAÇÕES =================
ARQUIVO_DADOS = 'spectral_indices_rois_final_v3.csv'
ARQUIVO_AGRONOMICO = os.path.join('Dataset', 'PlanilhaFiltrada.xlsx')
SEED = 42

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
    """Cruza dados espectrais com agronômicos do Excel."""
    try:
        df_agro = pd.read_excel(arquivo_excel, header=2)
        df_agro.columns = [str(c).strip().lower() for c in df_agro.columns]
        
    except Exception as e:
        print(f"⚠️  Erro ao ler Excel: {e}")
        if 'Dose_N' not in df_espectral.columns:
            df_espectral['Dose_N'] = 0
        return df_espectral
    
    col_id = next((c for c in df_agro.columns if 'id' in c and 'parcela' in c), 'id parcela')
    col_dose = next((c for c in df_agro.columns if 'dose' in c), None)
    col_clor_m = next((c for c in df_agro.columns if 'médio' in c and 'clorofila' in c), None)
    col_clor_s = next((c for c in df_agro.columns if 'superior' in c and 'clorofila' in c), None)
    
    # Prepara para o merge
    df_espectral['ID_Numeric'] = df_espectral['ID_Amostra'].astype(int)
    
    # Remove duplicata
    cols_para_manter = [c for c in df_espectral.columns if c != 'Dose_N']
    df_espectral_limpo = df_espectral[cols_para_manter]

    df_merge = df_espectral_limpo.merge(
        df_agro[[col_id, col_dose, col_clor_m, col_clor_s]],
        left_on='ID_Numeric',
        right_on=col_id,
        how='left'
    )
    
    df_merge.rename(columns={col_dose: 'Dose_N'}, inplace=True)
    df_merge['Dose_N'] = pd.to_numeric(df_merge['Dose_N'], errors='coerce').fillna(0)
    df_merge = df_merge.loc[:, ~df_merge.columns.duplicated()]
    
    return df_merge

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

# ================= CLASSIFICAÇÃO MULTI-CLASSE (CATBOOST SEM PIPELINE) =================

def classificar_multiclasse(X, y, cols_nomes, titulo="Classificação Multi-Classe (Doses)"):
    """Classificação multi-classe usando CatBoost com SMOTE manual."""
    
    # Label Encoding
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.3, random_state=SEED, stratify=y_encoded
    )
    
    print(f"\n  📊 Treino Original: {len(X_train)} | Teste: {len(X_test)}")
    
    # 1. Aplicar SMOTE manualmente (fora do Pipeline para evitar erro do sklearn 1.6)
    print("  ⚖️  Aplicando SMOTE...")
    smote = SMOTE(random_state=SEED, k_neighbors=1)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print(f"  📊 Treino Resampled: {len(X_resampled)}")

    # 2. Configurar CatBoost
    cat = CatBoostClassifier(
        iterations=500,
        learning_rate=0.1,
        depth=6,
        loss_function='MultiClass',
        verbose=0,
        random_seed=SEED,
        allow_writing_files=False
    )
    
    print("  🐱 Treinando CatBoost...")
    cat.fit(X_resampled, y_resampled)
    
    # Predição
    y_pred = cat.predict(X_test)
    # CatBoost às vezes retorna array de arrays (n, 1), precisamos de (n,)
    if y_pred.ndim > 1:
        y_pred = y_pred.ravel()
    
    # Métricas
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"\n  ✓ Acurácia: {acc:.4f}")
    print(f"  ✓ Precisão: {precision:.4f}")
    print(f"  ✓ Recall: {recall:.4f}")
    print(f"  ✓ F1-Score: {f1:.4f}")
    
    # Relatório
    print(f"\n  Relatório por Classe:")
    print(classification_report(y_test, y_pred, target_names=[str(c) for c in le.classes_], zero_division=0))
    
    # Matriz
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Contagem'},
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    ax.set_title(f"Matriz de Confusão - {titulo} (CatBoost)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Predito (kg/ha)", fontsize=12)
    ax.set_ylabel("Verdadeiro (kg/ha)", fontsize=12)
    plt.tight_layout()
    plt.savefig(f'matriz_confusao_cat_{titulo.lower().replace(" ", "_")}.png', dpi=300)
    print(f"  ✓ Salvo: matriz_confusao_cat_{titulo.lower().replace(' ', '_')}.png")
    plt.show()
    
    # Feature Importance
    importancias = cat.get_feature_importance()
    indices = np.argsort(importancias)[::-1][:10]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    wls_top = [extrair_valor_onda(cols_nomes[i]) for i in indices]
    ax.barh(range(len(indices)), importancias[indices], color='steelblue', edgecolor='black')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([f'{wl:.0f}nm' for wl in wls_top])
    ax.set_xlabel('Importância (Feature Importance)', fontsize=12)
    ax.set_title(f"Top 10 Bandas - {titulo}", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(f'importancia_cat_multi.png', dpi=300)
    print(f"  ✓ Salvo: importancia_cat_multi.png")
    plt.show()
    
    return {
        'acc': acc, 'precision': precision, 'recall': recall, 'f1': f1,
        'cm': cm, 'classes': le.classes_, 'y_test': y_test, 'y_pred': y_pred
    }

# ================= CLASSIFICAÇÃO BINÁRIA (CATBOOST SEM PIPELINE) =================

def classificar_binaria(X, y_dose, cols_nomes, titulo="Classificação Binária (Com/Sem N)"):
    """Classificação binária usando CatBoost com SMOTE manual."""
    
    y_binary = np.where(y_dose > 0, "Com N", "Sem N")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_binary)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.3, random_state=SEED, stratify=y_encoded
    )
    
    print(f"\n  📊 Treino Original: {len(X_train)} | Teste: {len(X_test)}")
    print(f"  Classes: {le.classes_}")
    
    # 1. SMOTE Manual
    print("  ⚖️  Aplicando SMOTE...")
    smote = SMOTE(random_state=SEED, k_neighbors=1)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    # 2. CatBoost
    cat = CatBoostClassifier(
        iterations=500,
        learning_rate=0.1,
        depth=6,
        loss_function='Logloss',
        verbose=0,
        random_seed=SEED,
        allow_writing_files=False
    )
    
    print("  🐱 Treinando CatBoost...")
    cat.fit(X_resampled, y_resampled)
    
    # Predição
    y_pred = cat.predict(X_test)
    if y_pred.ndim > 1: y_pred = y_pred.ravel()
        
    y_pred_proba = cat.predict_proba(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    
    print(f"\n  ✓ Acurácia: {acc:.4f}")
    print(f"  ✓ Precisão: {precision:.4f}")
    print(f"  ✓ Recall: {recall:.4f}")
    print(f"  ✓ F1-Score: {f1:.4f}")
    print(f"  ✓ ROC-AUC: {roc_auc:.4f}")
    
    print(f"\n  Relatório por Classe:")
    print(classification_report(y_test, y_pred, target_names=[str(c) for c in le.classes_], zero_division=0))
    
    cm = confusion_matrix(y_test, y_pred)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar_kws={'label': 'Contagem'},
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax1)
    ax1.set_title(f"Matriz de Confusão - {titulo}", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Predito", fontsize=11)
    ax1.set_ylabel("Verdadeiro", fontsize=11)
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
    ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Caso Aleatório')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Taxa de Falsos Positivos', fontsize=11)
    ax2.set_ylabel('Taxa de Verdadeiros Positivos', fontsize=11)
    ax2.set_title('Curva ROC (CatBoost)', fontsize=12, fontweight='bold')
    ax2.legend(loc="lower right", fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'matriz_roc_cat_{titulo.lower().replace(" ", "_")}.png', dpi=300)
    print(f"  ✓ Salvo: matriz_roc_cat_{titulo.lower().replace(' ', '_')}.png")
    plt.show()
    
    # Importância Binária
    importancias = cat.get_feature_importance()
    indices = np.argsort(importancias)[::-1][:10]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    wls_top = [extrair_valor_onda(cols_nomes[i]) for i in indices]
    ax.barh(range(len(indices)), importancias[indices], color='forestgreen', edgecolor='black')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([f'{wl:.0f}nm' for wl in wls_top])
    ax.set_xlabel('Importância (Feature Importance)', fontsize=12)
    ax.set_title(f"Top 10 Bandas - {titulo}", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(f'importancia_cat_binaria.png', dpi=300)
    print(f"  ✓ Salvo: importancia_cat_binaria.png")
    plt.show()

    return {
        'acc': acc, 'precision': precision, 'recall': recall, 'f1': f1, 'roc_auc': roc_auc,
        'cm': cm, 'classes': le.classes_, 'fpr': fpr, 'tpr': tpr
    }

# ================= PLOTAR COMPARAÇÃO =================

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
        ax.set_xticks([])
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=10)
        
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Comparação de Métricas CatBoost: Multi-Classe vs Binária', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('comparacao_metricas_cat.png', dpi=300)
    print("\n✓ Salvo: comparacao_metricas_cat.png")
    plt.show()

# ================= EXECUÇÃO PRINCIPAL =================

if __name__ == "__main__":
    print("="*70)
    print(" CLASSIFICAÇÃO COM CATBOOST - RESPOSTA AO NITROGÊNIO")
    print("="*70)
    
    if not os.path.exists(ARQUIVO_DADOS):
        sys.exit(f"ERRO: {ARQUIVO_DADOS} não encontrado.")
    
    # 1. Carregar e preparar dados
    print("\n1. Carregando dados...")
    df_raw = pd.read_csv(ARQUIVO_DADOS, sep=';', decimal=',')
    
    print("2. Cruzando com Excel...")
    if os.path.exists(ARQUIVO_AGRONOMICO):
        df = cruzar_com_excel(df_raw, ARQUIVO_AGRONOMICO)
    else:
        df = df_raw.copy()
        df['Dose_N'] = 0
    
    # 3. Selecionar bandas (SEM adicionar novos índices)
    print("3. Selecionando bandas...")
    cols_totais = [c for c in df.columns if c.startswith('d1_Band_')]
    cols_selecionadas, _ = selecionar_colunas_por_lista(cols_totais, BANDAS_ALVO)
    
    X = df[cols_selecionadas].apply(pd.to_numeric, errors='coerce').fillna(0)
    y_dose = pd.to_numeric(df['Dose_N'], errors='coerce').fillna(0)
    y_dose_str = y_dose.astype(int).astype(str)
    
    print(f"   {len(X)} amostras × {len(cols_selecionadas)} bandas (Features originais)\n")
    
    # 4. Classificação Multi-Classe
    print("="*70)
    print(" CLASSIFICAÇÃO MULTI-CLASSE (CATBOOST)")
    print("="*70)
    
    resultados_multi = classificar_multiclasse(X, y_dose_str, cols_selecionadas,
                                               titulo="Classificação Multi-Classe")
    
    # 5. Classificação Binária
    print("\n" + "="*70)
    print(" CLASSIFICAÇÃO BINÁRIA (CATBOOST)")
    print("="*70)
    
    resultados_bin = classificar_binaria(X, y_dose, cols_selecionadas,
                                         titulo="Classificação Binária")
    
    # 6. Comparação
    print("\n" + "="*70)
    print(" COMPARAÇÃO DE MÉTRICAS")
    print("="*70)
    
    plotar_comparacao_metricas(resultados_multi, resultados_bin)
    
    # 7. Resumo
    print("\n" + "="*70)
    print(" RESUMO FINAL (CATBOOST)")
    print("="*70)
    
    print(f"\n📊 MULTI-CLASSE:")
    print(f"  Acurácia: {resultados_multi['acc']:.4f}")
    print(f"  F1-Score: {resultados_multi['f1']:.4f}")
    
    print(f"\n📊 BINÁRIA:")
    print(f"  Acurácia: {resultados_bin['acc']:.4f}")
    print(f"  F1-Score: {resultados_bin['f1']:.4f}")
    print(f"  ROC-AUC: {resultados_bin['roc_auc']:.4f}")
    
    print(f"\n{'='*70}\n✓ ANÁLISE COMPLETA FINALIZADA\n{'='*70}\n")