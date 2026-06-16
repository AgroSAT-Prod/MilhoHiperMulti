import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, roc_auc_score,
                             roc_curve, auc)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# ================= CONFIGURAÇÕES =================
ARQUIVO_DADOS = 'spectral_indices_rois.csv'
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
    """Cruza dados espectrais com dados agronômicos, priorizando a coluna 'Tratamento'."""
    try:
        df_agro = pd.read_excel(arquivo_excel, header=2)
        df_agro.columns = [str(c).strip().lower() for c in df_agro.columns]
        
    except Exception as e:
        print(f"⚠️  Erro ao ler Excel: {e}")
        df_espectral['Tratamento'] = 'SEM N'
        return df_espectral
    
    col_id = next((c for c in df_agro.columns if 'id' in c and 'parcela' in c), 'id parcela')
    
    # Prioriza 'tratamento', depois tenta 'dose'
    col_alvo = next((c for c in df_agro.columns if 'tratamento' in c), None)
    if col_alvo is None:
        col_alvo = next((c for c in df_agro.columns if 'dose' in c), None)
    
    col_clor_m = next((c for c in df_agro.columns if 'médio' in c and 'clorofila' in c), None)
    col_clor_s = next((c for c in df_agro.columns if 'superior' in c and 'clorofila' in c), None)
    
    df_espectral['ID_Numeric'] = df_espectral['ID_Amostra'].astype(int)
    
    # Seleção de colunas para o merge
    cols_to_merge = [col_id]
    if col_alvo: cols_to_merge.append(col_alvo)
    if col_clor_m: cols_to_merge.append(col_clor_m)
    if col_clor_s: cols_to_merge.append(col_clor_s)

    df_merge = df_espectral.merge(
        df_agro[cols_to_merge],
        left_on='ID_Numeric',
        right_on=col_id,
        how='left'
    )
    
    # Cria target de clorofila (opcional)
    if col_clor_m and col_clor_s:
        df_merge['Y_Clorofila'] = np.where(
            df_merge['Parte'] == 'M',
            df_merge[col_clor_m],
            df_merge[col_clor_s]
        )
    else:
        df_merge['Y_Clorofila'] = 0
    
    # Renomeia a coluna alvo para 'Tratamento'
    if col_alvo:
        df_merge.rename(columns={col_alvo: 'Tratamento'}, inplace=True)
    else:
        df_merge['Tratamento'] = 'SEM N' # Valor default
    
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

# ================= ANÁLISE PCA (NOVO) =================

def analisar_pca(X, y_labels, titulo="PCA - Distribuição dos Tratamentos"):
    """
    Realiza PCA (2 componentes) e plota a distribuição das amostras.
    """
    print(f"\nCalculando PCA para: {titulo}...")

    # 1. Padronização dos dados (Mean=0, Variance=1) - CRÍTICO PARA PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. Aplicar PCA
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(X_scaled)

    # 3. Criar DataFrame com os resultados
    df_pca = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2'])
    df_pca['Tratamento'] = y_labels.values # .values garante alinhamento sem índice

    # 4. Calcular Variância Explicada
    var_explicada = pca.explained_variance_ratio_
    pc1_var = var_explicada[0] * 100
    pc2_var = var_explicada[1] * 100
    total_var = pc1_var + pc2_var

    print(f"  Variância Explicada: PC1={pc1_var:.2f}%, PC2={pc2_var:.2f}% (Total: {total_var:.2f}%)")

    # 5. Plotar
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x='PC1', y='PC2', 
        hue='Tratamento', 
        data=df_pca, 
        palette='viridis', 
        s=60, 
        alpha=0.7,
        edgecolor='k'
    )
    
    plt.title(f'{titulo}\nVariância Total Explicada: {total_var:.2f}%', fontsize=14, fontweight='bold')
    plt.xlabel(f'Componente Principal 1 ({pc1_var:.2f}%)', fontsize=12)
    plt.ylabel(f'Componente Principal 2 ({pc2_var:.2f}%)', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Tratamento')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    nome_arquivo = f'pca_{titulo.lower().replace(" ", "_")}.png'
    plt.savefig(nome_arquivo, dpi=300)
    print(f"  ✓ Gráfico salvo: {nome_arquivo}")
    plt.show()

# ================= CLASSIFICAÇÃO MULTI-CLASSE (Por Tratamento) =================

def classificar_multiclasse(X, y, cols_nomes, titulo="Classificação Multi-Classe (Tratamentos)"):
    """Classificação multi-classe para os diferentes tratamentos."""
    
    # Codificar labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.3, random_state=SEED, stratify=y_encoded
    )
    
    print(f"\n  📊 Treino: {len(X_train)} | Teste: {len(X_test)}")
    print(f"  Classes encontradas: {le.classes_}")
    
    # Pipeline com SMOTE + Random Forest
    pipeline = Pipeline([
        ('smote', SMOTE(random_state=SEED, k_neighbors=1)),
        ('rf', RandomForestClassifier(n_estimators=100, class_weight='balanced',
                                     max_depth=10, random_state=SEED, n_jobs=-1))
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    # Métricas
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"\n  ✓ Acurácia: {acc:.4f}")
    print(f"  ✓ Precisão: {precision:.4f}")
    print(f"  ✓ Recall: {recall:.4f}")
    print(f"  ✓ F1-Score: {f1:.4f}")
    
    # Relatório por classe
    print(f"\n  Relatório por Classe:")
    print(classification_report(y_test, y_pred, target_names=[str(c) for c in le.classes_], zero_division=0))
    
    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Contagem'},
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    ax.set_title(f"Matriz de Confusão - {titulo}", fontsize=14, fontweight='bold')
    ax.set_xlabel("Predito (Tratamento)", fontsize=12)
    ax.set_ylabel("Verdadeiro (Tratamento)", fontsize=12)
    plt.tight_layout()
    plt.savefig(f'matriz_confusao_{titulo.lower().replace(" ", "_")}.png', dpi=300)
    print(f"  ✓ Salvo: matriz_confusao_{titulo.lower().replace(' ', '_')}.png")
    # plt.show()
    
    # Importância das features
    rf_model = pipeline.named_steps['rf']
    importancias = rf_model.feature_importances_
    indices = np.argsort(importancias)[::-1][:10]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    wls_top = [extrair_valor_onda(cols_nomes[i]) for i in indices]
    ax.barh(range(len(indices)), importancias[indices], color='steelblue', edgecolor='black')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([f'{wl:.0f}nm' if wl else cols_nomes[i] for i, wl in enumerate(wls_top)])
    ax.set_xlabel('Importância', fontsize=12)
    ax.set_title(f"Top 10 Bandas - {titulo}", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(f'importancia_top10_{titulo.lower().replace(" ", "_")}.png', dpi=300)
    print(f"  ✓ Salvo: importancia_top10_{titulo.lower().replace(' ', '_')}.png")
    plt.show()
    
    return {
        'acc': acc, 'precision': precision, 'recall': recall, 'f1': f1,
        'cm': cm, 'classes': le.classes_, 'y_test': y_test, 'y_pred': y_pred
    }

# ================= CLASSIFICAÇÃO BINÁRIA (COM N vs SEM N) =================

def classificar_binaria(X, y_raw, cols_nomes, titulo="Classificação Binária (Com/Sem N)"):
    """
    Classificação binária: Com Nitrogênio vs Sem Nitrogênio.
    """
    
    # Padronizar texto: string, strip, maiúsculo
    y_str = y_raw.astype(str).str.strip().str.upper()
    
    # Lógica binária baseada em texto
    y_binary = np.where(y_str == 'SEM N', "Sem N", "Com N")
    
    # Validação de classes
    classes_unicas = np.unique(y_binary)
    if len(classes_unicas) < 2:
        print(f"\n⚠️ ERRO CRÍTICO: Apenas uma classe encontrada para binário: {classes_unicas}")
        print("Verifique se o nome 'SEM N' está escrito corretamente na planilha do Excel.")
        sys.exit("Interrompendo execução para evitar crash do modelo.")

    # Codificar labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_binary)
    
    # Split treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.3, random_state=SEED, stratify=y_encoded
    )
    
    print(f"\n  📊 Treino: {len(X_train)} | Teste: {len(X_test)}")
    print(f"  Classes Binárias: {le.classes_}")
    
    # Pipeline com SMOTE + Random Forest
    pipeline = Pipeline([
        ('smote', SMOTE(random_state=SEED, k_neighbors=1)),
        ('rf', RandomForestClassifier(n_estimators=100, class_weight='balanced',
                                     max_depth=10, random_state=SEED, n_jobs=-1))
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.named_steps['rf'].predict_proba(X_test)
    
    # Métricas
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # ROC-AUC
    try:
        if len(le.classes_) == 2:
            roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        else:
            roc_auc = 0.5
    except:
        roc_auc = 0.5 
    
    print(f"\n  ✓ Acurácia: {acc:.4f}")
    print(f"  ✓ Precisão: {precision:.4f}")
    print(f"  ✓ Recall: {recall:.4f}")
    print(f"  ✓ F1-Score: {f1:.4f}")
    print(f"  ✓ ROC-AUC: {roc_auc:.4f}")
    
    # Relatório por classe
    print(f"\n  Relatório por Classe:")
    print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))
    
    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Matriz de confusão
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar_kws={'label': 'Contagem'},
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax1)
    ax1.set_title(f"Matriz de Confusão - {titulo}", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Predito", fontsize=11)
    ax1.set_ylabel("Verdadeiro", fontsize=11)
    
    # Curva ROC
    if len(le.classes_) == 2:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
        ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Caso Aleatório')
    else:
        ax2.text(0.5, 0.5, "Roc Curve Indisponível", ha='center')

    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Taxa de Falsos Positivos', fontsize=11)
    ax2.set_ylabel('Taxa de Verdadeiros Positivos', fontsize=11)
    ax2.set_title('Curva ROC', fontsize=12, fontweight='bold')
    ax2.legend(loc="lower right", fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'matriz_roc_{titulo.lower().replace(" ", "_")}.png', dpi=300)
    print(f"  ✓ Salvo: matriz_roc_{titulo.lower().replace(' ', '_')}.png")
    plt.show()
    
    # Importância das features
    rf_model = pipeline.named_steps['rf']
    importancias = rf_model.feature_importances_
    indices = np.argsort(importancias)[::-1][:10]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    wls_top = [extrair_valor_onda(cols_nomes[i]) for i in indices]
    ax.barh(range(len(indices)), importancias[indices], color='forestgreen', edgecolor='black')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([f'{wl:.0f}nm' if wl else cols_nomes[i] for i, wl in enumerate(wls_top)])
    ax.set_xlabel('Importância', fontsize=12)
    ax.set_title(f"Top 10 Bandas - {titulo}", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(f'importancia_top10_{titulo.lower().replace(" ", "_")}.png', dpi=300)
    print(f"  ✓ Salvo: importancia_top10_{titulo.lower().replace(' ', '_')}.png")
    plt.show()
    
    return {
        'acc': acc, 'precision': precision, 'recall': recall, 'f1': f1, 'roc_auc': roc_auc,
        'cm': cm, 'classes': le.classes_, 
    }

# ================= PLOTAR COMPARAÇÃO DE MÉTRICAS =================

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
        ax.set_xticks([])
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=10)
        
        # Adicionar valores nas barras
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Comparação de Métricas: Multi-Classe vs Binária', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('comparacao_metricas_classificacao.png', dpi=300)
    print("\n✓ Salvo: comparacao_metricas_classificacao.png")
    plt.show()

# ================= EXECUÇÃO PRINCIPAL =================

if __name__ == "__main__":
    print("="*70)
    print(" CLASSIFICAÇÃO - RESPOSTA AO TRATAMENTO DE NITROGÊNIO")
    print("="*70)
    
    if not os.path.exists(ARQUIVO_DADOS):
        sys.exit(f"ERRO: {ARQUIVO_DADOS} não encontrado.")
    
    # 1. Carregar e preparar dados
    print("\n1. Carregando dados...")
    df_raw = pd.read_csv(ARQUIVO_DADOS, sep=';', decimal=',')
    
    print("2. Cruzando com Excel (Buscando 'Tratamento' ou 'Dose')...")
    if os.path.exists(ARQUIVO_AGRONOMICO):
        df = cruzar_com_excel(df_raw, ARQUIVO_AGRONOMICO)
    else:
        df = df_raw.copy()
        df['Tratamento'] = 'SEM N' 
    
    # 3. Selecionar bandas
    print("3. Selecionando bandas...")
    cols_totais = [c for c in df.columns if c.startswith('d1_Band_')]
    cols_selecionadas, _ = selecionar_colunas_por_lista(cols_totais, BANDAS_ALVO)
    
    X = df[cols_selecionadas].apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # Labels em Texto (para Multi-classe e PCA)
    y_tratamento = df['Tratamento'].astype(str)
    
    print(f"   {len(X)} amostras × {len(cols_selecionadas)} bandas")
    print(f"   Tratamentos identificados: {y_tratamento.unique()}\n")
    
    # 4. ANÁLISE PCA (NOVA ETAPA)
    print("="*70)
    print(" ANÁLISE DE COMPONENTES PRINCIPAIS (PCA)")
    print("="*70)
    analisar_pca(X, y_tratamento, titulo="PCA - Tratamentos")
    
    # 5. Classificação Multi-Classe
    print("\n" + "="*70)
    print(" CLASSIFICAÇÃO MULTI-CLASSE (Por Tratamento)")
    print("="*70)
    
    resultados_multi = classificar_multiclasse(X, y_tratamento, cols_selecionadas,
                                             titulo="Classificação Multi-Classe (Tratamentos)")
    
    # 6. Classificação Binária
    print("\n" + "="*70)
    print(" CLASSIFICAÇÃO BINÁRIA (Com N vs Sem N)")
    print("="*70)
    
    resultados_bin = classificar_binaria(X, y_tratamento, cols_selecionadas,
                                       titulo="Classificação Binária")
    
    # 7. Comparação de métricas
    print("\n" + "="*70)
    print(" COMPARAÇÃO DE MÉTRICAS")
    print("="*70)
    
    plotar_comparacao_metricas(resultados_multi, resultados_bin)
    
    # 8. Resumo final
    print("\n" + "="*70)
    print(" RESUMO FINAL")
    print("="*70)
    
    print(f"\n📊 MULTI-CLASSE:")
    print(f"  Acurácia: {resultados_multi['acc']:.4f}")
    print(f"  F1-Score: {resultados_multi['f1']:.4f}")
    
    print(f"\n📊 BINÁRIA:")
    print(f"  Acurácia: {resultados_bin['acc']:.4f}")
    print(f"  F1-Score: {resultados_bin['f1']:.4f}")
    print(f"  ROC-AUC: {resultados_bin['roc_auc']:.4f}")
    
    print(f"\n{'='*70}\n✓ ANÁLISE COMPLETA FINALIZADA\n{'='*70}\n")