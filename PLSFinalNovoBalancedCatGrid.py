import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# ================= CONFIGURAÇÕES =================
ARQUIVO_DADOS = 'spectral_indices_rois_final_v3.csv'
ARQUIVO_AGRONOMICO = os.path.join('Dataset', 'PlanilhaFiltrada.xlsx')
SEED = 42
BANDAS_ALVO = [430, 450, 460, 500, 520, 550, 600, 625, 640, 660, 685, 720, 722, 740, 990, 995, 970]

# ================= FUNÇÕES AUXILIARES =================
def extrair_valor_onda(nome_coluna):
    try: return float(nome_coluna.replace('d1_', '').replace('Band_', '').replace('nm', ''))
    except: return None

def cruzar_com_excel(df_espectral, arquivo_excel):
    try:
        df_agro = pd.read_excel(arquivo_excel, header=2)
        df_agro.columns = [str(c).strip().lower() for c in df_agro.columns]
    except: return df_espectral
    
    col_id = next((c for c in df_agro.columns if 'id' in c and 'parcela' in c), 'id parcela')
    col_dose = next((c for c in df_agro.columns if 'dose' in c), None)
    
    df_espectral['ID_Numeric'] = df_espectral['ID_Amostra'].astype(int)
    cols_keep = [c for c in df_espectral.columns if c != 'Dose_N']
    df_merge = df_espectral[cols_keep].merge(df_agro[[col_id, col_dose]], left_on='ID_Numeric', right_on=col_id, how='left')
    df_merge.rename(columns={col_dose: 'Dose_N'}, inplace=True)
    df_merge['Dose_N'] = pd.to_numeric(df_merge['Dose_N'], errors='coerce').fillna(0)
    return df_merge.loc[:, ~df_merge.columns.duplicated()]

def selecionar_colunas(df, bandas):
    selec = []
    for b in bandas:
        for c in df.columns:
            if c.startswith('d1_Band_'):
                wl = extrair_valor_onda(c)
                if wl and abs(wl - b) <= 1.0 and c not in selec: selec.append(c); break
    return selec

# ================= VISUALIZAÇÃO t-SNE =================
def plotar_tsne(X, y, titulo="Distribuição Espectral (t-SNE)"):
    print(f"\n🎨 Gerando t-SNE para {titulo}...")
    
    # Normalizar para visualização
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Redução de dimensionalidade (CORREÇÃO: removido n_iter explícito)
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=30)
    X_embedded = tsne.fit_transform(X_scaled)
    
    # Plot
    plt.figure(figsize=(10, 8))
    df_plot = pd.DataFrame(X_embedded, columns=['x', 'y'])
    df_plot['Dose'] = y
    
    # Define ordem das cores para ficar lógico (0 -> 360)
    sns.scatterplot(data=df_plot, x='x', y='y', hue='Dose', palette='viridis', 
                    style='Dose', s=80, alpha=0.8)
    
    plt.title(titulo, fontsize=15, fontweight='bold')
    plt.xlabel('Dimensão t-SNE 1')
    plt.ylabel('Dimensão t-SNE 2')
    plt.legend(title='Dose N (kg/ha)', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('analise_tsne_doses.png', dpi=300)
    print("✓ Salvo: analise_tsne_doses.png")
    plt.show()

# ================= EXECUÇÃO =================
if __name__ == "__main__":
    print("="*70)
    print(" OTIMIZAÇÃO FINAL & VISUALIZAÇÃO")
    print("="*70)
    
    # 1. Carga
    if not os.path.exists(ARQUIVO_DADOS):
        sys.exit(f"ERRO: {ARQUIVO_DADOS} não encontrado.")

    df = pd.read_csv(ARQUIVO_DADOS, sep=';', decimal=',')
    if os.path.exists(ARQUIVO_AGRONOMICO): df = cruzar_com_excel(df, ARQUIVO_AGRONOMICO)
    else: df['Dose_N'] = 0
    
    cols = selecionar_colunas(df, BANDAS_ALVO)
    X = df[cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    y = df['Dose_N'].astype(int)
    
    print(f"   {len(X)} amostras carregadas.")

    # 2. Visualização t-SNE
    plotar_tsne(X, y)
    
    # 3. Preparação para Tuning
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.3, random_state=SEED, stratify=y_enc)
    
    print("\n⚖️  Aplicando SMOTE no Treino...")
    smote = SMOTE(random_state=SEED, k_neighbors=1)
    X_tr_sm, y_tr_sm = smote.fit_resample(X_train, y_train)
    
    # 4. Grid de Hiperparâmetros (NATIVO DO CATBOOST)
    print("\n🔍 Iniciando Busca de Hiperparâmetros (CatBoost Native)...")
    
    # Grid de busca
    param_grid = {
        'depth': [4, 6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'iterations': [300, 500, 800],
        'l2_leaf_reg': [1, 3, 5, 9],
        'bagging_temperature': [0, 1]
    }
    
    # Modelo base
    model = CatBoostClassifier(
        loss_function='MultiClass', 
        verbose=0, 
        random_seed=SEED, 
        allow_writing_files=False
    )
    
    # Busca nativa (Ignora erros de versão do sklearn)
    search_results = model.randomized_search(
        param_grid,
        X=X_tr_sm,
        y=y_tr_sm,
        cv=3,
        n_iter=20, # 20 tentativas aleatórias
        partition_random_seed=SEED,
        calc_cv_statistics=False,
        search_by_train_test_split=False, # Usa CV real
        verbose=False,
        plot=False
    )
    
    print(f"\n🏆 Melhores Parâmetros: {search_results['params']}")
    
    # 5. Avaliação Final do Modelo Otimizado
    print("\n📊 Avaliando Modelo Campeão no Test Set...")
    
    # Treina o modelo final com os melhores parâmetros encontrados
    best_model = CatBoostClassifier(
        loss_function='MultiClass',
        verbose=0,
        random_seed=SEED,
        allow_writing_files=False,
        **search_results['params'] # Desempacota os melhores params
    )
    
    best_model.fit(X_tr_sm, y_tr_sm)
    
    y_pred = best_model.predict(X_test)
    if y_pred.ndim > 1: y_pred = y_pred.ravel()
    
    acc = accuracy_score(y_test, y_pred)
    print(f"\n🚀 Acurácia Final Otimizada: {acc:.4f}")
    
    # Relatório
    print(classification_report(y_test, y_pred, target_names=[str(c) for c in le.classes_], digits=4))
    
    # Matriz Final
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'Matriz Otimizada (Acc: {acc:.4f})', fontweight='bold')
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.tight_layout()
    plt.savefig('matriz_final_otimizada.png', dpi=300)
    print("✓ Salvo: matriz_final_otimizada.png")
    
    plt.show()
    
    print("\n✅ PROCESSO FINALIZADO.")