import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import shap
import itertools
import random

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# ================= CONFIGURAÇÕES DO TORNEIO =================
ARQUIVO_DADOS = 'spectral_indices_rois.csv'
ARQUIVO_AGRONOMICO = os.path.join('Dataset', 'PlanilhaFiltrada.xlsx')
SEED = 42

# Lista de Candidatas (17 bandas)
BANDAS_CANDIDATAS = [430, 450, 460, 500, 520, 550, 600, 625, 640, 660, 685, 720, 722, 740, 990, 995, 970]

# Configurações da Busca
K_MIN = 3               # Mínimo de bandas
K_MAX = 7               # Máximo de bandas
N_TENTATIVAS_POR_K = 20 # Quantas combinações testar por tamanho (Total = 5 * 20 = 100 treinos)
EPOCHS_BUSCA = 100      
EPOCHS_FINAL = 150      

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

def mapear_colunas_reais(df, bandas_alvo):
    mapa = {}
    for alvo in bandas_alvo:
        melhor_col = None
        menor_diff = 9999
        for col in df.columns:
            if 'Band_' in col and 'd1_' not in col:
                wl = extrair_valor_onda(col)
                if wl:
                    diff = abs(wl - alvo)
                    if diff < menor_diff and diff < 2.0:
                        menor_diff = diff
                        melhor_col = col
        if melhor_col:
            mapa[alvo] = melhor_col
    return mapa

# ================= MODELO SUPERVISIONADO DINÂMICO =================
def criar_ae_compacto(input_dim, num_classes, latent_dim):
    input_layer = Input(shape=(input_dim,))
    
    # Encoder
    x = Dense(16, activation='relu')(input_layer)
    x = BatchNormalization()(x)
    
    latent_space = Dense(latent_dim, activation='relu', name='latent')(x)
    
    # Decoder
    dec = Dense(16, activation='relu')(latent_space)
    output_recon = Dense(input_dim, activation='linear', name='recon')(dec)
    
    # Classifier
    clf = Dense(8, activation='relu')(latent_space)
    output_class = Dense(num_classes, activation='softmax', name='class')(clf)
    
    model = Model(inputs=input_layer, outputs=[output_recon, output_class])
    encoder = Model(inputs=input_layer, outputs=latent_space)
    
    model.compile(optimizer=Adam(0.002), 
                  loss={'recon': 'mse', 'class': 'categorical_crossentropy'},
                  loss_weights={'recon': 1.0, 'class': 0.5},
                  metrics={'class': 'accuracy'})
    return model, encoder

# ================= EXECUÇÃO PRINCIPAL =================
if __name__ == "__main__":
    print("="*70)
    print(f" ⚔️  GRANDE TORNEIO DE BANDAS ({K_MIN} a {K_MAX} features)")
    print("="*70)
    
    # 1. Preparação de Dados
    if not os.path.exists(ARQUIVO_DADOS): sys.exit("Arquivo não encontrado")
    
    df = pd.read_csv(ARQUIVO_DADOS, sep=';', decimal=',')
    if os.path.exists(ARQUIVO_AGRONOMICO): df = cruzar_com_excel(df, ARQUIVO_AGRONOMICO)
    else: df['Dose_N'] = 0
    
    df_filtered = df[df['Dose_N'] > 0].copy()
    y = df_filtered['Dose_N'].astype(int).values
    
    mapa_bandas = mapear_colunas_reais(df_filtered, BANDAS_CANDIDATAS)
    bandas_disponiveis = list(mapa_bandas.keys())
    print(f"Bandas disponíveis: {len(bandas_disponiveis)}")

    # Labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    y_cat = to_categorical(y_enc)
    num_classes = len(np.unique(y_enc))

    # Variáveis Globais do Torneio
    melhor_acc_global = 0.0
    melhor_combo_global = None
    historico_tamanhos = [] # Para guardar média de acc por tamanho

    # ================= LOOP DE TAMANHO (K) =================
    for k in range(K_MIN, K_MAX + 1):
        print(f"\n📦 TESTANDO SUBSETS DE TAMANHO {k}...")
        
        # Ajusta latent dim dinamicamente (metade das features, min 2)
        latent_dim_atual = max(2, k // 2 + 1)
        
        # Gerar combinações
        todas_combos = list(itertools.combinations(bandas_disponiveis, k))
        random.shuffle(todas_combos)
        combos_k = todas_combos[:N_TENTATIVAS_POR_K]
        
        scores_k = []
        
        for i, combo in enumerate(combos_k):
            cols_atuais = [mapa_bandas[b] for b in combo]
            
            # Dados
            X_subset = df_filtered[cols_atuais].values
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_subset)
            
            X_tr, X_val, y_tr, y_val, yc_tr, yc_val = train_test_split(
                X_scaled, y_enc, y_cat, test_size=0.2, random_state=SEED, stratify=y_enc
            )
            
            # Treino Rápido
            ae, enc = criar_ae_compacto(input_dim=k, num_classes=num_classes, latent_dim=latent_dim_atual)
            ae.fit(X_tr, {'recon': X_tr, 'class': yc_tr}, epochs=EPOCHS_BUSCA, batch_size=128, verbose=0)
            
            # Híbrido
            X_tr_lat = enc.predict(X_tr, verbose=0)
            X_val_lat = enc.predict(X_val, verbose=0)
            X_tr_final = np.hstack([X_tr, X_tr_lat])
            X_val_final = np.hstack([X_val, X_val_lat])
            
            clf = CatBoostClassifier(iterations=150, depth=4, verbose=0, allow_writing_files=False, random_seed=SEED)
            clf.fit(X_tr_final, y_tr)
            
            acc = accuracy_score(y_val, clf.predict(X_val_final))
            scores_k.append(acc)
            
            print(f"   Subset {k} [{i+1}/{len(combos_k)}] -> Acc: {acc:.4f} {combo}")
            
            if acc > melhor_acc_global:
                melhor_acc_global = acc
                melhor_combo_global = (k, combo, latent_dim_atual)
                print(f"   🌟 NOVO RECORDE GLOBAL!")

        # Média deste tamanho
        media_k = np.mean(scores_k)
        max_k = np.max(scores_k)
        historico_tamanhos.append({'k': k, 'media': media_k, 'max': max_k})
        print(f"   --> Resumo K={k}: Média {media_k:.4f} | Max {max_k:.4f}")

    # ================= ANÁLISE DE TRADE-OFF =================
    print("\n" + "="*70)
    print(" 📈 ANÁLISE DE TRADE-OFF (Dimensão vs Performance)")
    print("="*70)
    df_hist = pd.DataFrame(historico_tamanhos)
    
    plt.figure(figsize=(10, 6))
    plt.plot(df_hist['k'], df_hist['media'], marker='o', label='Acurácia Média', linewidth=2)
    plt.plot(df_hist['k'], df_hist['max'], marker='*', linestyle='--', label='Melhor Acurácia', color='green')
    plt.title('Trade-off: Número de Bandas vs Acurácia', fontsize=14)
    plt.xlabel('Número de Bandas (k)')
    plt.ylabel('Acurácia (Validação)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(range(K_MIN, K_MAX + 1))
    plt.savefig('tradeoff_bandas.png')
    print("✓ Gráfico de trade-off salvo (tradeoff_bandas.png)")
    
    # ================= TREINAMENTO CAMPEÃO FINAL =================
    k_win, combo_win, ld_win = melhor_combo_global
    print("\n" + "="*70)
    print(f" 🏆 GRANDE CAMPEÃO: {k_win} Bandas -> {combo_win}")
    print(f"    Acc Recorde: {melhor_acc_global:.4f}")
    print("="*70)
    
    print("Treinando modelo definitivo...")
    cols_campeao = [mapa_bandas[b] for b in combo_win]
    feat_names = cols_campeao + [f"Latent_{i}" for i in range(ld_win)]
    
    X = df_filtered[cols_campeao].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test, y_cat_train, y_cat_test = train_test_split(
        X_scaled, y_enc, y_cat, test_size=0.2, random_state=SEED, stratify=y_enc
    )
    
    # AE Robusto
    autoencoder, encoder = criar_ae_compacto(input_dim=k_win, num_classes=num_classes, latent_dim=ld_win)
    early = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    
    autoencoder.fit(
        X_train, {'recon': X_train, 'class': y_cat_train},
        validation_data=(X_test, {'recon': X_test, 'class': y_cat_test}),
        epochs=EPOCHS_FINAL, batch_size=64, callbacks=[early], verbose=1
    )
    
    # CatBoost Robusto
    X_tr_lat = encoder.predict(X_train)
    X_te_lat = encoder.predict(X_test)
    X_tr_final = np.hstack([X_train, X_tr_lat])
    X_te_final = np.hstack([X_test, X_te_lat])
    
    final_clf = CatBoostClassifier(iterations=1000, depth=6, learning_rate=0.03, verbose=0, allow_writing_files=False)
    final_clf.fit(X_tr_final, y_train)
    
    y_pred = final_clf.predict(X_te_final)
    if y_pred.ndim > 1: y_pred = y_pred.ravel()
    
    print("\nRELATÓRIO FINAL:")
    doses_reais = le.inverse_transform(np.unique(y_test))
    print(classification_report(y_test, y_pred, target_names=[str(d) for d in doses_reais], digits=4))
    
    # Matriz
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=doses_reais, yticklabels=doses_reais)
    plt.title(f'Matriz Final ({k_win} Bandas)')
    plt.tight_layout()
    plt.savefig('matriz_campeao_geral.png')
    
    # SHAP
    try:
        explainer = shap.TreeExplainer(final_clf)
        shap_vals = explainer.shap_values(X_te_final[:500])
        target = -1 
        vals_target = shap_vals[target] if isinstance(shap_vals, list) else shap_vals
        plt.figure()
        shap.summary_plot(vals_target, X_te_final[:500], feature_names=feat_names, show=False)
        plt.title(f"Impacto das {k_win} Melhores Bandas", fontsize=14)
        plt.tight_layout()
        plt.savefig('shap_campeao_geral.png')
    except: pass
    
    print("\n✅ TORNEIO COMPLETO FINALIZADO.")