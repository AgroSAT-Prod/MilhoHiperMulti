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

SEED = 42

# ================= FUNÇÕES =================

def avaliar_modelo(y_teste, y_pred, titulo, labels_nomes=None):
    print(f"\n{'='*20} {titulo} {'='*20}")
    print(f"Acurácia (Teste): {accuracy_score(y_teste, y_pred):.2%}")
    print("\nRelatório de Classificação:")
    print(classification_report(y_teste, y_pred, target_names=labels_nomes))

    cm = confusion_matrix(y_teste, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels_nomes if labels_nomes else "auto",
                yticklabels=labels_nomes if labels_nomes else "auto")
    plt.title(titulo)
    plt.xlabel("Predição")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.show()

def plotar_importancia(modelo, colunas_X, titulo):
    importancias = modelo.feature_importances_
    indices = np.argsort(importancias)[::-1][:15]

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(indices)), importancias[indices])
    plt.xticks(range(len(indices)),
               [colunas_X[i].replace('d1_Band_', '').replace('Band_', '').replace('nm', '') + 'nm'
                for i in indices],
               rotation=45)
    plt.title(f"Top 15 Bandas - {titulo}")
    plt.ylabel("Importância (Gini)")
    plt.tight_layout()
    plt.show()

def plotar_assinatura_media_por_classe(df, bandas, alvo, titulo):
    plt.figure(figsize=(12, 6))
    wls = [float(c.replace('d1_Band_', '').replace('Band_', '').replace('nm', '')) for c in bandas]
    grupos = df.groupby(alvo)[bandas].mean()
    cores = plt.cm.viridis(np.linspace(0, 1, len(grupos)))

    for cor, (classe, linha) in zip(cores, grupos.iterrows()):
        plt.plot(wls, linha, label=str(classe), color=cor)

    plt.axvspan(690, 740, color='gray', alpha=0.1)
    plt.legend()
    plt.title(titulo)
    plt.xlabel("Comprimento de Onda (nm)")
    plt.ylabel("Reflectância / Derivada")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# ================= EXECUÇÃO =================

if __name__ == "__main__":

    if not os.path.exists(ARQUIVO_DADOS):
        print("Arquivo não encontrado.")
        sys.exit()

    df = pd.read_csv(ARQUIVO_DADOS, sep=';', decimal=',')
    print(f"Dataset carregado: {len(df)} amostras")

    bandas = [c for c in df.columns if c.startswith('d1_Band_') or c.startswith('Band_')]
    X = df[bandas].apply(pd.to_numeric, errors='coerce').fillna(0)
    y_dose = df['Dose_N'].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_dose, test_size=0.3, random_state=SEED, stratify=y_dose
    )

    print("\nTreinando RF Multiclasse com OOB...")
    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=SEED,
        oob_score=True,
        n_jobs=-1
    )

    rf.fit(X_train, y_train)

    print(f"OOB Score: {rf.oob_score_:.2%}")

    # ===== Avaliação Hold-out =====
    y_pred = rf.predict(X_test)
    classes = sorted(rf.classes_, key=lambda x: float(x))
    avaliar_modelo(y_test, y_pred, "Hold-out (Doses)", classes)

    # ===== Avaliação OOB (CORRETA) =====
    oob_probs = rf.oob_decision_function_
    mask_valid = ~np.isnan(oob_probs).any(axis=1)

    y_true_oob = y_train.reset_index(drop=True)[mask_valid]
    y_pred_oob = rf.classes_[np.argmax(oob_probs[mask_valid], axis=1)]

    print("\nAvaliação OOB (somente treino):")
    print(f"Amostras válidas OOB: {mask_valid.sum()} / {len(y_train)}")
    print(classification_report(y_true_oob, y_pred_oob))

    plotar_importancia(rf, bandas, "Doses")
    plotar_assinatura_media_por_classe(df, bandas, 'Dose_N', "Doses de Nitrogênio")

    print("\nPipeline finalizado com sucesso ✅")
