import os
import glob
import numpy as np
import pandas as pd
import spectral.io.envi as envi
import matplotlib.pyplot as plt

# ================= CONFIGURAÇÕES =================
PASTA_DO_SCRIPT = os.path.dirname(os.path.abspath(__file__))
DIRETORIO_DADOS = os.path.join(PASTA_DO_SCRIPT, r'D:\AgroSAT\A\B') # Dataset

# Configurações de Banda
WAVELENGTH_RED = 670.0
WAVELENGTH_NIR = 800.0
NDVI_THRESHOLD = 0.25 

# ================= FUNÇÕES =================
def encontrar_banda_mais_proxima(header, alvo_wl):
    try:
        wls = [float(w) for w in header['wavelength']]
        wls_arr = np.array(wls)
        idx = (np.abs(wls_arr - alvo_wl)).argmin()
        return idx, wls_arr[idx]
    except KeyError:
        return None, None

def processar_imagem(caminho_arquivo):
    path_bil = caminho_arquivo
    path_hdr = caminho_arquivo + ".hdr"
    nome_amostra = os.path.basename(caminho_arquivo).replace('.bil', '')

    if not os.path.exists(path_hdr):
        print(f"HDR não encontrado para: {nome_amostra}")
        return None, None

    try:
        img_obj = envi.open(path_hdr, path_bil)
        dados = img_obj.load()
        metadata = img_obj.metadata
        
        idx_red, _ = encontrar_banda_mais_proxima(metadata, WAVELENGTH_RED)
        idx_nir, _ = encontrar_banda_mais_proxima(metadata, WAVELENGTH_NIR)
        
        # NDVI Mask
        banda_red = dados[:, :, idx_red].astype(float)
        banda_nir = dados[:, :, idx_nir].astype(float)
        denominador = (banda_nir + banda_red)
        denominador[denominador == 0] = 0.00001
        ndvi = (banda_nir - banda_red) / denominador
        
        mascara = ndvi > NDVI_THRESHOLD
        
        # Extração
        rows, cols, bands = dados.shape
        dados_flat = dados.reshape(-1, bands)
        mascara_flat = mascara.reshape(-1)
        pixels_filtrados = dados_flat[mascara_flat, :]
        
        if pixels_filtrados.shape[0] == 0:
            print(f"AVISO: {nome_amostra} - Nenhum pixel de planta detectado.")
            return None, None

        espectro_medio = np.mean(pixels_filtrados, axis=0)
        wls_float = [float(w) for w in metadata['wavelength']]
        
        print(f"OK: {nome_amostra} processado.")
        return wls_float, espectro_medio

    except Exception as e:
        print(f"ERRO ao processar {nome_amostra}: {e}")
        return None, None

# ================= EXECUÇÃO AUTOMÁTICA =================
print(f"Buscando arquivos .bil em: {DIRETORIO_DADOS}")

# Localiza todos os arquivos .bil na pasta Dataset
todos_arquivos = glob.glob(os.path.join(DIRETORIO_DADOS, "*.bil"))
print(f"Total de arquivos encontrados: {len(todos_arquivos)}")

lista_dados = []
colunas_wavelength = []

for arquivo in todos_arquivos:
    wls, espectro = processar_imagem(arquivo)
    
    if espectro is not None:
        if not colunas_wavelength:
            colunas_wavelength = [f"Band_{w:.2f}nm" for w in wls]
        
        # Extrai ID e Parte do nome (Ex: '8M' -> ID=8, Parte=M)
        nome_arquivo = os.path.basename(arquivo).replace('.bil', '')
        
        # Lógica simples para separar ID e Parte (assume formato NumeroLetra, ex: 8M)
        # Se seus arquivos sempre terminam em M ou S, podemos usar isso:
        parte = nome_arquivo[-1] # Pega a última letra (M ou S)
        id_amostra = nome_arquivo[:-1] # Pega tudo antes da última letra (8)
        
        linha = {
            'ID_Amostra': id_amostra,
            'Parte_Planta': parte,
            'Nome_Arquivo': nome_arquivo
        }
        
        for i, valor in enumerate(espectro):
            linha[colunas_wavelength[i]] = valor
            
        lista_dados.append(linha)

# Salvar
if lista_dados:
    df = pd.DataFrame(lista_dados)
    
    # Organiza colunas
    cols_iniciais = ['ID_Amostra', 'Parte_Planta', 'Nome_Arquivo']
    cols_bandas = [c for c in df.columns if c not in cols_iniciais]
    df = df[cols_iniciais + cols_bandas]
    
    df.to_csv('leituras_hiperspectrais_COMPLETO.csv', index=False, sep=';', decimal=',')
    print(f"\nCONCLUÍDO! Tabela mestre gerada com {len(df)} linhas.")
else:
    print("Nenhum dado extraído.")