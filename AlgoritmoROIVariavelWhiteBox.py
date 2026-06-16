import os
import glob
import cv2
import numpy as np
import pandas as pd
import spectral.io.envi as envi
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import gc  # Importante para limpar memória

# ================= CONFIGURAÇÕES =================
DATASET_DIR = r"D:\AgroSAT\A\B"
N_ROIS_INICIAL = 1000

# Bandas de interesse (nm)
WAVELENGTH_RED = 670.0
WAVELENGTH_NIR = 800.0

# AJUSTE 1: NDVI reduzido levemente para folhas mais claras/jovens
NDVI_THRESHOLD = 0.22

# AJUSTE 2: Máscara HSV expandida para tons verde-limão e pálidos
# Hue 40->25 (pega amarelados), Saturation 40->25 (pega pálidos)
LOWER_GREEN = np.array([25, 25, 25])
UPPER_GREEN = np.array([85, 255, 255])

# Processamento espectral
TRIM_MIN = 420.0
TRIM_MAX = 950.0
SG_WINDOW = 11
SG_POLY = 2
SG_DERIV = 1

# ================= FUNÇÕES DE UTILIDADE =================
def encontrar_banda(header, alvo):
    wls = np.array([float(w) for w in header['wavelength']])
    return np.argmin(np.abs(wls - alvo))

def encontrar_hdr(caminho_bil):
    pasta = os.path.dirname(caminho_bil)
    base = os.path.basename(caminho_bil).replace(".bil", "")
    candidatos = glob.glob(os.path.join(pasta, f"{base}*.hdr"))
    return candidatos[0] if candidatos else None

def mascara_hsv(caminho_tiff, shape_ref):
    if not os.path.exists(caminho_tiff):
        return None, None
    img = cv2.imread(caminho_tiff)
    if img is None:
        return None, None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
    mask = cv2.resize(mask, (shape_ref[1], shape_ref[0]), interpolation=cv2.INTER_NEAREST)
    return mask.astype(bool), img

def rois_sobrepostos(r1, r2):
    y0_1, y1_1, x0_1, x1_1 = r1
    y0_2, y1_2, x0_2, x1_2 = r2
    return not (x1_1 <= x0_2 or x1_2 <= x0_1 or y1_1 <= y0_2 or y1_2 <= y0_1)

def gerar_rois(mask, n_rois):
    coords = np.column_stack(np.where(mask))
    if len(coords) == 0:
        return [], 0
    
    area = len(coords)
    # Lado ajustado dinamicamente
    lado = max(int(np.sqrt(area / (n_rois * 1.2))), 3)
    if lado < 3: lado = 3
    
    rois = []
    tentativas = 0
    max_tentativas = 40000 
    
    while len(rois) < n_rois and tentativas < max_tentativas:
        tentativas += 1
        idx = np.random.randint(len(coords))
        y, x = coords[idx]

        y0 = max(0, y - lado // 2)
        y1 = min(mask.shape[0], y + lado // 2)
        x0 = max(0, x - lado // 2)
        x1 = min(mask.shape[1], x + lado // 2)
        
        if y1 - y0 < lado or x1 - x0 < lado:
            continue
        
        # Verifica se a área é válida na máscara
        if not np.all(mask[y0:y1, x0:x1]):
            continue
        
        novo_roi = (y0, y1, x0, x1)
        if any(rois_sobrepostos(novo_roi, r) for r in rois):
            continue
        
        rois.append(novo_roi)
    
    return rois, lado

# ================= FUNÇÕES DE ÍNDICES =================
def calcular_ndvi(nir, red):
    return (nir - red) / (nir + red + 1e-6)

def calcular_savi(nir, red, l=0.5):
    return ((nir - red) / (nir + red + l)) * (1 + l)

def extrair_wavelengths(cols):
    wls = []
    for c in cols:
        try:
            wls.append(float(c.replace('Band_', '').replace('d1_Band_', '').replace('nm', '')))
        except:
            pass
    return wls

def preprocessar_espectro(espectro, cols_bandas):
    wls = extrair_wavelengths(cols_bandas)
    indices_validos = [i for i, wl in enumerate(wls) if TRIM_MIN <= wl <= TRIM_MAX]
    
    if not indices_validos:
        return np.zeros_like(espectro), []

    espectro_trim = espectro[indices_validos]
    
    # SNV
    media = np.mean(espectro_trim)
    std = np.std(espectro_trim)
    espectro_snv = (espectro_trim - media) / (std + 1e-8)
    
    # Savitzky-Golay
    try:
        espectro_sg = savgol_filter(espectro_snv, window_length=SG_WINDOW, polyorder=SG_POLY, deriv=SG_DERIV)
    except:
        espectro_sg = espectro_snv
    
    return espectro_sg, indices_validos

# ================= PIPELINE PRINCIPAL =================
arquivos_bil = glob.glob(os.path.join(DATASET_DIR, "*.bil"))
print(f"Arquivos encontrados: {len(arquivos_bil)}\n")

# ================= PASSO 1: DETERMINAR O NÚMERO COMUM DE ROIs =================
print("=" * 60)
print("ETAPA 1: Determinando número máximo comum de ROIs")
print("=" * 60)

rois_possiveis = {}
imagens_caminhos = {} # Guarda apenas caminhos, não o cubo inteiro
imagens_sem_rois = []

for bil in arquivos_bil:
    nome = os.path.basename(bil).replace(".bil", "")
    print(f"Analisando: {nome}", end=" ")
    
    hdr = encontrar_hdr(bil)
    if hdr is None:
        print("-> ⚠️ HDR não encontrado.")
        continue
    
    tiff = os.path.join(DATASET_DIR, f"{nome}-RGB.tiff")
    
    try:
        # Carrega imagem
        img_obj = envi.open(hdr, bil)
        cube = img_obj.load() # Carrega na RAM
        meta = img_obj.metadata
        
        idx_red = encontrar_banda(meta, WAVELENGTH_RED)
        idx_nir = encontrar_banda(meta, WAVELENGTH_NIR)
        
        red = cube[:, :, idx_red].astype(float)
        nir = cube[:, :, idx_nir].astype(float)
        ndvi = calcular_ndvi(nir, red)
        
        # Limpa bandas individuais da memória
        del red, nir
        
        mask_ndvi = (ndvi > NDVI_THRESHOLD)
        if mask_ndvi.ndim == 3: mask_ndvi = mask_ndvi[:, :, 0]
        
        mask_hsv, _ = mascara_hsv(tiff, mask_ndvi.shape)
        
        # === CORREÇÃO DE MEMÓRIA: Apaga o cubo AGORA ===
        del cube, img_obj
        gc.collect() 
        # ===============================================

        if mask_hsv is None:
            print("-> ⚠️ RGB não encontrado.")
            continue
        
        mask_final = mask_ndvi & mask_hsv
        
        rois, lado = gerar_rois(mask_final, N_ROIS_INICIAL)
        n_rois_gerados = len(rois)
        
        print(f"| ROIs: {n_rois_gerados} | Lado: {lado}px")
        
        if n_rois_gerados > 0:
            rois_possiveis[nome] = n_rois_gerados
            # Armazena APENAS os caminhos
            imagens_caminhos[nome] = {
                'bil': bil, 'hdr': hdr, 'tiff': tiff
            }
        else:
            imagens_sem_rois.append(nome)
            
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        gc.collect()

if not rois_possiveis:
    print("\n❌ ERRO: Nenhuma imagem com ROIs válidos encontrada.")
    exit()

n_rois_comum = min(rois_possiveis.values())
print(f"\nNÚMERO COMUM DE ROIs: {n_rois_comum}")
print("=" * 60)

# ================= PASSO 2: EXTRAIR ROIs (RECARREGANDO IMAGENS) =================
print("=" * 60)
print("ETAPA 2: Extraindo ROIs de todas as imagens válidas")
print("=" * 60)

dados_csv = []

for nome, paths in imagens_caminhos.items():
    print(f"Processando: {nome}", end=" ")
    
    try:
        # 1. Recarrega a imagem
        img_obj = envi.open(paths['hdr'], paths['bil'])
        cube = img_obj.load()
        meta = img_obj.metadata
        
        # 2. Recalcula máscara
        idx_red = encontrar_banda(meta, WAVELENGTH_RED)
        idx_nir = encontrar_banda(meta, WAVELENGTH_NIR)
        
        red = cube[:, :, idx_red].astype(float)
        nir = cube[:, :, idx_nir].astype(float)
        ndvi = calcular_ndvi(nir, red)
        
        mask_ndvi = (ndvi > NDVI_THRESHOLD)
        if mask_ndvi.ndim == 3: mask_ndvi = mask_ndvi[:, :, 0]
        
        mask_hsv, _ = mascara_hsv(paths['tiff'], mask_ndvi.shape)
        mask_final = mask_ndvi & mask_hsv
        
        # 3. Gera ROIs finais
        rois, lado = gerar_rois(mask_final, n_rois_comum)
        
        # 4. Extração
        cols_bandas_raw = [f"Band_{float(wl):.2f}nm" for wl in meta['wavelength']]
        
        for roi_id, (y0, y1, x0, x1) in enumerate(rois):
            bloco = cube[y0:y1, x0:x1, :]
            espectro_medio = bloco.reshape(-1, bloco.shape[2]).mean(axis=0)
            
            espectro_proc, indices_validos = preprocessar_espectro(espectro_medio, cols_bandas_raw)
            
            red_roi = bloco[:, :, idx_red].mean()
            nir_roi = bloco[:, :, idx_nir].mean()
            
            linha = {
                "ID_Amostra": nome[:-1],
                "Parte": nome[-1],
                "ROI_ID": roi_id,
                "Lado_Pixel": lado,
                "NDVI": calcular_ndvi(nir_roi, red_roi),
                "SAVI": calcular_savi(nir_roi, red_roi)
            }
            
            for i, wl in enumerate(meta['wavelength']):
                linha[f"Band_{float(wl):.2f}nm"] = espectro_medio[i]
            
            for i, idx in enumerate(indices_validos):
                linha[f"d1_{cols_bandas_raw[idx]}"] = espectro_proc[i]
            
            dados_csv.append(linha)
            
        print(f"-> ✓ Feito ({len(rois)} ROIs)")
        
        # === LIMPEZA CRÍTICA ===
        del cube, img_obj, red, nir, bloco, mask_final
        gc.collect()
        
    except Exception as e:
        print(f"-> ❌ Erro: {e}")
        gc.collect()

# ================= SALVAR RESULTADOS =================
print("\n" + "=" * 60)
print("SALVANDO RESULTADOS")
print("=" * 60)

if dados_csv:
    df = pd.DataFrame(dados_csv)
    
    cols_meta = ["ID_Amostra", "Parte", "ROI_ID", "Lado_Pixel", "NDVI", "SAVI"]
    cols_bandas = [c for c in df.columns if c.startswith("Band_")]
    cols_derivadas = [c for c in df.columns if c.startswith("d1_")]
    
    df_final = df[cols_meta + cols_bandas + cols_derivadas]
    df_final.to_csv("spectral_indices_rois_balanced_final.csv", index=False, sep=";", decimal=",")
    
    print("\n✅ PROCESSAMENTO FINALIZADO")
    print(f"Total de ROIs extraídos: {len(df_final)}")
    print(f"Arquivo salvo: spectral_indices_rois_balanced_final.csv")
else:
    print("\n❌ ERRO: Nenhum dado foi extraído.")