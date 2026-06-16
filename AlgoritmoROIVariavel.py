import os
import glob
import cv2
import numpy as np
import pandas as pd
import spectral.io.envi as envi
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# ================= CONFIGURAÇÕES =================
DATASET_DIR = r"D:\AgroSAT\A\B"
N_ROIS = 1000

# Bandas de interesse (nm)
WAVELENGTH_RED = 670.0
WAVELENGTH_NIR = 800.0
NDVI_THRESHOLD = 0.25

# Máscara HSV para verde
LOWER_GREEN = np.array([40, 40, 40])
UPPER_GREEN = np.array([85, 255, 255])

# Processamento espectral
TRIM_MIN = 420.0
TRIM_MAX = 950.0
SG_WINDOW = 11
SG_POLY = 2
SG_DERIV = 1

# ================= FUNÇÕES DE UTILIDADE =================

def encontrar_banda(header, alvo):
    """Encontra índice da banda mais próxima ao comprimento de onda alvo"""
    wls = np.array([float(w) for w in header['wavelength']])
    return np.argmin(np.abs(wls - alvo))

def encontrar_hdr(caminho_bil):
    """Localiza arquivo .hdr correspondente ao .bil"""
    pasta = os.path.dirname(caminho_bil)
    base = os.path.basename(caminho_bil).replace(".bil", "")
    candidatos = glob.glob(os.path.join(pasta, f"{base}*.hdr"))
    return candidatos[0] if candidatos else None

def mascara_hsv(caminho_tiff, shape_ref):
    """Gera máscara HSV a partir de imagem RGB"""
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
    """Verifica se dois ROIs se sobrepõem"""
    y0_1, y1_1, x0_1, x1_1 = r1
    y0_2, y1_2, x0_2, x1_2 = r2
    
    return not (x1_1 <= x0_2 or x1_2 <= x0_1 or y1_1 <= y0_2 or y1_2 <= y0_1)

def gerar_rois(mask, n_rois):
    """Gera ROIs não sobrepostos dentro da máscara"""
    coords = np.column_stack(np.where(mask))
    
    if len(coords) == 0:
        return [], 0
    
    area = len(coords)
    lado = max(int(np.sqrt(area / n_rois)), 5)
    
    rois = []
    tentativas = 0
    max_tentativas = 30000
    
    while len(rois) < n_rois and tentativas < max_tentativas:
        tentativas += 1
        y, x = coords[np.random.randint(len(coords))]
        
        y0 = max(0, y - lado // 2)
        y1 = min(mask.shape[0], y + lado // 2)
        x0 = max(0, x - lado // 2)
        x1 = min(mask.shape[1], x + lado // 2)
        
        if y1 - y0 < lado or x1 - x0 < lado:
            continue
        
        if not np.all(mask[y0:y1, x0:x1]):
            continue
        
        novo_roi = (y0, y1, x0, x1)
        
        if any(rois_sobrepostos(novo_roi, r) for r in rois):
            continue
        
        rois.append(novo_roi)
    
    return rois, lado

# ================= FUNÇÕES DE ÍNDICES =================

def calcular_ndvi(nir, red):
    """Calcula NDVI"""
    return (nir - red) / (nir + red + 1e-6)

def calcular_savi(nir, red, l=0.5):
    """Calcula SAVI (Soil-Adjusted Vegetation Index)"""
    return ((nir - red) / (nir + red + l)) * (1 + l)

def calcular_gndvi(nir, green):
    """Calcula GNDVI (Green NDVI)"""
    return (nir - green) / (nir + green + 1e-6)

def extrair_wavelengths(cols):
    """Extrai wavelengths dos nomes de coluna"""
    wls = []
    for c in cols:
        try:
            wls.append(float(c.replace('Band_', '').replace('d1_Band_', '').replace('nm', '')))
        except:
            pass
    return wls

def preprocessar_espectro(espectro, cols_bandas):
    """Aplica trimming, SNV e Savitzky-Golay"""
    # Trimming
    wls = extrair_wavelengths(cols_bandas)
    indices_validos = [i for i, wl in enumerate(wls) if TRIM_MIN <= wl <= TRIM_MAX]
    espectro_trim = espectro[indices_validos]
    
    # SNV
    media = np.mean(espectro_trim)
    std = np.std(espectro_trim)
    espectro_snv = (espectro_trim - media) / (std + 1e-8)
    
    # Savitzky-Golay (1ª derivada)
    espectro_sg = savgol_filter(espectro_snv, window_length=SG_WINDOW, 
                                polyorder=SG_POLY, deriv=SG_DERIV)
    
    return espectro_sg, indices_validos

# ================= PIPELINE PRINCIPAL =================

dados_csv = []
arquivos_bil = glob.glob(os.path.join(DATASET_DIR, "*.bil"))

print(f"Arquivos encontrados: {len(arquivos_bil)}\n")

for bil in arquivos_bil:
    nome = os.path.basename(bil).replace(".bil", "")
    print(f"Processando: {nome}")
    
    # Localiza arquivos associados
    hdr = encontrar_hdr(bil)
    if hdr is None:
        print("  ⚠️  HDR não encontrado, pulando.\n")
        continue
    
    tiff = os.path.join(DATASET_DIR, f"{nome}-RGB.tiff")
    
    # Carrega imagem hiperspectral
    img_obj = envi.open(hdr, bil)
    cube = img_obj.load()
    meta = img_obj.metadata
    
    # Encontra bandas espectrais
    idx_red = encontrar_banda(meta, WAVELENGTH_RED)
    idx_nir = encontrar_banda(meta, WAVELENGTH_NIR)
    
    # Calcula NDVI
    red = cube[:, :, idx_red].astype(float)
    nir = cube[:, :, idx_nir].astype(float)
    ndvi = calcular_ndvi(nir, red)
    mask_ndvi = (ndvi > NDVI_THRESHOLD)
    
    # Remove dimensão extra se existir
    if mask_ndvi.ndim == 3:
        mask_ndvi = mask_ndvi[:, :, 0]
    
    # Máscara HSV
    mask_hsv, img_rgb = mascara_hsv(tiff, mask_ndvi.shape)
    if mask_hsv is None:
        print("  ⚠️  RGB não encontrado, pulando.\n")
        continue
    
    # Máscara final (intersecção)
    mask_final = mask_ndvi & mask_hsv
    
    # Gera ROIs
    rois, lado = gerar_rois(mask_final, N_ROIS)
    print(f"  ROIs válidos: {len(rois)} | Lado do pixel: {lado}")
    
    if len(rois) == 0:
        print("  Nenhum ROI gerado, pulando.\n")
        continue
    
    # Extração espectral para cada ROI
    cols_bandas_raw = [f"Band_{float(wl):.2f}nm" for wl in meta['wavelength']]
    
    for roi_id, (y0, y1, x0, x1) in enumerate(rois):
        # Extrai bloco espectral
        bloco = cube[y0:y1, x0:x1, :]
        espectro_medio = bloco.reshape(-1, bloco.shape[2]).mean(axis=0)
        
        # Preprocessamento
        espectro_proc, indices_validos = preprocessar_espectro(espectro_medio, cols_bandas_raw)
        
        # Calcula índices para esse ponto
        red_roi = bloco[:, :, idx_red].mean()
        nir_roi = bloco[:, :, idx_nir].mean()
        ndvi_roi = calcular_ndvi(nir_roi, red_roi)
        savi_roi = calcular_savi(nir_roi, red_roi)
        
        # Monta linha do CSV
        linha = {
            "ID_Amostra": nome[:-1],  # Remove última letra
            "Parte": nome[-1],         # Última letra (M ou S)
            "ROI_ID": roi_id,
            "Lado_Pixel": lado,
            "NDVI": ndvi_roi,
            "SAVI": savi_roi
        }
        
        # Adiciona bandas brutas
        for i, wl in enumerate(meta['wavelength']):
            linha[f"Band_{float(wl):.2f}nm"] = espectro_medio[i]
        
        # Adiciona bandas processadas (1ª derivada)
        cols_processadas = [cols_bandas_raw[i] for i in indices_validos]
        for i, idx in enumerate(indices_validos):
            linha[f"d1_{cols_bandas_raw[idx]}"] = espectro_proc[i]
        
        dados_csv.append(linha)
    
    print(f"  ✓ {len(rois)} ROIs processados\n")

# ================= SALVAR RESULTADOS =================

if dados_csv:
    df = pd.DataFrame(dados_csv)
    
    # Reordena colunas
    cols_meta = ["ID_Amostra", "Parte", "ROI_ID", "Lado_Pixel", "NDVI", "SAVI"]
    cols_bandas = [c for c in df.columns if c.startswith("Band_")]
    cols_derivadas = [c for c in df.columns if c.startswith("d1_")]
    
    df_final = df[cols_meta + cols_bandas + cols_derivadas]
    
    df_final.to_csv("spectral_indices_rois2.csv", index=False, sep=";", decimal=",")
    
    print("✅ PROCESSAMENTO FINALIZADO")
    print(f"Total de ROIs extraídos: {len(df_final)}")
    print(f"Arquivo salvo: spectral_indices_rois.csv")
    print(f"Colunas: {len(df_final.columns)}")
else:
    print("❌ ERRO: Nenhum dado foi extraído.")