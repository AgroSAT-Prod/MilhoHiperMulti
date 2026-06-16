import os
import glob
import cv2
import numpy as np
import pandas as pd
import spectral.io.envi as envi
from scipy.signal import savgol_filter

# ================= CONFIGURAÇÕES =================
DATASET_DIR = "Dataset"
ARQUIVO_AGRONOMICO = os.path.join('Dataset', 'PlanilhaFiltrada.xlsx')

# Meta final por classe (balanceamento rígido)
META_ROIS_POR_CLASSE = 300  

# LIMITES DE TAMANHO DO ROI (A CORREÇÃO PRINCIPAL)
MAX_LADO_PIXEL = 65    # Nunca cria quadrados maiores que isso (evita erro na Dose 180)
MIN_LADO_PIXEL = 5     # Tamanho mínimo aceitável

# Bandas de interesse (nm)
WAVELENGTH_RED = 670.0
WAVELENGTH_NIR = 720.0
NDVI_THRESHOLD = 0.25

# Máscara HSV para verde
LOWER_GREEN = np.array([40, 40, 40])
UPPER_GREEN = np.array([85, 255, 255])

# Processamento espectral
TRIM_MIN = 420.0
TRIM_MAX = 740.0
SG_WINDOW = 11
SG_POLY = 2
SG_DERIV = 1

# ================= FUNÇÕES DE SUPORTE =================

def encontrar_banda(header, alvo):
    wls = np.array([float(w) for w in header['wavelength']])
    return np.argmin(np.abs(wls - alvo))

def encontrar_hdr(caminho_bil):
    pasta = os.path.dirname(caminho_bil)
    base = os.path.basename(caminho_bil).replace(".bil", "")
    candidatos = glob.glob(os.path.join(pasta, f"{base}*.hdr"))
    return candidatos[0] if candidatos else None

def carregar_doses_reais():
    try:
        df_agro = pd.read_excel(ARQUIVO_AGRONOMICO, header=2)
        df_agro.columns = [str(c).strip().lower() for c in df_agro.columns]
        col_id = next((c for c in df_agro.columns if 'id' in c and 'parcela' in c), 'id parcela')
        col_dose = next((c for c in df_agro.columns if 'dose' in c), None)
        df_agro = df_agro.dropna(subset=[col_id, col_dose])
        return dict(zip(df_agro[col_id].astype(int), df_agro[col_dose]))
    except Exception as e:
        print(f"⚠️  Erro Excel: {e}")
        return {}

def gerar_mascara_valida(tiff_path, header_envi, bil_path):
    if not os.path.exists(tiff_path): return None, None
    img_rgb = cv2.imread(tiff_path)
    if img_rgb is None: return None, None
    
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)
    mask_hsv = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)

    try:
        img_obj = envi.open(header_envi, bil_path)
        meta = img_obj.metadata
        idx_red = encontrar_banda(meta, WAVELENGTH_RED)
        idx_nir = encontrar_banda(meta, WAVELENGTH_NIR)
        
        cube = img_obj.open_memmap(interleave='bip')
        red = cube[:, :, idx_red].astype(float)
        nir = cube[:, :, idx_nir].astype(float)
        
        ndvi = (nir - red) / (nir + red + 1e-6)
        mask_ndvi = (ndvi > NDVI_THRESHOLD)
        
        if mask_hsv.shape[:2] != mask_ndvi.shape[:2]:
            mask_hsv = cv2.resize(mask_hsv, (mask_ndvi.shape[1], mask_ndvi.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        mask_final = mask_ndvi & (mask_hsv > 0)
        return mask_final, img_rgb
    except:
        return None, None

def rois_sobrepostos(r1, r2):
    y0_1, y1_1, x0_1, x1_1 = r1
    y0_2, y1_2, x0_2, x1_2 = r2
    return not (x1_1 <= x0_2 or x1_2 <= x0_1 or y1_1 <= y0_2 or y1_2 <= y0_1)

def tentar_extrair(mask, n_rois, lado):
    """Tenta extrair N rois com tamanho fixo LADO"""
    coords = np.column_stack(np.where(mask))
    if len(coords) == 0: return []
    
    rois = []
    tentativas = 0
    max_tentativas = n_rois * 100 # Aumentei as tentativas por ciclo
    
    while len(rois) < n_rois and tentativas < max_tentativas:
        tentativas += 1
        idx = np.random.randint(len(coords))
        y, x = coords[idx]
        
        y0, y1 = max(0, y - lado // 2), min(mask.shape[0], y + lado // 2)
        x0, x1 = max(0, x - lado // 2), min(mask.shape[1], x + lado // 2)
        
        if (y1 - y0) < lado or (x1 - x0) < lado: continue
        
        # Checagem rápida: 4 cantos e centro devem ser True (otimização)
        if not (mask[y0, x0] and mask[y1-1, x1-1] and mask[y0, x1-1] and mask[y1-1, x0] and mask[y, x]):
            continue
            
        if not np.all(mask[y0:y1, x0:x1]): continue
        
        novo_roi = (y0, y1, x0, x1)
        if any(rois_sobrepostos(novo_roi, r) for r in rois): continue
        
        rois.append(novo_roi)
    
    return rois

def gerar_rois_recursivo(mask, n_target):
    """
    Estratégia 'Shrink-to-Fit':
    Tenta pegar ROIs grandes. Se não conseguir todos, diminui o tamanho
    e tenta pegar os restantes nas áreas menores.
    """
    coords = np.column_stack(np.where(mask))
    if len(coords) < 100: return [], 0
    
    area_total = len(coords)
    
    # Cálculo inicial agressivo
    lado_ideal = int(np.sqrt((area_total * 0.5) / n_target))
    
    # CLAMP: Limita o tamanho máximo para evitar o problema da Dose 180
    lado_atual = min(lado_ideal, MAX_LADO_PIXEL)
    lado_atual = max(lado_atual, MIN_LADO_PIXEL)
    
    rois_coletados = []
    
    # Loop de tentativas com redução de tamanho
    while len(rois_coletados) < n_target and lado_atual >= MIN_LADO_PIXEL:
        faltam = n_target - len(rois_coletados)
        
        # Tenta extrair o que falta com o tamanho atual
        # Passamos uma cópia da máscara para garantir isolamento se quiséssemos "apagar" os usados
        # mas aqui vamos apenas verificar sobreposição
        novos_rois = tentar_extrair(mask, faltam, lado_atual)
        
        # Adiciona os novos que não sobrepõem os JÁ coletados
        for novo in novos_rois:
            if not any(rois_sobrepostos(novo, r) for r in rois_coletados):
                rois_coletados.append(novo)
                
        if len(rois_coletados) >= n_target:
            break
            
        # Se não conseguiu tudo, reduz o tamanho e tenta de novo
        lado_anterior = lado_atual
        lado_atual = int(lado_atual * 0.75) # Reduz 25%
        
        if lado_atual == lado_anterior: # Evita loop infinito se int() não mudar
            lado_atual -= 1
            
    return rois_coletados, lado_ideal # Retorna lado inicial como referência

def preprocessar_espectro(espectro, wls_list):
    indices_validos = [i for i, wl in enumerate(wls_list) if TRIM_MIN <= wl <= TRIM_MAX]
    espectro_trim = espectro[indices_validos]
    media = np.mean(espectro_trim)
    std = np.std(espectro_trim)
    espectro_snv = (espectro_trim - media) / (std + 1e-8)
    try:
        espectro_sg = savgol_filter(espectro_snv, window_length=SG_WINDOW, polyorder=SG_POLY, deriv=SG_DERIV)
    except:
        espectro_sg = espectro_snv
    return espectro_sg, indices_validos

# ================= PIPELINE PRINCIPAL =================

print("="*70)
print(" 1. INVENTÁRIO DO DATASET")
print("="*70)

doses_dict = carregar_doses_reais()
arquivos_bil = glob.glob(os.path.join(DATASET_DIR, "*.bil"))
inventario = []

for bil in arquivos_bil:
    nome = os.path.basename(bil).replace(".bil", "")
    hdr = encontrar_hdr(bil)
    tiff = os.path.join(DATASET_DIR, f"{nome}-RGB.tiff")
    
    try:
        id_amostra = int(nome[:-1])
        dose = doses_dict.get(id_amostra, 0)
        dose = round(dose, 0)
    except:
        continue

    if hdr and os.path.exists(tiff):
        mask, _ = gerar_mascara_valida(tiff, hdr, bil)
        if mask is not None:
            area_pixels = np.sum(mask)
            if area_pixels > 200:
                inventario.append({
                    'bil': bil, 'hdr': hdr, 'tiff': tiff,
                    'nome': nome, 'id_amostra': id_amostra,
                    'dose': dose, 'area': area_pixels
                })

df_inv = pd.DataFrame(inventario)
print(f"Arquivos válidos: {len(df_inv)}")

# ================= ALOCAÇÃO =================

print("\n" + "="*70)
print(" 2. ALOCAÇÃO DE ROIs (Teto Máximo)")
print("="*70)

dados_processamento = []
grupos = df_inv.groupby('dose')

for dose, grupo in grupos:
    area_total_dose = grupo['area'].sum()
    print(f"Dose {int(dose)}: {len(grupo)} arquivos | Área Total: {area_total_dose}")
    
    for _, row in grupo.iterrows():
        fator = row['area'] / area_total_dose
        rois_alvo = int(META_ROIS_POR_CLASSE * fator)
        
        # Garante pelo menos 1 ROI se a imagem for decente, para não perder variabilidade
        if rois_alvo == 0 and row['area'] > 5000:
            rois_alvo = 1
            
        if rois_alvo > 0:
            row_dict = row.to_dict()
            row_dict['rois_target'] = rois_alvo
            dados_processamento.append(row_dict)

# ================= EXTRAÇÃO =================

print("\n" + "="*70)
print(" 3. EXTRAÇÃO RECURSIVA (ROBUSTA)")
print("="*70)

dados_csv = []
rois_extraidos_real = {}

for item in dados_processamento:
    nome = item['nome']
    rois_target = item['rois_target']
    dose = item['dose']
    
    print(f"-> {nome} (Dose {int(dose)}) | Alvo: {rois_target} ROIs")
    
    mask, img_rgb = gerar_mascara_valida(item['tiff'], item['hdr'], item['bil'])
    
    # A MÁGICA ACONTECE AQUI:
    rois, lado_tentativa = gerar_rois_recursivo(mask, rois_target)
    
    sucesso = len(rois)
    rois_extraidos_real[dose] = rois_extraidos_real.get(dose, 0) + sucesso
    
    if sucesso < rois_target:
        print(f"   ⚠️  Parcial: {sucesso}/{rois_target} (Tentou lado {lado_tentativa}px, reduziu para encaixar)")
    else:
        print(f"   ✓  Sucesso: {sucesso}/{rois_target}")

    # Extração de dados
    if sucesso > 0:
        img_obj = envi.open(item['hdr'], item['bil'])
        cube = img_obj.load()
        meta = img_obj.metadata
        idx_red = encontrar_banda(meta, WAVELENGTH_RED)
        idx_nir = encontrar_banda(meta, WAVELENGTH_NIR)
        wls = [float(w) for w in meta['wavelength']]
        col_names_raw = [f"Band_{w:.2f}nm" for w in wls]
        
        for roi_id, (y0, y1, x0, x1) in enumerate(rois):
            lado_real = y1 - y0 # O lado real pode variar agora!
            bloco = cube[y0:y1, x0:x1, :]
            espectro_medio = bloco.reshape(-1, bloco.shape[2]).mean(axis=0)
            espectro_proc, idxs_validos = preprocessar_espectro(espectro_medio, wls)
            
            red_val = bloco[:, :, idx_red].mean()
            nir_val = bloco[:, :, idx_nir].mean()
            ndvi = (nir_val - red_val) / (nir_val + red_val + 1e-6)
            savi = ((nir_val - red_val) / (nir_val + red_val + 0.5)) * 1.5
            
            linha = {
                "ID_Amostra": item['id_amostra'], "Parte": nome[-1],
                "Dose_N": dose, "ROI_ID": roi_id,
                "Lado_Pixel": lado_real, "NDVI": ndvi, "SAVI": savi
            }
            
            for i, val in enumerate(espectro_medio): linha[col_names_raw[i]] = val
            col_names_proc = [col_names_raw[i] for i in idxs_validos]
            for i, val in enumerate(espectro_proc): linha[f"d1_{col_names_proc[i]}"] = val
                
            dados_csv.append(linha)

# ================= FINALIZAÇÃO =================

if dados_csv:
    df_final = pd.DataFrame(dados_csv)
    cols_meta = ["ID_Amostra", "Parte", "Dose_N", "ROI_ID", "Lado_Pixel", "NDVI", "SAVI"]
    cols_rest = [c for c in df_final.columns if c not in cols_meta]
    df_final = df_final[cols_meta + cols_rest]
    
    df_final.to_csv("spectral_indices_rois_final_v3.csv", index=False, sep=";", decimal=",")
    
    print("\nRELATÓRIO DE BALANCEAMENTO:")
    for dose in sorted(rois_extraidos_real.keys()):
        print(f"Dose {int(dose):3d}: {rois_extraidos_real[dose]:4d} ROIs")
    print("\n✅ Concluído com sucesso.")
else:
    print("❌ Falha total.")