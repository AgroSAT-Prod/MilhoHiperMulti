import os
import glob
import cv2
import numpy as np
import pandas as pd
import spectral.io.envi as envi
import matplotlib.pyplot as plt

# ================= CONFIG =================
N_ROIS = 1000
NDVI_THRESHOLD = 0.25

WAVELENGTH_RED = 670.0
WAVELENGTH_NIR = 800.0

LOWER_GREEN = np.array([40, 40, 40])
UPPER_GREEN = np.array([85, 255, 255])

DATASET_DIR = "Dataset"

# ================= FUNÇÕES =================

def encontrar_banda(header, alvo):
    wls = np.array([float(w) for w in header['wavelength']])
    return np.abs(wls - alvo).argmin()


def encontrar_hdr(caminho_bil):
    pasta = os.path.dirname(caminho_bil)
    base = os.path.basename(caminho_bil).replace(".bil", "")
    candidatos = glob.glob(os.path.join(pasta, f"{base}*.hdr"))
    return candidatos[0] if len(candidatos) > 0 else None


def mascara_hsv(caminho_tiff, shape_ref):
    if not os.path.exists(caminho_tiff):
        return None, None

    img = cv2.imread(caminho_tiff)
    if img is None:
        return None, None

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)

    mask = cv2.resize(
        mask,
        (shape_ref[1], shape_ref[0]),
        interpolation=cv2.INTER_NEAREST
    )

    return mask.astype(bool), img


def rois_sobrepostos(r1, r2):
    y0_1, y1_1, x0_1, x1_1 = r1
    y0_2, y1_2, x0_2, x1_2 = r2

    if x1_1 <= x0_2 or x1_2 <= x0_1:
        return False
    if y1_1 <= y0_2 or y1_2 <= y0_1:
        return False

    return True


def gerar_rois(mask, n_rois):
    coords = np.column_stack(np.where(mask))
    area = len(coords)

    if area == 0:
        return [], 0

    lado = max(int(np.sqrt(area / n_rois)), 5)

    rois = []
    tentativas = 0

    while len(rois) < n_rois and tentativas < 30000:
        tentativas += 1

        y, x = coords[np.random.randint(len(coords))]

        y0 = y - lado // 2
        y1 = y + lado // 2
        x0 = x - lado // 2
        x1 = x + lado // 2

        if y0 < 0 or x0 < 0 or y1 >= mask.shape[0] or x1 >= mask.shape[1]:
            continue

        if not np.all(mask[y0:y1, x0:x1]):
            continue

        novo_roi = (y0, y1, x0, x1)

        if any(rois_sobrepostos(novo_roi, r) for r in rois):
            continue

        rois.append(novo_roi)

    return rois, lado

# ================= PIPELINE =================

dados_csv = []

arquivos_bil = glob.glob(os.path.join(DATASET_DIR, "*.bil"))
print(f"Arquivos encontrados: {len(arquivos_bil)}")

for bil in arquivos_bil:
    nome = os.path.basename(bil).replace(".bil", "")
    print(f"\nProcessando {nome}")

    hdr = encontrar_hdr(bil)
    if hdr is None:
        print("⚠️ HDR não encontrado, pulando.")
        continue

    tiff = os.path.join(DATASET_DIR, f"{nome}-RGB.tiff")

    img = envi.open(hdr, bil)
    cube = img.load()
    meta = img.metadata

    idx_red = encontrar_banda(meta, WAVELENGTH_RED)
    idx_nir = encontrar_banda(meta, WAVELENGTH_NIR)

    red = cube[:, :, idx_red].astype(float)
    nir = cube[:, :, idx_nir].astype(float)

    ndvi = (nir - red) / (nir + red + 1e-6)
    mask_ndvi = (ndvi > NDVI_THRESHOLD)
    if mask_ndvi.ndim == 3:
        mask_ndvi = mask_ndvi[:, :, 0]


    mask_hsv, img_rgb = mascara_hsv(tiff, mask_ndvi.shape)
    if mask_hsv is None:
        print("⚠️ RGB não encontrado, pulando.")
        continue

    mask_final = mask_ndvi & mask_hsv

    rois, lado = gerar_rois(mask_final, N_ROIS)
    print(f"ROIs válidos: {len(rois)} | Lado: {lado}")

    if len(rois) == 0:
        continue

    # ===== VISUALIZAÇÃO (opcional) =====
    """
    vis = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    vis = cv2.resize(vis, (mask_final.shape[1], mask_final.shape[0]))

    for y0, y1, x0, x1 in rois:
        cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 0, 0), 1)

    plt.figure(figsize=(6, 6))
    plt.title(f"{nome} | ROIs={len(rois)} | Lado={lado}")
    plt.imshow(vis)
    plt.axis("off")
    plt.show()
    """

    # ===== EXTRAÇÃO ESPECTRAL =====
    for roi_id, (y0, y1, x0, x1) in enumerate(rois):
        bloco = cube[y0:y1, x0:x1, :]
        espectro = bloco.reshape(-1, bloco.shape[2]).mean(axis=0)

        linha = {
            "ID_Amostra": nome[:-1],
            "Parte": nome[-1],
            "ROI_ID": roi_id,
            "Lado_ROI": lado
        }

        for wl, val in zip(meta['wavelength'], espectro):
            linha[f"Band_{float(wl):.2f}nm"] = val

        dados_csv.append(linha)

# ================= SALVAR =================

df = pd.DataFrame(dados_csv)
df.to_csv("hiperespectral_ROIs.csv", index=False, sep=";", decimal=",")

print("\n✅ Workflow finalizado — ROIs sem sobreposição, NDVI + HSV.")
