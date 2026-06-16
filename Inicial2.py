import os
import glob
import numpy as np
import pandas as pd
import spectral.io.envi as envi
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ================= CONFIGURAÇÕES =================
PASTA_DO_SCRIPT = os.path.dirname(os.path.abspath(__file__))
DIRETORIO_DADOS = os.path.join(PASTA_DO_SCRIPT, 'Dataset')

# Configurações de Banda
WAVELENGTH_RED = 670.0
WAVELENGTH_NIR = 800.0
NDVI_THRESHOLD_INICIAL = 0.25

# ================= FUNÇÕES =================
def encontrar_banda_mais_proxima(header, alvo_wl):
    try:
        wls = [float(w) for w in header['wavelength']]
        wls_arr = np.array(wls)
        idx = (np.abs(wls_arr - alvo_wl)).argmin()
        return idx, wls_arr[idx]
    except KeyError:
        return None, None

def plotar_ndvi_com_threshold(arquivo_bil, threshold_inicial=0.25):
    """Visualiza NDVI, máscara e imagem RGB original com slider para ajustar threshold."""
    
    path_bil = arquivo_bil
    path_hdr = arquivo_bil + ".hdr"
    nome_amostra = os.path.basename(arquivo_bil).replace('.bil', '')

    if not os.path.exists(path_hdr):
        print(f"HDR não encontrado para: {nome_amostra}")
        return

    try:
        img_obj = envi.open(path_hdr, path_bil)
        dados = img_obj.load()
        metadata = img_obj.metadata
        
        idx_red, wl_red = encontrar_banda_mais_proxima(metadata, WAVELENGTH_RED)
        idx_nir, wl_nir = encontrar_banda_mais_proxima(metadata, WAVELENGTH_NIR)
        
        print(f"Banda RED encontrada: {wl_red:.2f}nm (índice {idx_red})")
        print(f"Banda NIR encontrada: {wl_nir:.2f}nm (índice {idx_nir})")
        
        # Calcular NDVI base
        banda_red = dados[:, :, idx_red].astype(float)
        banda_nir = dados[:, :, idx_nir].astype(float)
        
        denominador = (banda_nir + banda_red)
        denominador[denominador == 0] = 0.00001
        ndvi = (banda_nir - banda_red) / denominador
        
        # Para visualização RGB, usar 3 bandas
        idx_r, _ = encontrar_banda_mais_proxima(metadata, 670.0)  # Red
        idx_g, _ = encontrar_banda_mais_proxima(metadata, 550.0)  # Green
        idx_b, _ = encontrar_banda_mais_proxima(metadata, 460.0)  # Blue
        
        rgb = np.dstack([
            dados[:, :, idx_r].astype(float),
            dados[:, :, idx_g].astype(float),
            dados[:, :, idx_b].astype(float)
        ])
        
        # Normalizar RGB para visualização
        rgb_norm = np.zeros_like(rgb)
        for i in range(3):
            rgb_min = rgb[:, :, i].min()
            rgb_max = rgb[:, :, i].max()
            if rgb_max > rgb_min:
                rgb_norm[:, :, i] = (rgb[:, :, i] - rgb_min) / (rgb_max - rgb_min)
        
        # Criar figura com subplots
        fig = plt.figure(figsize=(18, 12))
        
        # Subplot 1: RGB original
        ax1 = plt.subplot(2, 3, 1)
        im1 = ax1.imshow(rgb_norm)
        ax1.set_title(f"Imagem RGB Original\n{nome_amostra}", fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # Subplot 2: NDVI (mapa contínuo)
        ax2 = plt.subplot(2, 3, 2)
        im2 = ax2.imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
        ax2.set_title(f"Índice NDVI\nWavelengths - Red: {wl_red:.2f}nm, NIR: {wl_nir:.2f}nm", 
                      fontsize=12, fontweight='bold')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, label='NDVI Value')
        
        # Subplot 3: Máscara binária (ajustável via slider)
        ax3 = plt.subplot(2, 3, 3)
        
        # Criar slider para threshold
        ax_slider = plt.axes([0.25, 0.05, 0.5, 0.03])
        slider_threshold = Slider(
            ax_slider, 'NDVI Threshold', 
            -1.0, 1.0, 
            valinit=threshold_inicial, 
            valstep=0.01,
            color='steelblue'
        )
        
        # Função para atualizar visualização com novo threshold
        def atualizar_mascara(val):
            threshold = slider_threshold.val
            mascara = ndvi > threshold
            
            ax3.clear()
            im3 = ax3.imshow(mascara, cmap='RdYlGn', vmin=0, vmax=1)
            pixels_planta = np.sum(mascara)
            pixels_total = ndvi.shape[0] * ndvi.shape[1]
            percentual = (pixels_planta / pixels_total) * 100
            
            ax3.set_title(f"Máscara Binária (Threshold = {threshold:.3f})\n"
                         f"Pixels de Planta: {pixels_planta} ({percentual:.2f}%)", 
                         fontsize=12, fontweight='bold')
            ax3.axis('off')
            
            # Subplot 4: Imagem mascarada
            ax4.clear()
            imagem_mascarada = rgb_norm.copy()
            # Aplicar máscara corretamente
            for i in range(3):
                imagem_mascarada[:, :, i] = imagem_mascarada[:, :, i] * mascara.astype(float)
            ax4.imshow(imagem_mascarada)
            ax4.set_title(f"RGB Mascarado\n(Apenas pixels com NDVI > {threshold:.3f})", 
                         fontsize=12, fontweight='bold')
            ax4.axis('off')
            
            # Subplot 5: Histograma NDVI
            ax5.clear()
            ax5.hist(ndvi.flatten(), bins=100, color='steelblue', alpha=0.7, edgecolor='black')
            ax5.axvline(threshold, color='red', linestyle='--', linewidth=2.5, label=f'Threshold = {threshold:.3f}')
            ax5.set_xlabel('Valor de NDVI', fontsize=11, fontweight='bold')
            ax5.set_ylabel('Frequência', fontsize=11, fontweight='bold')
            ax5.set_title('Distribuição de NDVI', fontsize=12, fontweight='bold')
            ax5.legend(fontsize=10)
            ax5.grid(True, alpha=0.3)
            
            # Subplot 6: Estatísticas
            ax6.clear()
            ax6.axis('off')
            
            ndvi_pixels_planta = ndvi[mascara]
            stats_text = f"""
ESTATÍSTICAS DO NDVI

Threshold Atual: {threshold:.3f}

GERAL:
  Min: {ndvi.min():.4f}
  Max: {ndvi.max():.4f}
  Mean: {ndvi.mean():.4f}
  Std: {ndvi.std():.4f}

PIXELS DE PLANTA (NDVI > {threshold:.3f}):
  Total: {pixels_planta} / {pixels_total}
  Percentual: {percentual:.2f}%
  Min NDVI: {ndvi_pixels_planta.min():.4f}
  Max NDVI: {ndvi_pixels_planta.max():.4f}
  Mean NDVI: {ndvi_pixels_planta.mean():.4f}
  Std NDVI: {ndvi_pixels_planta.std():.4f}
            """
            ax6.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                    verticalalignment='center', bbox=dict(boxstyle='round', 
                    facecolor='wheat', alpha=0.5))
            
            fig.canvas.draw_idle()
        
        # Criar subplots vazios para serem preenchidos
        ax4 = plt.subplot(2, 3, 4)
        ax5 = plt.subplot(2, 3, 5)
        ax6 = plt.subplot(2, 3, 6)
        
        # Atualizar inicialmente
        atualizar_mascara(threshold_inicial)
        
        # Conectar slider ao callback
        slider_threshold.on_changed(atualizar_mascara)
        
        plt.suptitle(f"Análise de NDVI e Threshold - {nome_amostra}", 
                     fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0.1, 1, 0.96])
        plt.show()
        
    except Exception as e:
        print(f"ERRO ao processar {nome_amostra}: {e}")
        import traceback
        traceback.print_exc()

# ================= EXECUÇÃO =================
print(f"Buscando arquivos .bil em: {DIRETORIO_DADOS}")

# Localiza o PRIMEIRO arquivo .bil na pasta Dataset para análise
arquivos = glob.glob(os.path.join(DIRETORIO_DADOS, "*.bil"))

if arquivos:
    print(f"Total de arquivos encontrados: {len(arquivos)}")
    print(f"\nAnalisando primeiro arquivo: {os.path.basename(arquivos[0])}")
    print("\nDica: Use o slider para ajustar o threshold de NDVI e visualizar o efeito!")
    print("Observar se a máscara está capturando bem as folhas (sem muito solo/sombra)")
    
    plotar_ndvi_com_threshold(arquivos[0], NDVI_THRESHOLD_INICIAL)
else:
    print("Nenhum arquivo .bil encontrado!")