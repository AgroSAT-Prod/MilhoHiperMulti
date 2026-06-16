import cv2
import numpy as np
import matplotlib.pyplot as plt

def criar_mascara_tiff(caminho_arquivo):
    # 1. Carregar a imagem TIFF
    # O OpenCV carrega imagens em formato BGR (Blue, Green, Red) por padrão
    img_bgr = cv2.imread(caminho_arquivo)

    if img_bgr is None:
        print("Erro: Não foi possível carregar a imagem. Verifique o caminho.")
        return

    # 2. Converter para HSV (Hue, Saturation, Value)
    # HSV é melhor para separar cores do que RGB, pois separa a "cor" (H) da "luminosidade" (V)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # 3. Definir o Threshold (Intervalo de Cor)
    # Exemplo: Filtrar a cor VERDE (comum em vegetação)
    # No OpenCV, o Hue vai de 0 a 179.
    # Verdes geralmente estão entre 35 e 85.
    
    # [Hue, Saturation, Value]
    lower_green = np.array([40, 40, 40])   # Limite inferior
    upper_green = np.array([85, 255, 255]) # Limite superior

    # 4. Criar a Máscara
    # A função inRange cria uma imagem binária: 255 (branco) onde a condição é verdadeira, 0 (preto) onde não é.
    mask = cv2.inRange(img_hsv, lower_green, upper_green)

    # 5. (Opcional) Aplicar a máscara na imagem original para ver o resultado recortado
    result = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)

    # 6. Visualização
    # Converter BGR para RGB para o Matplotlib exibir as cores corretamente
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(15, 5))

    # Imagem Original
    plt.subplot(1, 3, 1)
    plt.title("Imagem Original (TIFF)")
    plt.imshow(img_rgb)
    plt.axis('off')

    # Máscara (Preto e Branco)
    plt.subplot(1, 3, 2)
    plt.title("Máscara (Threshold)")
    plt.imshow(mask, cmap='gray')
    plt.axis('off')

    # Resultado (Apenas a área filtrada)
    plt.subplot(1, 3, 3)
    plt.title("Resultado com Máscara")
    plt.imshow(result_rgb)
    plt.axis('off')

    plt.show()

# --- Execução ---
# Substitua pelo caminho do seu arquivo
# Note o 'r' antes das aspas
caminho = r"C:\Users\muril\OneDrive\Documentos\Experiments-Prod\ProjetoMilho\Dataset\1M-RGB.tiff"
# E lembre-se de descomentar a linha abaixo para rodar a função:
criar_mascara_tiff(caminho)