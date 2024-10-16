import cv2
import numpy as np
from gri import convert_to_grayscale
def prewitt_filter(image):
    # Prewitt kenar tespit filtresi
    kernel_x = np.array([[-1, 0, 1],
                         [-1, 0, 1],
                         [-1, 0, 1]])

    kernel_y = np.array([[-1, -1, -1],
                         [0, 0, 0],
                         [1, 1, 1]])


    # Kenarları tespit etmek için filtreleri uygula
    prewitt_x = cv2.filter2D(image, -1, kernel_x)
    prewitt_y = cv2.filter2D(image, -1, kernel_y)

    # Kenar görüntüsünü birleştir
    edges = cv2.bitwise_or(prewitt_x, prewitt_y)

    return edges
#
# # Görüntüyü yükle
# image = convert_to_grayscale(image_path='./dataset/train/train/skin_cancer/skin_cancer_10.jpg')
#
#
#
# # Prewitt filtresini uygula
# edges = prewitt_filter(image)
# # Renkleri tersine çevir
# edges = 255 - edges
#
# # Görüntüyü göster
# cv2.imshow('Original Image', image)
# cv2.imshow('Prewitt Filter', edges)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
