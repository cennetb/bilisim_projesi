import cv2
import numpy as np
from scipy.signal import wiener
import matplotlib.pyplot as plt
def image_show(image, window_name):
    """Verilen görüntüyü belirtilen pencere adıyla ekranda gösterir."""
    cv2.imshow(window_name, image)
def add_gaussian_noise(image, mean=0, std_dev=25):
    height, width = image.shape
    gauss_noise = np.random.normal(mean, std_dev, (height, width))
    noisy_image = image + gauss_noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

    return noisy_image
def sobel_edge_detection(image):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # X yönünde Sobel
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Y yönünde Sobel
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)  # Sobel büyüklüğünü hesapla
    return np.clip(sobel_combined, 0, 255).astype(np.uint8)

def morphological_opening(image, kernel_size=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))  # Dikdörtgen çekirdek
    opened_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return opened_image
def custom_convolution(image, kernel):
    """Manuel konvolüsyon işlemi."""
    kernel_height, kernel_width = kernel.shape
    pad_h, pad_w = kernel_height // 2, kernel_width // 2
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    output_image = np.zeros_like(image, dtype=np.float32)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded_image[i:i + kernel_height, j:j + kernel_width]
            output_image[i, j] = np.sum(region * kernel)

    return np.clip(output_image, 0, 255).astype(np.uint8)
def sharpen_filter(image):
    """Keskinleştirme filtresi."""
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]], dtype=np.float32)
    return custom_convolution(image, kernel)
def edge_detection_filter(image):
    """Kenar bulma filtresi."""
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]], dtype=np.float32)
    return custom_convolution(image, kernel)

if __name__ == '__main__':
    image_path = """./images.jpeg"""
    image = cv2.imread(image_path)
    #gri tonlama
    image1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #yeniden boyutlandırma
    image2 = cv2.resize(image, (256, 256))
    #gürültü ekleme
    image3 = add_gaussian_noise(image1)
    #gürültü temizleme
    image4 = cv2.blur(image3, (3, 3))
    # gürültü temizleme
    image5 = wiener(image1, (5, 5))
    image5 = np.clip(image5, 0, 255).astype(np.uint8)
    #kenar bulma
    image6 = sobel_edge_detection(image)
    #otsu siyah beyaz
    _, image7 = cv2.threshold(image1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #açma
    image8 = morphological_opening(image7)
    #konvolüsyon
    # Keskinleştirme
    image9 = sharpen_filter(image1)
    # Kenar bulma
    image10 = edge_detection_filter(image9)

    cv2.imshow("Orijinal", image)
    cv2.imshow("Gri tonlamalı", image1)
    cv2.imshow("Yeniden boyutlandirilmis", image2)
    cv2.imshow("Gurultuf eklnemis", image3)
    cv2.imshow("Gurultu temizlenmis", image4)
    cv2.imshow("Gurultu temizlenmis2", image5)
    cv2.imshow("Kenar bulunmus", image6)
    cv2.imshow("Siyah beyaza çevirilmis", image7)
    cv2.imshow("Acma islemi yapilmis", image8)
    cv2.imshow("kesknlestirme", image9)
    cv2.imshow("kenar bulma", image7)


    # Pencerelerin ekranda kalması için bir tuşa basılmasını bekle
    cv2.waitKey(0)  # 0 sonsuza kadar bekler
    cv2.destroyAllWindows()  # Tüm pencereleri kapatır
