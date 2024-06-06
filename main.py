import cv2
import matplotlib.pyplot as plt
import numpy as np

def image_show(image, window_name):
    """Verilen görüntüyü belirtilen pencere adıyla ekranda gösterir."""
    cv2.imshow(window_name, image)

def gri(image):
    """Görüntü üzerinde bir işlem yap ve yeni bir görüntü döndür."""
    # Örnek işlem: Görüntüyü gri tonlamaya çevir
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def iso(image):

    _, isodata_goruntu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return isodata_goruntu

def apply_median_filter(image, kernel_size):

    filtered_image = cv2.medianBlur(image, kernel_size)
    return filtered_image


def find_contours(image):
    """Görüntüdeki şekillerin sınırlarını bulur ve koordinatlarını döndürür."""
    # Görüntüyü gri tonlamaya çevirme (gerekirse)
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Görüntüyü ikili hale getirme
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # Kenarları bulma
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours

def draw_contours(image, contours):
    """Görüntüdeki şekillerin sınırlarını çiz ve ekranda göster."""
    # Konturların üzerine çizgi çekme
    cv2.drawContours(image, contours, -1, (0, 255, 0), 5)




if __name__ == '__main__':
    resim = cv2.imread("./dataset/train/train/skin_cancer/skin_cancer_07.jpg")
    resim = cv2.resize(resim, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    resim1 = gri(resim)
    resim2 = iso(resim1)
    resim3 = apply_median_filter(resim2, 33)
    contours = find_contours(resim3)
    for i, contour in enumerate(contours):
        print(f"Shape {i + 1}: {contour}")
    resim4 = cv2.cvtColor(resim3, cv2.COLOR_GRAY2BGR)
    draw_contours(resim, contours)

    # Görüntüleri aynı anda açma
    images = [resim, resim1, resim2, resim3, resim4]
    window_names = ["ILK HALI", "GRI HALI", "ISO HALI", "MEDIAN HALI", "AUTOCANNY"]

    for window_name, img in zip(window_names, images):
        image_show(img, window_name)




    cv2.waitKey(0)
    cv2.destroyAllWindows()

