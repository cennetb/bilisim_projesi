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


import cv2
import numpy as np


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


def draw_circles(image, contours):
    """Görüntüdeki şekillerin etrafına minimum çemberler çizer ve çaplarını hesaplar."""
    circle_diameters = []

    for contour in contours:
        # Minimum çevreleyen çemberi bulma
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)

        # Çemberi çizme
        cv2.circle(image, center, radius, (255, 0, 0), 2)

        # Çapı hesaplayıp listeye ekleme
        diameter = 2 * radius
        circle_diameters.append(diameter)

    # Kaç tane sınır bulunduğunu yazdır
    print(f"Toplam {len(circle_diameters)} tane sınır bulundu.")

    return circle_diameters


def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Unsharp masking yöntemi ile görüntüyü keskinleştirir."""
    # Gaussian blur ile bulanık görüntü oluşturma
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)

    # Orijinal görüntü ile bulanık görüntü arasındaki farkı hesaplama
    sharpened = float(amount + 1) * image - float(amount) * blurred

    # Keskinleştirilmiş görüntüyü normalize etme
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)

    # Eşikleme uygulama
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)

    return sharpened

if __name__ == '__main__':
    # origin = cv2.imread("./dataset/train/train/not_skin_cancer/not_skin_cancer_21.jpg")
    # image = cv2.imread("./dataset/train/train/not_skin_cancer/not_skin_cancer_21.jpg")
    origin = cv2.imread("./dataset/train/train/skin_cancer/skin_cancer_60.jpg")
    image = cv2.imread("././dataset/train/train/skin_cancer/skin_cancer_60.jpg")
    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    image1 = gri(image)
    image2 = cv2.equalizeHist(image1)
    # image2 = cv2.Laplacian(image, cv2.CV_64F)
    # image2 = cv2.convertScaleAbs(image2)
   # image2 = unsharp_mask(image2, kernel_size=(5, 5), sigma=1.0, amount=1.5, threshold=10)
    image3 = iso(image1)
    image4 = apply_median_filter(image3, 33)
    contours = find_contours(image4)
    draw_contours(image, contours)
    diameters = draw_circles(image, contours)


    # Görüntüleri aynı anda açma
    images = [origin, image1, image2, image3, image, image4]
    window_names = ["ILK HALI", "GRI HALI", "HISTOGRAM", "ISO HALI", "SINIRLI", "MEDIAN HALI"]

    for window_name, img in zip(window_names, images):
        image_show(img, window_name)



    cv2.waitKey(0)
    cv2.destroyAllWindows()

