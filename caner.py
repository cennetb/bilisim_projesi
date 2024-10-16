import cv2

def gri(image):
    """Görüntü üzerinde bir işlem yap ve yeni bir görüntü döndür."""
    # Örnek işlem: Görüntüyü gri tonlamaya çevir
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_median_filter(image, kernel_size):
    """Median filtreleme uygula."""
    filtered_image = cv2.medianBlur(image, kernel_size)
    return filtered_image

def iso(image):
    """Görüntüyü ikili hale getirme (Isodata thresholding)."""
    _, isodata_goruntu = cv2.threshold(image, 254, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return isodata_goruntu

def find_contours(image):
    """Görüntüdeki şekillerin sınırlarını bulur ve koordinatlarını döndürür."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    _, binary = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def draw_contours(image, contours):
    """Görüntüdeki şekillerin sınırlarını çiz ve ekranda göster."""
    cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

def apply_gaussian_blur(image_path, kernel_size=(17, 17)):
    # Resmi yükle
    image = cv2.imread(image_path)
    if image is None:
        print("Resim yüklenemedi.")
        return

    # Gaussian Blur filtresini uygula
    blurred_image = cv2.GaussianBlur(image, kernel_size, 0)

if __name__ == '__main__':
    # Resim yolu ve adını belirleme
    image_path = "./dataset/train/train/skin_cancer/caner.jpg"
    image = cv2.imread(image_path)
    gray = gri(image)
    # gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_LINEAR)
    blurred_image = cv2.GaussianBlur(gray, (17, 17), 0)
    isoo = iso(blurred_image)
    # contours, _ = cv2.findContours(isoo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(image, contours, -1, (0, 255, 0), 4)
    contours = find_contours(isoo)
    draw_contours(image, contours)


    cv2.imshow("son", image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()