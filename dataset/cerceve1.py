import cv2

# Renkli görüntüyü yükle
renkli_goruntu = cv2.imread('train/train/skin_cancer/skin_cancer_10.jpg')

# Gri tonlamalı görüntüyü yükle
gri_goruntu = cv2.imread('train/train/skin_cancer/skin_cancer_10.jpg', cv2.IMREAD_GRAYSCALE)

# Kenar tespiti yap
kenarlar = cv2.Canny(gri_goruntu, 100, 200)

# Kenarları bul ve çerçevele
kontur, _ = cv2.findContours(kenarlar, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Çerçeveyi renkli görüntü üzerinde çiz
for cnt in kontur:
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(renkli_goruntu, (x, y), (x + w, y + h), (0, 255, 0), 1)

# Çerçeveli görüntüyü göster
cv2.imshow('Cerceveli Goruntu', renkli_goruntu)
cv2.waitKey(0)
cv2.destroyAllWindows()
