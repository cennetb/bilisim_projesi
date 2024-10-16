import cv2

# Görüntüyü yükle (gri tonlamalı ve histogram eşitlenmiş)
gri_goruntu = cv2.imread('train/train/skin_cancer/skin_cancer_45.jpg')

# Kenar tespiti yap
kenarlar = cv2.Canny(gri_goruntu, 100, 500)

# Kenarları bul ve çerçevele
kontur, _ = cv2.findContours(kenarlar, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in kontur:
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(gri_goruntu, (x, y), (x + w, y + h), (0, 255, 0), 1)

# Çerçeveli görüntüyü göster
gri_goruntu= cv2.resize(gri_goruntu, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
cv2.imshow('Cerceveli Goruntu', gri_goruntu)
cv2.waitKey(0)
cv2.destroyAllWindows()
