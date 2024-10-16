import cv2

# Görüntüyü yükle (gri tonlamalı ve histogram eşitlenmiş)
gri_goruntu = cv2.imread('./dataset/train/train/skin_cancer/skin_cancer_08.jpg', cv2.IMREAD_GRAYSCALE)
image = cv2.imread('./dataset/train/train/skin_cancer/skin_cancer_08.jpg')
gri_goruntu= cv2.resize(gri_goruntu, None, fx=5, fy=5, interpolation=cv2.INTER_LINEAR)

# ISODATA eşikleme
_, isodata_goruntu = cv2.threshold(gri_goruntu, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

 #Çerçeveli görüntüyü göster
cv2.imshow('ISODATA Goruntu', isodata_goruntu)


# Sobel filtresi uygula
sobel_goruntu_x = cv2.Sobel(gri_goruntu, cv2.CV_64F, 1, 0, ksize=5)
sobel_goruntu_y = cv2.Sobel(gri_goruntu, cv2.CV_64F, 0, 1, ksize=5)
sobel_goruntu = cv2.magnitude(sobel_goruntu_x, sobel_goruntu_y)

# Kenarları bul ve çerçevele
kenarlar = cv2.Canny(sobel_goruntu.astype('uint8'), 100, 500)
kontur, _ = cv2.findContours(kenarlar, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Çerçeveli görüntüyü çiz
for cnt in kontur:
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(gri_goruntu, (x, y), (x + w, y + h), (0, 255, 0), 1)


image= cv2.resize(image, None, fx=3, fy=3, interpolation=cv2.INTER_LINEAR)


# Çerçeveli görüntüyü göster
cv2.imshow('Cerceveli Goruntu', gri_goruntu)
cv2.imshow('Orijinal Goruntu', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
