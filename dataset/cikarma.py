import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils import gauss_kernel_olustur, konvolusyon, median_filtreleme
foto = cv2.imread("train/train/skin_cancer/skin_cancer_10.jpg", 0)

foto = cv2.equalizeHist(foto)
kernel = np.ones((5,5),np.uint8)

closing = cv2.morphologyEx(foto, cv2.MORPH_CLOSE, kernel)
cv2.imshow('closing', closing)
cv2.waitKey(0)
cv2.destroyAllWindows()

# opening = cv2.morphologyEx(foto, cv2.MORPH_OPEN, kernel)
# foto = cv2.erode(foto,kernel,iterations=0)
# foto = cv2.dilate(foto,kernel,iterations=1)
# foto = cv2.erode(foto,kernel,iterations=1)


print(foto.shape)
plt.imshow(foto, cmap="gray")
plt.show()

esik_degeri = 90
esik_maske = foto > esik_degeri

plt.imshow(esik_maske, cmap="gray")
plt.show()

medyan_filtre_boyutlari = [3, 5, 7]
medyan_cikis = []
for medyan_filtre_boyutu in medyan_filtre_boyutlari:
    m = n = medyan_filtre_boyutu
    medyan_cikis.append(median_filtreleme(foto, m, n))

print(medyan_cikis.shape)

plt.imshow(medyan_cikis, cmap="gray")
plt.show()

bulanik_esik_maske = medyan_cikis > esik_degeri

plt.imshow(bulanik_esik_maske, cmap="gray")
plt.show()

#cv2.imshow('erode', erosion)

cv2.waitKey(0)
cv2.destroyAllWindows()