import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import konvolusyon, median_filtreleme, gauss_kernel_olustur

foto = cv2.imread("./dataset/train/train/skin_cancer/skin_cancer_06.jpg", 0)

plt.imshow(foto, cmap="gray")
plt.show()

sigmalar = [1, 3, 5]
gauss_cikis = []
for sigma in sigmalar:
    m = n = 6*sigma + 1
    gauss_kernel = gauss_kernel_olustur(m, n, K=1, sigma=sigma)
    gauss_cikis.append(konvolusyon(foto, gauss_kernel))

medyan_filtre_boyutlari = [3, 5, 7]
medyan_cikis = []
for medyan_filtre_boyutu in medyan_filtre_boyutlari:
    m = n = medyan_filtre_boyutu
    medyan_cikis.append(median_filtreleme(foto, m, n))

foto_boyutu = (448, 448)

#ilk_sira = np.hstack([cv2.resize(cikis, foto_boyutu) for cikis in [foto, *gauss_cikis]])
ikinci_sira = np.hstack([cv2.resize(cikis, foto_boyutu) for cikis in [foto, *medyan_cikis]])

altalta = np.vstack((ikinci_sira))


plt.imshow(altalta, cmap="gray")
plt.show()