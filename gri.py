import cv2
import numpy as np

# Renkli görüntüyü yükle
renkli_goruntu = cv2.imread('./dataset/train/train/skin_cancer/skin_cancer_65.jpg')
kernel = np.ones((10,10),np.uint8)
# Görüntüyü gri tonlamalı hale dönüştür
gri_tonlamali_goruntu = cv2.cvtColor(renkli_goruntu, cv2.COLOR_BGR2GRAY)

# Gri tonlamalı görüntüyü göster
gri_tonlamali_goruntu = cv2.resize(gri_tonlamali_goruntu, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
cv2.imshow('Gri Tonlamalı Görüntü', gri_tonlamali_goruntu)

opening = cv2.morphologyEx(gri_tonlamali_goruntu, cv2.MORPH_OPEN, kernel)
#closing = cv2.morphologyEx(gri_tonlamali_goruntu, cv2.MORPH_CLOSE, kernel)
#gradyan =cv2.morphologyEx(gri_tonlamali_goruntu,cv2.MORPH_GRADIENT,kernel)
#cv2.imshow('Gri', gradyan)
cv2.imshow('opening', opening)
#cv2.imshow('Closing', closing)
cv2.waitKey(0)
cv2.destroyAllWindows()