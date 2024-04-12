import cv2
import numpy as np

foto = cv2.imread("./dataset/train/train/not_skin_cancer/not_skin_cancer_01.jpg")

print(foto.shape)
#opencv B G R sırasında tutuyor renk değerlerini = kanal 0=blue 1=green 2=red

# x = 50
# y = 25
# kanal = 2
#
# yogunluk = foto[y, x, kanal]
# print("Yoğunluk: ",yogunluk)
#
# #numpy ile max min değerlerini bulabiliryoruz
# max_yogunluk = np.max(foto)
# min_yogunluk = np.min(foto)
# print("Max yoğunluğu:", max_yogunluk)
# print("Min yoğunluğu:", min_yogunluk)
#
# #fotoğrafın kanal değerini girmediğmiz için bütün kananllarda yoğunluk değerini verdi
# #print(foto[y, x])
#
# #fotoğrafı kırpma işlemi
# crop = foto[25:194, 50:259]

mavi_kanali = foto[:, :, 0]
print("Fotoğrafın boyutu: ", foto.shape)
print("Mavi kanalının boyutu: ", mavi_kanali.shape)

cv2.imshow("Fotoğraf",mavi_kanali )
cv2.waitKey(0)
cv2.destroyAllWindows()

#pixel değerlerini yazdırma
crop = foto[25:40, 25:40, 1]
print(crop)