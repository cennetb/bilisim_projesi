import cv2
import matplotlib.pyplot as plt
import numpy as np

def main():
    foto = cv2.imread("./dataset/train/train/skin_cancer/skin_cancer_10.jpg", 0)
    hist_es_foto = cv2.equalizeHist(foto)

    yanyana = np.hstack((foto, hist_es_foto))

    plt.imshow(yanyana, cmap='gray')
    plt.show()



if __name__ == "__main__":
    main()


