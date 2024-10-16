import cv2
import pandas as pd
import numpy as np
def read_image_data():
    # Görüntü dosyasını oku
    image = cv2.imread("not_skin_cancer_00.jpg")

    # Görüntü verilerini Pandas DataFrame'e dönüştür
    data = image.reshape(-1, 3)
    df = pd.DataFrame(data, columns=['B', 'G', 'R'])

    return df


# İlk beş satırı görüntüle
if __name__ == '__main__':
    image_data = read_image_data()
    print(image_data.head())
