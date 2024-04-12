import cv2

def image_show():
    resim = cv2.imread("not_skin_cancer_00.jpg")
    cv2.imshow("KANSER ", resim)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    image_show()
